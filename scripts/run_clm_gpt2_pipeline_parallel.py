#!/usr/bin/env python
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import chain

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from datasets import IterableDataset, IterableDatasetDict, load_dataset
from torch.profiler import ProfilerActivity

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.engine import PipelineEngine


# ----- versions -----
check_min_version("4.0.0")
require_version("datasets>=2.14.0", "pip install datasets>=2.14.0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def debug_dist_state(tag=""):
    import socket
    hostname = socket.gethostname()
    env_keys = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS",
        "MASTER_ADDR", "MASTER_PORT",
    ]
    env = {k: os.environ.get(k, None) for k in env_keys}
    print(f"\n[{tag}] HOST={hostname} PID={os.getpid()} ENV={env}", flush=True)

    print(f"[{tag}] dist_available={dist.is_available()} initialized={dist.is_initialized()}", flush=True)
    if dist.is_initialized():
        print(
            f"[{tag}] RANK={dist.get_rank()} WORLD_SIZE={dist.get_world_size()} BACKEND={dist.get_backend()}",
            flush=True,
        )


def split_streaming_dataset(full_streaming_dataset, validation_percentage: int = 5) -> IterableDatasetDict:
    if not (0 < validation_percentage < 100):
        raise ValueError("validation_percentage must be between 0 and 100 (exclusive)")

    def split_generator(is_train: bool):
        for i, example in enumerate(full_streaming_dataset):
            if is_train:
                if i % 100 > validation_percentage:
                    yield example
            else:
                if i % 100 < validation_percentage:
                    yield example

    features = full_streaming_dataset.features
    train_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": True}, features=features)
    validation_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": False}, features=features)
    return IterableDatasetDict({"train": train_stream, "validation": validation_stream})


def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# -----------------------
# Pipeline conversion
# -----------------------
def make_gpt2_pipeline_module(hf_model, tie_embeddings=True):
    """
    Convert HF GPT2LMHeadModel into a DeepSpeed PipelineModule.
    Expects input from stage0 as (input_ids, attention_mask, labels).
    Returns scalar loss (Finalize computes loss) for training.
    """
    gpt2 = hf_model.transformer
    wte = gpt2.wte
    wpe = gpt2.wpe
    drop = gpt2.drop
    blocks = gpt2.h
    ln_f = gpt2.ln_f
    lm_head = hf_model.lm_head

    if tie_embeddings:
        try:
            lm_head.weight = wte.weight
        except Exception:
            pass

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = wte
            self.wpe = wpe
            self.drop = drop

        def forward(self, inputs):
            input_ids, attention_mask, labels = inputs
            bsz, seqlen = input_ids.shape
            device = input_ids.device

            position_ids = torch.arange(0, seqlen, device=device).unsqueeze(0).expand(bsz, -1)
            x = self.wte(input_ids) + self.wpe(position_ids)
            x = self.drop(x)

            if attention_mask is None:
                attn = None
            else:
                attn = (1.0 - attention_mask.float()) * -1e4
                attn = attn[:, None, None, :]

            return (x, attn, labels)

    class GPT2BlockWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, inputs):
            x, attn, labels = inputs
            out = self.block(x, attention_mask=attn, use_cache=False)
            x = out[0]
            return (x, attn, labels)

    class Finalize(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_f = ln_f
            self.lm_head = lm_head

        def forward(self, inputs):
            x, _attn, labels = inputs
            x = self.ln_f(x)
            logits = self.lm_head(x)  # (bsz, seqlen, vocab)

            if labels is None:
                return logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss

    layers = [LayerSpec(Embeddings)]
    for blk in blocks:
        layers.append(LayerSpec(GPT2BlockWrapper, blk))
    layers.append(LayerSpec(Finalize))

    # Explicit num_stages = world_size makes behavior deterministic under Slurm
    num_stages = dist.get_world_size() if dist.is_initialized() else 1

    pipe = PipelineModule(
        layers=layers,
        loss_fn=None,  # loss returned by Finalize
        num_stages=num_stages,
        partition_method="parameters",
        activation_checkpoint_interval=0,
    )
    return pipe


# -----------------------
# HF arg dataclasses
# -----------------------
@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default=None)
    model_type: str | None = field(default=None)
    config_overrides: str | None = field(default=None)
    config_name: str | None = field(default=None)
    tokenizer_name: str | None = field(default=None)
    cache_dir: str | None = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: str = field(default=None)
    trust_remote_code: bool = field(default=False)
    dtype: str | None = field(default=None)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used with --config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    dataset_name: str | None = field(default=None)
    dataset_config_name: str | None = field(default=None)
    train_file: str | None = field(default=None)
    validation_file: str | None = field(default=None)
    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)
    streaming: bool = field(default=False)
    block_size: int | None = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: int | None = field(default=5)
    preprocessing_num_workers: int | None = field(default=None)
    keep_linebreaks: bool = field(default=True)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.train_file is not None:
            ext = self.train_file.split(".")[-1]
            assert ext in ["csv", "json", "txt"], "`train_file` should be csv/json/txt"
        if self.validation_file is not None:
            ext = self.validation_file.split(".")[-1]
            assert ext in ["csv", "json", "txt"], "`validation_file` should be csv/json/txt"


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Init logging early
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # IMPORTANT: init dist BEFORE building PipelineModule (we use world_size)
    # Under Slurm, torch.distributed.run isn't used, so we must init here.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    debug_dist_state("startup")

    set_seed(training_args.seed)

    # --------------------
    # Load dataset(s)
    # --------------------
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file

        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )

        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    extension,
                    data_files=data_files,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

    # --------------------
    # Load config/tokenizer/HF model
    # --------------------
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        if model_args.config_overrides is not None:
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    # GPT-2 has no pad token; for batching, set to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=dtype if dtype != "auto" else None,
    )

    # Resize embeddings if tokenizer grew
    embedding_size = hf_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        hf_model.resize_token_embeddings(len(tokenizer))

    # --------------------
    # Tokenize + group texts
    # --------------------
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning("Long input will be chunked into smaller bits")
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing",
            )
        else:
            tokenized = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names)

    max_pos_embeddings = getattr(config, "max_position_embeddings", 1024)
    if data_args.block_size is None:
        block_size = min(tokenizer.model_max_length, max_pos_embeddings, 1024)
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        # create attention mask (all ones; no padding in grouped blocks)
        result["attention_mask"] = [[1] * block_size for _ in range(len(result["input_ids"]))]
        return result

    with training_args.main_process_first(desc="grouping texts"):
        if not data_args.streaming:
            lm_datasets = tokenized.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping into {block_size}",
            )
        else:
            lm_datasets = tokenized.map(group_texts, batched=True)

    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples is not None:
        if data_args.streaming:
            train_dataset = train_dataset.take(data_args.max_train_samples)
        else:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    # --------------------
    # Build PipelineModule from HF model
    # --------------------
    pipe_model = make_gpt2_pipeline_module(hf_model, tie_embeddings=True)

    if training_args.deepspeed is None:
        raise ValueError("PP requires --deepspeed config with pipeline.enabled and pipeline.stages")

    # --------------------
    # DeepSpeed initialize
    # --------------------
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=pipe_model,
        model_parameters=[p for p in pipe_model.parameters() if p.requires_grad],
        config=training_args.deepspeed,
    )

    is_pipe = isinstance(model_engine, PipelineEngine)

    if _is_main_process():
        print(f"DeepSpeed engine type: {type(model_engine)}")
        print(f"Pipeline enabled: {is_pipe}")

    if is_pipe:
        if _is_main_process():
            print(
                f"PP stage id={model_engine.stage_id} "
                f"num_stages={model_engine.num_stages} "
                f"is_first={model_engine.is_first_stage()} "
                f"is_last={model_engine.is_last_stage()}"
            )

    # --------------------
    # Training loop (PipelineEngine)
    # --------------------
    if training_args.do_train:
        from torch.utils.data import DataLoader

        # Only FIRST stage loads data
        if is_pipe and model_engine.is_first_stage():
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                shuffle=not data_args.streaming,
                collate_fn=default_data_collator,
                drop_last=True,
            )
            data_iter = iter(train_dataloader)
        else:
            train_dataloader = None
            data_iter = None

        model_engine.train()
        total_loss = 0.0
        steps = 0
        start_time = time.time()

        # estimate steps if dataset is sized; else just loop a fixed number of steps
        if not data_args.streaming:
            steps_per_epoch = len(train_dataloader) if train_dataloader is not None else 0
            total_steps = steps_per_epoch * int(training_args.num_train_epochs)
        else:
            # streaming: you probably want max_train_steps; fallback
            total_steps = getattr(training_args, "max_steps", 1000)
            if total_steps <= 0:
                total_steps = 1000

        for step in range(1, total_steps + 1):
            # PipelineEngine handles forward/backward/step internally
            loss = model_engine.train_batch(data_iter=data_iter)

            # loss is only meaningful on last stage; others may get None
            if loss is not None:
                total_loss += float(loss)
                steps += 1

            if training_args.logging_steps and step % training_args.logging_steps == 0:
                if _is_main_process():
                    avg_loss = total_loss / max(1, steps)
                    elapsed = time.time() - start_time
                    print(f"step={step} avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s", flush=True)

        # Save checkpoint (each stage saves its shard)
        os.makedirs(training_args.output_dir, exist_ok=True)
        model_engine.save_checkpoint(training_args.output_dir)

        if _is_main_process():
            tokenizer.save_pretrained(training_args.output_dir)
            try:
                hf_model.config.save_pretrained(training_args.output_dir)
            except Exception:
                pass

            metrics = {
                "train_loss": total_loss / max(1, steps),
                "train_steps_with_loss": steps,
                "train_runtime_sec": time.time() - start_time,
            }
            print("TRAIN METRICS:", metrics, flush=True)

    # Clean shutdown
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
