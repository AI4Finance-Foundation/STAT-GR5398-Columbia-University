
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import sys
from functools import partial
from pathlib import Path

import datasets
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import (
    SUPPORTED_MODEL_FAMILIES,
    lora_module_dict,
    tokenize,
    write_adapter_manifest,
    validate_lora_target_modules,
)


DEFAULT_DATASET = "FinGPT/fingpt-forecaster-dow30-202305-202405"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REQUIRED_COLUMNS = {"prompt", "answer", "label", "symbol", "period"}


def cuda_memory_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    gib = 1024**3
    device = torch.cuda.current_device()
    return {
        "gpu_memory_allocated_gib": round(torch.cuda.memory_allocated(device) / gib, 4),
        "gpu_memory_reserved_gib": round(torch.cuda.memory_reserved(device) / gib, 4),
        "gpu_memory_peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device) / gib, 4
        ),
        "gpu_memory_peak_reserved_gib": round(
            torch.cuda.max_memory_reserved(device) / gib, 4
        ),
    }


class MonitoringTrainer(Trainer):
    """Add CUDA memory measurements to normal Trainer logs."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        enriched_logs = dict(logs)
        enriched_logs.update(cuda_memory_metrics())
        super().log(enriched_logs, start_time=start_time)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    assignment_dir = script_dir.parent if script_dir.name == "source_code" else script_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-family",
        choices=("llama3", "deepseek", "qwen2.5"),
        help="Normally inferred from --model-name; set only for a compatible custom path.",
    )
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset_name",
        default=DEFAULT_DATASET,
    )
    parser.add_argument("--test-dataset")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=assignment_dir / "pretrained-models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=assignment_dir / "finetuned_models" / "qwen_baseline",
    )
    parser.add_argument("--run-name", default="qwen_baseline")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--resume-from-checkpoint")

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--scheduler", default="constant")
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit", "8bit"),
        default="4bit",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="no")
    parser.add_argument("--eval-steps", type=float, default=0.1)
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="epoch")
    parser.add_argument("--save-steps", type=float, default=0.1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--report-to",
        default="tensorboard",
        help="Comma-separated integrations: tensorboard, wandb, or none.",
    )
    parser.add_argument("--ds-config", type=Path)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=-1)

    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate one collated batch without training or saving an adapter.",
    )
    return parser.parse_args()


def load_dataset_source(name: str, cache_dir: Path, token: str | None):
    path = Path(name).expanduser()
    if path.exists():
        if (path / "dataset_dict.json").is_file():
            return datasets.load_from_disk(path)
        if path.is_dir():
            data_files = {
                split: str(path / f"{split}.parquet")
                for split in ("train", "test")
                if (path / f"{split}.parquet").is_file()
            }
            if data_files:
                return datasets.load_dataset(
                    "parquet",
                    data_files=data_files,
                    cache_dir=str(cache_dir / "datasets"),
                )
        raise ValueError(f"Unsupported local dataset path: {path}")

    return datasets.load_dataset(
        name,
        cache_dir=str(cache_dir / "datasets"),
        token=token,
    )


def validate_dataset(dataset, source_name: str) -> None:
    if "train" not in dataset or "test" not in dataset:
        raise ValueError(
            f"{source_name} must provide existing train and test splits; no replacement split "
            "will be created."
        )
    for split in ("train", "test"):
        missing = REQUIRED_COLUMNS.difference(dataset[split].column_names)
        if missing:
            raise ValueError(f"{source_name} {split} split is missing columns: {sorted(missing)}")


def select_samples(split, maximum: int | None):
    if maximum is None:
        return split
    if maximum <= 0:
        raise ValueError("Sample limits must be positive integers.")
    return split.select(range(min(maximum, len(split))))


def infer_model_family(model_name: str, config, explicit_family: str | None) -> str:
    if explicit_family:
        return explicit_family
    if model_name in SUPPORTED_MODEL_FAMILIES:
        return SUPPORTED_MODEL_FAMILIES[model_name]

    normalized = model_name.lower()
    if config.model_type == "qwen2" or "qwen" in normalized:
        return "qwen2.5"
    if "deepseek" in normalized:
        return "deepseek"
    if config.model_type == "llama" or "llama" in normalized:
        return "llama3"
    raise ValueError(
        f"Cannot infer a supported model family for {model_name!r}; pass --model-family."
    )


def resolve_precision(name: str) -> tuple[str, torch.dtype]:
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            name = "bf16"
        elif torch.cuda.is_available():
            name = "fp16"
        else:
            name = "float32"
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }[name]
    return name, dtype


def build_quantization_config(mode: str, compute_dtype: torch.dtype):
    if mode == "none":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(f"--quantization {mode} requires an NVIDIA CUDA GPU.")
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def parse_report_to(value: str) -> list[str]:
    integrations = [item.strip() for item in value.split(",") if item.strip()]
    if not integrations or integrations == ["none"]:
        return []
    unsupported = set(integrations).difference({"tensorboard", "wandb"})
    if unsupported:
        raise ValueError(f"Unsupported reporting integrations: {sorted(unsupported)}")
    if "wandb" in integrations and not os.environ.get("WANDB_API_KEY"):
        raise EnvironmentError("Set WANDB_API_KEY before enabling WandB reporting.")
    return integrations


def validate_deepspeed_config(path: Path | None, quantization: str, optim: str):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"DeepSpeed configuration not found: {path}")
    config = json.loads(path.read_text())
    for key in (
        "train_micro_batch_size_per_gpu",
        "train_batch_size",
        "gradient_accumulation_steps",
    ):
        if config.get(key) != "auto":
            raise ValueError(f"DeepSpeed {key} must be 'auto' to match TrainingArguments.")
    stage = config.get("zero_optimization", {}).get("stage")
    if stage not in (0, 1, 2):
        raise ValueError("This pipeline supports DeepSpeed ZeRO stages 0, 1, or 2.")
    if quantization != "none":
        raise ValueError(
            "This pipeline does not combine DeepSpeed with bitsandbytes quantization. "
            "Use the default single-GPU QLoRA launcher, or set --quantization none for DeepSpeed."
        )
    if optim != "adamw_torch":
        raise ValueError("DeepSpeed runs must use --optim adamw_torch in this pipeline.")
    return config


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def environment_metadata() -> dict:
    gpu_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 1024**3, 3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "transformers_version": package_version("transformers"),
        "peft_version": package_version("peft"),
        "accelerate_version": package_version("accelerate"),
        "datasets_version": package_version("datasets"),
        "deepspeed_version": package_version("deepspeed"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_devices": gpu_devices,
    }


def prepare_split(split, args, tokenizer, description: str):
    original_count = len(split)
    mapped = split.map(
        partial(tokenize, args, tokenizer),
        remove_columns=split.column_names,
        desc=f"Tokenizing {description}",
    )
    mapped = mapped.filter(
        lambda feature: not feature["exceed_max_length"],
        desc=f"Filtering long {description} examples",
    )
    kept_count = len(mapped)
    if kept_count == 0:
        raise ValueError(
            f"Every {description} example exceeds --max-length {args.max_length}."
        )
    sequence_lengths = mapped["sequence_length"]
    mapped = mapped.remove_columns(["exceed_max_length", "sequence_length"])
    return mapped, {
        "original_count": original_count,
        "kept_count": kept_count,
        "filtered_count": original_count - kept_count,
        "minimum_sequence_length": min(sequence_lengths),
        "maximum_sequence_length": max(sequence_lengths),
    }


def ensure_output_directory(args) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite_output_dir and not args.resume_from_checkpoint:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. Use a new directory, "
                "--overwrite-output-dir, or --resume-from-checkpoint."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def write_run_config(args, details: dict) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(details)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_output_directory(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if args.model_name.startswith("meta-llama/") and not hf_token:
        raise EnvironmentError(
            "Set HF_TOKEN after receiving access to the gated Meta Llama repository."
        )

    report_to = parse_report_to(args.report_to)
    precision_name, compute_dtype = resolve_precision(args.precision)
    deepspeed_config = validate_deepspeed_config(
        args.ds_config,
        args.quantization,
        args.optim,
    )
    quantization_config = build_quantization_config(args.quantization, compute_dtype)

    config = AutoConfig.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    model_family = infer_model_family(args.model_name, config, args.model_family)
    target_modules = lora_module_dict[model_family]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither a padding token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset_source(args.dataset_name, args.cache_dir, hf_token)
    validate_dataset(dataset, args.dataset_name)
    train_split = select_samples(dataset["train"], args.max_train_samples)

    if args.test_dataset:
        test_source = load_dataset_source(args.test_dataset, args.cache_dir, hf_token)
        validate_dataset(test_source, args.test_dataset)
        test_split = test_source["test"]
    else:
        test_split = dataset["test"]
    test_split = select_samples(test_split, args.max_eval_samples)

    tokenized_train, train_stats = prepare_split(train_split, args, tokenizer, "train")
    tokenized_eval = None
    eval_stats = None
    if args.eval_strategy != "no":
        tokenized_eval, eval_stats = prepare_split(test_split, args, tokenizer, "test")

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "token": hf_token,
        "trust_remote_code": args.trust_remote_code,
        "dtype": compute_dtype,
    }
    if quantization_config is not None:
        process_index = args.local_rank if args.local_rank >= 0 else 0
        model_kwargs.update(
            quantization_config=quantization_config,
            device_map={"": process_index},
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    target_counts = validate_lora_target_modules(model, target_modules)
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    non_lora_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if non_lora_trainable:
        raise RuntimeError(f"Unexpected non-LoRA trainable parameters: {non_lora_trainable[:10]}")

    run_details = {
        "resolved_model_family": model_family,
        "resolved_precision": precision_name,
        "hf_token_available": bool(hf_token),
        "wandb_api_key_available": bool(os.environ.get("WANDB_API_KEY")),
        "reporting_integrations": report_to,
        "environment": environment_metadata(),
        "deepspeed": {
            "enabled": deepspeed_config is not None,
            "config": deepspeed_config,
        },
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "effective_batch_size": (
            args.batch_size
            * args.gradient_accumulation_steps
            * int(os.environ.get("WORLD_SIZE", "1"))
        ),
        "lora_target_modules": target_modules,
        "lora_target_instance_counts": target_counts,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": 100.0 * trainable_parameters / total_parameters,
        "train_dataset": train_stats,
        "eval_dataset": eval_stats,
    }
    if args.local_rank in (-1, 0):
        write_run_config(args, run_details)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=precision_name == "bf16",
        fp16=precision_name == "fp16",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        include_num_input_tokens_seen=True,
        report_to=report_to,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=str(args.ds_config) if args.ds_config else None,
        skip_memory_metrics=False,
        use_cache=False,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = MonitoringTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    if args.dry_run:
        batch = next(iter(trainer.get_train_dataloader()))
        print("Dry run passed.")
        print({key: tuple(value.shape) for key, value in batch.items()})
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a Linux NVIDIA CUDA environment for this assignment.")

    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = dict(train_result.metrics)
    device = torch.cuda.current_device()
    metrics.update(
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
    )
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        manifest = write_adapter_manifest(args.output_dir)
        print(
            f"Saved and validated LoRA adapter only at {args.output_dir} "
            f"({manifest['adapter_weights_file']}, {manifest['adapter_weights_bytes']} bytes)."
        )


if __name__ == "__main__":
    main(parse_args())
