import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import chain
from random import choice
from typing import Optional

import numpy as np
import torch
import transformers
import wandb
from datasets import concatenate_datasets, load_from_disk
from torch import _dynamo as dynamo
from torch.nn import CrossEntropyLoss
from torch.utils import cpp_extension
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from utils.templates import construct_cell_type_template, construct_prediction_template

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

if "LOCAL_RANK" in os.environ:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
else:
    LOCAL_RANK = 0

logger = logging.getLogger(__name__)


@dataclass
class CustomTrainingArguments:
    model_name: str = field(
        default="gpt2", metadata={"help": "Hugging Face model name."}
    )
    seed: int = field(
        default=42, metadata={"help": "Seed for shuffling training dataset."}
    )
    data_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Data seed for Hugging Face's trainer pipeline."},
    )
    set_torch_seed_manually: bool = field(
        default=False, metadata={"help": "Seed for PyTorch."}
    )
    torch_cuda_seed: int = field(
        default=42, metadata={"help": "Seed for PyTorch CUDA."}
    )
    eval_dataset_size: int = field(
        default=1000,
        metadata={"help": "Number of samples to use from evaluation dataset."},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Whether to evaluate on steps, epochs, or none."},
    )
    eval_steps: int = field(
        default=100,
        metadata={
            "help": "If evaluation_strategy is set to 'steps', will evaluate every number of steps here."
        },
    )
    eval_accumulation_steps: int = field(
        default=5,
        metadata={"help": "Number of evaluation steps before offloading to CPU."},
    )
    output_dir: str = field(
        default="<OUTPUT_DIRECTORY>",
        metadata={"help": "Output directory for training runs."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite output directory if nonempty."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={
            "help": "Whether to save model checkpoints on steps, epochs, or none."
        },
    )
    save_steps: int = field(
        default=500,
        metadata={
            "help": "If save_strategy is set to 'steps', will save model checkpoint every number of steps here."
        },
    )
    save_total_limit: int = field(
        default=100,
        metadata={
            "help": "Maximum number of model checkpoints saved in output directory."
            " Will overwrite earlier checkpoints if limit is exceeded."
        },
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Per device batch size used during training."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Per device batch size used during evaluation."}
    )
    num_train_epochs: int = field(
        default=5, metadata={"help": "Number of training epochs."}
    )
    wandb_project_name: str = field(
        default="cell2sentence", metadata={"help": "Wandb project name to save to."}
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint if resuming training."},
    )
    torch_compile: bool = field(
        default=False, metadata={"help": "Whether to use torch compile."}
    )
    torchdynamo: Optional[str] = field(
        default=None, metadata={"help": "Backend compiler for torch dynamo."}
    )
    torch_compile_backend: Optional[str] = field(
        default=None, metadata={"help": "Backend compiler for torch compile."}
    )
    dynamo_cache_size_limit: int = field(
        default=64, metadata={"help": "Number of graphs to cache for torch compile."}
    )
    dynamo_verbose: bool = field(
        default=False, metadata={"help": "Make dynamo config set to verbose."}
    )
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16."})
    ddp_backend: str = field(
        default="nccl", metadata={"help": "Backend for distributed data parallelism."}
    )
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of workers to use for dataloader."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Whether to checkpoint gradients during training. Improves GPU memory consumption."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of forward passes before backpropagation during training. Per device."
        },
    )
    logging_steps: int = field(
        default=100,
        metadata={
            "help": "Number of training steps before logging, where steps is the number of gradient unpdates."
        },
    )
    datasets_paths: str = field(
        default="<PATH_TO_DATASET_PATHS_JSON>",
        metadata={"help": "Path to json file where datasets are located."},
    )
    wandb_logging: bool = field(
        default=False, metadata={"help": "Whether to log to wandb."}
    )
    wandb_run_base_name: str = field(
        default="pbmc_finetune",
        metadata={"help": "Base name for wandb run. Start time will be appended."},
    )
    log_level: str = field(default="debug", metadata={"help": "Log level to use."})
    optim: str = field(
        default="adamw_torch",
        metadata={
            "help": "Optimizer to use. See Hugging Face options in TrainerArguments."
        },
    )
    deepspeed: str = field(
        default=None,
        metadata={"help": "Whether to use deepspeed for distributed training."},
    )


def main():
    if LOCAL_RANK == 0:
        logger.info(f"\nCUDA HOME: {cpp_extension.CUDA_HOME}")
        logger.info(f"\nTORCH CUDA VERSION: {torch.version.cuda}")

    logger.info(f"\nLOCAL RANK: {LOCAL_RANK}")

    assert torch.cuda.is_available(), "CUDA unavailable"

    device = torch.device("cuda")

    parser = HfArgumentParser((CustomTrainingArguments,))
    training_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    training_args_dict = asdict(training_args)

    if LOCAL_RANK == 0:
        logger.info(json.dumps(training_args_dict, indent=2))

    log_level = LOG_LEVELS[training_args.log_level]
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if training_args.set_torch_seed_manually:
        torch.cuda.manual_seed(training_args.torch_cuda_seed)
        logger.info(
            f"\nSET TORCH CUDA SEED MANUALLY. SEED VALUE: {training_args.torch_cuda_seed}"
        )

    if training_args.torch_compile:
        dynamo.config.cache_size_limit = training_args.dynamo_cache_size_limit
        dynamo.config.verbose = training_args.dynamo_verbose
    else:
        training_args.torchdynamo = None
        training_args.torch_compile_backend = None

    # Get train and val dataset paths from json file and load datasets
    with open(training_args.datasets_paths, "r") as f:
        file_type_to_name = json.load(f)

    train_dataset = load_from_disk(file_type_to_name["train"])
    train_dataset = train_dataset.shuffle(seed=training_args.seed)

    val_dataset = load_from_disk(file_type_to_name["val"])
    val_dataset = val_dataset.shuffle(seed=training_args.seed)

    if LOCAL_RANK == 0:
        logger.info(f"\nLENGTH OF TRAIN DATASET: {len(train_dataset)}")
        logger.info(train_dataset)

        logger.info(f"\nLENGTH OF EVAL DATASET: {len(val_dataset)}")
        logger.info(val_dataset)

    # Instantiate tokenizer and model for finetuning
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(training_args.model_name).to(device)

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Get current time and initialize wandb
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"{training_args.model_name}-{training_args.wandb_run_base_name}-{now}"

    if training_args.wandb_logging:
        if LOCAL_RANK == 0:
            wandb.init(project=training_args.wandb_project_name, name=run_name)
            wandb.watch(model, log="all", log_freq=10)

    def preprocess_function(examples):
        text_column = "cell_type"
        label_column = "input_ids"
        max_length = 1024

        batch_size = len(examples[text_column])
        inputs = []
        targets = []
        for i in range(batch_size):
            prompt_type = choice([0, 1, 2])
            if prompt_type == 0:
                input = construct_cell_type_template(examples["cell_type"][i])
                target = " ".join(examples["input_ids"][i].split(" ")[:100])
            elif prompt_type == 1:
                input = construct_cell_type_template("PBMC")
                target = " ".join(examples["input_ids"][i].split(" ")[:100])
            else:
                input = construct_prediction_template(
                    " ".join(examples["input_ids"][i].split(" ")[:100])
                )
                target = examples["cell_type"][i]

            inputs.append(input)
            targets.append(target)

        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.select(range(training_args.eval_dataset_size))

    # Collate function for training.
    def data_collator(examples):
        max_length = 0
        for i in range(len(examples)):
            input_length = len(examples[i]["input_ids"])
            max_length = max(max_length, input_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for i in range(len(examples)):
            sample_input_ids = examples[i]["input_ids"]
            label_input_ids = examples[i]["labels"]
            attention_mask = examples[i]["attention_mask"]

            final_input_ids = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            final_attention_mask = [0] * (
                max_length - len(sample_input_ids)
            ) + attention_mask
            final_label_input_ids = [-100] * (
                max_length - len(sample_input_ids)
            ) + label_input_ids

            batch_input_ids.append(final_input_ids)
            batch_attention_mask.append(final_attention_mask)
            batch_labels.append(final_label_input_ids)

        return {
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_mask),
            "labels": torch.tensor(batch_labels),
        }

    # Configure Trainer and start training
    output_dir = training_args.output_dir + f"/{run_name}"

    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=training_args.overwrite_output_dir,
        seed=training_args.seed,
        data_seed=training_args.data_seed,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        evaluation_strategy=training_args.evaluation_strategy,
        eval_steps=training_args.eval_steps,
        eval_accumulation_steps=training_args.eval_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        report_to="wandb",
        torch_compile=training_args.torch_compile,
        torchdynamo=training_args.torchdynamo,
        torch_compile_backend=training_args.torch_compile_backend,
        fp16=training_args.fp16,
        ddp_backend=training_args.ddp_backend,
        dataloader_num_workers=training_args.dataloader_num_workers,
        gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        logging_steps=training_args.logging_steps,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        optim=training_args.optim,
        deepspeed=training_args.deepspeed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    if LOCAL_RANK == 0:
        logger.info(f"\nDEEPSPEED ENABLED: {trainer.is_deepspeed_enabled}")
        logger.info(f"\nFINAL TRAINING ARGUMENTS: {trainer.args}")
    train_result = trainer.train(resume_from_checkpoint=training_args.checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
