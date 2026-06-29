import transformers
import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer

from modeling_gmsa import DataArguments, GMSA, ModelArguments, TrainingArguments
from training_utils import AutoEncodingTokenizeFunction, train_model


def _limit(dataset, sample_count, debug):
    limit = 32 if debug else sample_count
    if limit < 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.stage = "autoencoding"
    model_args.train = True
    training_args.project_name = "gmsa_nq_ae"
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={"train": data_args.train_file, "eval": data_args.test_file},
    )
    train_dataset = _limit(
        dataset["train"], data_args.train_samples, data_args.debug_data
    )
    eval_dataset = _limit(
        dataset["eval"], data_args.eval_samples, data_args.debug_data
    )
    print(f"Dataset size: train={len(train_dataset)}, eval={len(eval_dataset)}")

    if training_args.local_rank <= 0:
        suffix = "debug" if data_args.debug_data else str(training_args.max_steps)
        wandb.init(
            project=training_args.project_name,
            name=f"gmsa_ae_{suffix}_m{model_args.merge_size}",
        )

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    tokenize = AutoEncodingTokenizeFunction(
        tokenizer, training_args.model_max_length
    )
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=1000)
    eval_dataset = eval_dataset.map(tokenize, batched=True, batch_size=1000)

    model = GMSA(model_args, training_args, lora_config)
    train_model(model, train_dataset, eval_dataset, training_args, tokenizer)


if __name__ == "__main__":
    main()
