import json
import os
from pathlib import Path

import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import HfArgumentParser

from eval_utils import score_prediction
from modeling_gmsa import DataArguments, GMSA, ModelArguments, TrainingArguments
from training_utils import build_context_prompt_ids


CHECKPOINT_FILES = (
    "model.safetensors",
    "adapter_model.safetensors",
    "pytorch_model.bin",
)


def resolve_checkpoint(path):
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        return str(checkpoint)

    for filename in CHECKPOINT_FILES:
        candidate = checkpoint / filename
        if candidate.is_file():
            return str(candidate)

    checkpoint_dirs = []
    for child in checkpoint.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                checkpoint_dirs.append((int(child.name.rsplit("-", 1)[1]), child))
            except ValueError:
                continue
    for _, directory in sorted(checkpoint_dirs, reverse=True):
        for filename in CHECKPOINT_FILES:
            candidate = directory / filename
            if candidate.is_file():
                return str(candidate)
    return str(checkpoint)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_batch(items, tokenizer, max_length, device):
    encoded = [
        build_context_prompt_ids(
            tokenizer, item["input"], item["prompt"], max_length
        )
        for item in items
    ]
    max_sequence_length = max(len(input_ids) for input_ids, _ in encoded)
    input_ids = torch.full(
        (len(items), max_sequence_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, (ids, mask) in enumerate(encoded):
        sequence_length = len(ids)
        input_ids[index, :sequence_length] = torch.tensor(ids, device=device)
        attention_mask[index, :sequence_length] = True
        prompt_mask[index, :sequence_length] = torch.tensor(mask, device=device)
    return input_ids, attention_mask, prompt_mask


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.train = False
    model_args.stage = "finetune"
    training_args.deepspeed = None
    if data_args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GMSA(model_args, training_args, lora_config)
    if training_args.restore_from:
        checkpoint = resolve_checkpoint(training_args.restore_from)
        print(f"Loading checkpoint: {checkpoint}")
        model._load_checkpoint(checkpoint)
    else:
        print("No checkpoint provided; using base model weights.")
    model.to(device)
    model.eval()

    items = load_jsonl(data_args.test_file)
    if data_args.eval_samples > 0:
        items = items[: data_args.eval_samples]
    if not items:
        raise ValueError(f"No test samples loaded from: {data_args.test_file}")

    os.makedirs(data_args.eval_output_dir, exist_ok=True)
    result_path = os.path.join(
        data_args.eval_output_dir, "nq_inference_results.jsonl"
    )
    metrics_path = os.path.join(
        data_args.eval_output_dir, "nq_inference_metrics.json"
    )
    exact_scores = []
    f1_scores = []

    with torch.no_grad(), open(result_path, "w", encoding="utf-8") as output:
        steps = range(0, len(items), data_args.eval_batch_size)
        for start in tqdm(steps, desc="NQ inference"):
            batch = items[start : start + data_args.eval_batch_size]
            input_ids, attention_mask, prompt_mask = build_batch(
                batch, model.tokenizer, training_args.model_max_length, device
            )
            generated_ids = model(
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                labels=None,
            )
            predictions = model.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for item, prediction in zip(batch, predictions):
                prediction = prediction.strip()
                exact, f1 = score_prediction(prediction, item["answer"])
                exact_scores.append(exact)
                f1_scores.append(f1)
                output.write(
                    json.dumps(
                        {
                            "prompt": item["prompt"],
                            "prediction": prediction,
                            "answer": item["answer"],
                            "em": exact,
                            "f1": f1,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    metrics = {
        "total_samples": len(exact_scores),
        "avg_em": sum(exact_scores) / len(exact_scores),
        "avg_f1": sum(f1_scores) / len(f1_scores),
    }
    with open(metrics_path, "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)

    print(f"Results saved to: {result_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"EM: {metrics['avg_em']:.4f} F1: {metrics['avg_f1']:.4f}")


if __name__ == "__main__":
    main()
