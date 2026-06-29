import os

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
import safetensors.torch

def _require_columns(examples, columns):
    missing = [column for column in columns if column not in examples]
    if missing:
        raise ValueError(f"Missing required dataset columns: {', '.join(missing)}")


def select_answer(answer):
    references = answer if isinstance(answer, list) else [answer]
    for reference in references:
        if isinstance(reference, str) and reference.strip():
            return reference
    raise ValueError("answer must contain at least one non-empty string")


def build_context_prompt_ids(tokenizer, context, prompt, max_length):
    if not isinstance(context, str) or not isinstance(prompt, str):
        raise ValueError("input and prompt must be strings")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_length]
    available = max_length - len(prompt_ids)
    context_ids = context_ids[:available]
    input_ids = context_ids + prompt_ids
    prompt_mask = [0] * len(context_ids) + [1] * len(prompt_ids)
    return input_ids, prompt_mask


class SingleDeviceTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.local_rank != -1:
            return super()._wrap_model(
                model, training=training, dataloader=dataloader
            )
        if torch.cuda.is_available():
            return model.to(torch.device("cuda:0"))
        return model
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        deduped = {}
        seen_data_ptrs = {}
        for k, v in state_dict.items():
            ptr = v.data_ptr()
            if ptr in seen_data_ptrs:
                deduped[k] = v.clone()
            else:
                seen_data_ptrs[ptr] = k
                deduped[k] = v
        safetensors.torch.save_file(deduped, os.path.join(output_dir, "model.safetensors"))
        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)

class AutoEncodingTokenizeFunction:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = "Restate the aforementioned Text."

    def __call__(self, examples):
        _require_columns(examples, ("input",))
        input_ids = []
        prompt_masks = []
        labels = []
        eos_token = self.tokenizer.eos_token or ""

        for context in examples["input"]:
            if not isinstance(context, str):
                raise ValueError("input must be a string")
            encoded, prompt_mask = build_context_prompt_ids(
                self.tokenizer, context, self.instruction, self.max_length
            )
            target = self.tokenizer.encode(
                f"{context}{eos_token}", add_special_tokens=False
            )[: self.max_length]
            input_ids.append(torch.tensor(encoded))
            prompt_masks.append(torch.tensor(prompt_mask, dtype=torch.long))
            labels.append(torch.tensor(target))

        return {
            "input_ids": input_ids,
            "prompt_mask": prompt_masks,
            "labels": labels,
        }


class InstructFTTokenizeFunction:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        _require_columns(examples, ("input", "prompt", "answer"))
        input_ids = []
        prompt_masks = []
        labels = []
        eos_token = self.tokenizer.eos_token or ""

        rows = zip(examples["input"], examples["prompt"], examples["answer"])
        for context, prompt, answer in rows:
            encoded, prompt_mask = build_context_prompt_ids(
                self.tokenizer, context, prompt, self.max_length
            )
            target = select_answer(answer)
            answer_ids = self.tokenizer.encode(
                f"{target}{eos_token}", add_special_tokens=False
            )
            input_ids.append(torch.tensor(encoded))
            prompt_masks.append(torch.tensor(prompt_mask, dtype=torch.long))
            labels.append(torch.tensor(answer_ids))

        return {
            "input_ids": input_ids,
            "prompt_mask": prompt_masks,
            "labels": labels,
        }


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.as_tensor(feature["input_ids"]) for feature in features]
        labels = [torch.as_tensor(feature["labels"]) for feature in features]
        input_lens = [value.shape[0] for value in input_ids]
        prompt_masks = [
            torch.as_tensor(feature.get("prompt_mask", torch.zeros_like(ids)))
            for feature, ids in zip(features, input_ids)
        ]

        batch_input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        batch_prompt_mask = pad_sequence(
            prompt_masks, batch_first=True, padding_value=0
        )
        attention_mask = torch.zeros_like(batch_input_ids, dtype=torch.bool)
        for index, sequence_length in enumerate(input_lens):
            attention_mask[index, :sequence_length] = True

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "prompt_mask": batch_prompt_mask,
            "attention_mask": attention_mask,
        }


def train_model(model, train_dataset, eval_dataset, training_args, tokenizer):
    use_deepspeed = bool(getattr(training_args, "deepspeed", None))
    launched_with_torchrun = "LOCAL_RANK" in os.environ or "RANK" in os.environ
    if use_deepspeed and not launched_with_torchrun:
        raise RuntimeError(
            "DeepSpeed requires torchrun/deepspeed launcher. Launch with "
            "`torchrun --nproc_per_node=NUM_GPUS autoencoding.py ... "
            "--deepspeed ds_config.json`."
        )

    if not use_deepspeed and not launched_with_torchrun:
        for key in (
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_DEEPSPEED_CONFIG_FILE",
            "DEEPSPEED_CONFIG_FILE",
        ):
            os.environ.pop(key, None)
        training_args.local_rank = -1

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is "
                "not empty. Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
        print(training_args)

    if torch.cuda.is_available() and not use_deepspeed:
        training_args._n_gpu = 1
    if use_deepspeed and local_rank == 0:
        print(f"DeepSpeed enabled with config: {training_args.deepspeed}")

    trainer = SingleDeviceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForDynamicPadding(tokenizer.pad_token_id),
    )

    if torch.cuda.is_available() and not use_deepspeed:
        trainer.model = trainer.model.to(torch.device("cuda:0"))
        cpu_params = [
            name
            for name, parameter in trainer.model.named_parameters()
            if parameter.device.type == "cpu"
        ]
        cpu_buffers = [
            name
            for name, buffer in trainer.model.named_buffers()
            if buffer.device.type == "cpu"
        ]
        if cpu_params or cpu_buffers:
            raise RuntimeError(
                "Model still has CPU tensors after move. "
                f"cpu_params={cpu_params[:8]}, cpu_buffers={cpu_buffers[:8]}"
            )

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    if checkpoint:
        print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
