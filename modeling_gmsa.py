import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from peft import get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments as HFTrainingArguments,
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct"
    )
    stage: str = field(
        default="finetune",
        metadata={"help": "Training stage: autoencoding or finetune."},
    )
    merge_size: int = field(default=16, metadata={"help": "Group merge size."})
    merge_sizes: str = field(
        default="16,32", metadata={"help": "Candidate random merge sizes."}
    )
    is_random: bool = field(
        default=False, metadata={"help": "Sample merge size from merge_sizes."}
    )
    use_transform_layer: bool = field(
        default=True, metadata={"help": "Use a decoder block as the fusion layer."}
    )
    num_mem_fusion_layers: int = field(
        default=1, metadata={"help": "Number of memory fusion layers."}
    )
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    train: bool = field(
        default=False, metadata={"help": "Initialize the model for training."}
    )
    encoder_layers: int = field(default=8, metadata={"help": "Number of encoder layers."})


@dataclass
class DataArguments:
    train_file: str = field(
        default="/path/to/train.jsonl",
        metadata={"help": "Path to NQ training JSONL."},
    )
    test_file: str = field(
        default="/path/to/test.jsonl",
        metadata={"help": "Path to NQ test JSONL."},
    )
    train_samples: int = field(
        default=-1, metadata={"help": "Training samples; -1 uses all rows."}
    )
    eval_samples: int = field(
        default=-1, metadata={"help": "Evaluation samples; -1 uses all rows."}
    )
    eval_batch_size: int = field(default=2)
    eval_output_dir: str = field(default="./eval_result")
    debug_data: bool = field(default=False)


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=280000)
    report_to: Optional[str] = field(default="wandb")
    project_name: Optional[str] = field(default="gmsa")
    max_steps: int = field(default=10000)
    save_strategy: Optional[str] = field(default="steps")
    save_steps: int = field(default=2000)
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: int = field(default=20000)
    num_train_epochs: int = field(default=1)
    restore_from: str = field(default="")
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=100)
    deepspeed: Optional[str] = field(default=None)
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=1e-5)
    gradient_checkpointing: bool = field(default=True)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)


def print_trainable_parameters(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100 * trainable / total if total else 0
    print(
        f"trainable params: {trainable} || all params: {total} || "
        f"trainable%: {percentage:.2f}"
    )


class GMSA(nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.training_mode = model_args.train
        self.stage = model_args.stage
        if self.stage not in {"autoencoding", "finetune"}:
            raise ValueError(
                f"Unsupported stage: {self.stage}. "
                "Expected autoencoding or finetune."
            )

        dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        encoder_config = AutoConfig.from_pretrained(self.model_name)
        encoder_config.num_hidden_layers = model_args.encoder_layers
        encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name, config=encoder_config, trust_remote_code=True, torch_dtype=dtype
        )
        self.encoder = get_peft_model(encoder, lora_config)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_id = self.tokenizer.eos_token_id
        self.dim = self.encoder.config.hidden_size
        self.merge_size = model_args.merge_size
        self.merge_sizes = model_args.merge_sizes
        self.is_random = model_args.is_random
        self.use_transform_layer = model_args.use_transform_layer

        decoder_dim = self.decoder.config.hidden_size
        self.semantic_alignment_layer = nn.Linear(self.dim, decoder_dim).to(dtype=dtype)
        if self.use_transform_layer:
            fusion_config = AutoConfig.from_pretrained(self.model_name)
            fusion_config.num_hidden_layers = model_args.num_mem_fusion_layers
            self.memory_fusion_layer = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=fusion_config,
                trust_remote_code=True,
                torch_dtype=dtype,
            )

        if self.training_mode:
            self.init()

    def init(self):
        self._set_trainable_by_stage()
        print_trainable_parameters(self)
        if self.training_args.restore_from:
            print(
                "Loading from the pretrained checkpoint: "
                f"{self.training_args.restore_from}..."
            )
            self._load_checkpoint(self.training_args.restore_from)
            print(f"Finished loading from {self.training_args.restore_from}")

        print("Enabling gradient checkpointing...")
        for module in self._trainable_checkpoint_modules():
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _trainable_checkpoint_modules(self):
        modules = [self.encoder, self.decoder]
        if self.use_transform_layer:
            modules.append(self.memory_fusion_layer)
        return [
            module
            for module in modules
            if any(parameter.requires_grad for parameter in module.parameters())
        ]

    def _load_checkpoint(self, restore_from):
        checkpoint = restore_from
        if os.path.isdir(checkpoint):
            names = (
                "model.safetensors",
                "adapter_model.safetensors",
                "pytorch_model.bin",
            )
            checkpoint = next(
                (
                    os.path.join(restore_from, name)
                    for name in names
                    if os.path.isfile(os.path.join(restore_from, name))
                ),
                None,
            )
            if checkpoint is None:
                raise FileNotFoundError(
                    f"No supported checkpoint file found in: {restore_from}"
                )

        if checkpoint.endswith(".safetensors"):
            state_dict = load_file(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    @staticmethod
    def _set_lora_trainable(module, enabled=True):
        for name, parameter in module.named_parameters():
            is_lora = "lora_" in name or "modules_to_save" in name
            parameter.requires_grad = enabled and is_lora

    def _set_fusion_layers_trainable(self):
        self._set_requires_grad(self.memory_fusion_layer, False)
        base_model = getattr(
            self.memory_fusion_layer, "get_base_model", lambda: self.memory_fusion_layer
        )()
        core_model = getattr(base_model, "model", base_model)
        if hasattr(core_model, "layers"):
            self._set_requires_grad(core_model.layers, True)

    def _set_trainable_by_stage(self):
        if self.stage == "autoencoding":
            self._set_lora_trainable(self.encoder)
            self._set_requires_grad(self.decoder, False)
            if self.use_transform_layer:
                self._set_fusion_layers_trainable()
            self._set_requires_grad(
                self.semantic_alignment_layer, not self.use_transform_layer
            )
            return

        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.decoder, True)
        self._set_requires_grad(self.semantic_alignment_layer, False)
        if self.use_transform_layer:
            self._set_requires_grad(self.memory_fusion_layer, False)

    def tokens_to_embeddings(self, token_ids):
        return self.encoder.get_input_embeddings()(token_ids)

    def generate_merge_size(self):
        merge_sizes = [
            int(value.strip())
            for value in self.merge_sizes.split(",")
            if value.strip()
        ]
        if not merge_sizes or any(value <= 0 for value in merge_sizes):
            raise ValueError("merge_sizes must contain positive integers")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = next(self.parameters()).device
            sampled = random.choice(merge_sizes) if torch.distributed.get_rank() == 0 else 0
            value = torch.tensor(sampled, dtype=torch.long, device=device)
            torch.distributed.broadcast(value, src=0)
            return int(value.item())
        return random.choice(merge_sizes)

    def _generate_pooling_memories(self, hidden_states, attention_mask, merge_size):
        if merge_size <= 0:
            raise ValueError("merge_size must be positive")

        batch_size, _, hidden_size = hidden_states.shape
        memories = []
        valid_lengths = []
        for batch_index in range(batch_size):
            current = hidden_states[batch_index][attention_mask[batch_index].bool()]
            sequence_length = current.shape[0]
            if sequence_length == 0:
                pooled = current.new_zeros((1, hidden_size))
                valid_length = 0
            else:
                full_length = sequence_length - sequence_length % merge_size
                full_groups = current[:full_length].reshape(
                    -1, merge_size, hidden_size
                )
                pooled = full_groups.mean(dim=1)
                if full_length < sequence_length:
                    remainder = current[full_length:].mean(dim=0, keepdim=True)
                    pooled = torch.cat([pooled, remainder], dim=0)
                valid_length = pooled.shape[0]
            memories.append(pooled)
            valid_lengths.append(valid_length)

        max_length = max(memory.shape[0] for memory in memories)
        padded = hidden_states.new_zeros((batch_size, max_length, hidden_size))
        pooled_mask = torch.zeros(
            (batch_size, max_length), dtype=torch.bool, device=hidden_states.device
        )
        for index, (memory, valid_length) in enumerate(zip(memories, valid_lengths)):
            padded[index, : memory.shape[0]] = memory
            pooled_mask[index, :valid_length] = True

        if self.use_transform_layer:
            aligned = self.memory_fusion_layer(
                inputs_embeds=padded,
                attention_mask=pooled_mask,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]
        else:
            aligned = self.semantic_alignment_layer(padded)
        return aligned, pooled_mask

    def _append_decoder_prompt(
        self, input_ids, decoder_prompt_mask, compressed_embeds, compressed_mask
    ):
        prompt_lengths = decoder_prompt_mask.sum(dim=1)
        max_prompt_length = int(prompt_lengths.max().item())
        if max_prompt_length == 0:
            return compressed_embeds, compressed_mask

        batch_size = input_ids.shape[0]
        prompt_ids = torch.full(
            (batch_size, max_prompt_length),
            self.tokenizer.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        prompt_mask = torch.zeros(
            (batch_size, max_prompt_length),
            dtype=torch.bool,
            device=input_ids.device,
        )
        for index in range(batch_size):
            current = input_ids[index][decoder_prompt_mask[index]]
            prompt_ids[index, : current.shape[0]] = current
            prompt_mask[index, : current.shape[0]] = True

        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)
        return (
            torch.cat([compressed_embeds, prompt_embeds], dim=1),
            torch.cat([compressed_mask, prompt_mask], dim=1),
        )

    def forward(
        self,
        input_ids=None,
        prompt_mask=None,
        attention_mask=None,
        labels=None,
    ):
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        prompt_mask = (
            torch.zeros_like(input_ids, dtype=torch.bool)
            if prompt_mask is None
            else prompt_mask.bool()
        )
        encoder_mask = attention_mask & ~prompt_mask
        decoder_prompt_mask = prompt_mask & attention_mask

        encoder_outputs = self.encoder(
            inputs_embeds=self.tokens_to_embeddings(input_ids),
            attention_mask=encoder_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        merge_size = self.generate_merge_size() if self.is_random else self.merge_size
        compressed_embeds, compressed_mask = self._generate_pooling_memories(
            encoder_outputs.hidden_states[-1], encoder_mask, merge_size
        )
        prefix_embeds, prefix_mask = self._append_decoder_prompt(
            input_ids,
            decoder_prompt_mask,
            compressed_embeds,
            compressed_mask,
        )

        if labels is None:
            return self.decoder.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask.long(),
                max_new_tokens=100,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_id,
                do_sample=False,
                repetition_penalty=1.3,
                use_cache=True,
            )

        safe_labels = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
        label_embeds = self.decoder.get_input_embeddings()(safe_labels)
        full_embeds = torch.cat([prefix_embeds, label_embeds], dim=1)
        label_mask = labels != -100
        full_mask = torch.cat([prefix_mask, label_mask], dim=1)
        ignore_prefix = torch.full(
            prefix_mask.shape, -100, dtype=labels.dtype, device=labels.device
        )
        full_labels = torch.cat([ignore_prefix, labels], dim=1)
        outputs = self.decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            return_dict=True,
        )
        return {"loss": outputs.loss} if self.training else outputs

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_disable(*args, **kwargs)
