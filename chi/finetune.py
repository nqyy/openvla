import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Type
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from datasets import load_dataset
import wandb
from torch.utils.data import IterableDataset


from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

IGNORE_INDEX = -100


@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    dataset_name: str = "mbodiai/oxe_utokyo_xarm_pick_place"
    split: str = "default"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    batch_size: int = 16
    max_steps: int = 100_000
    save_steps: int = 5000
    learning_rate: float = 2e-5
    grad_accumulation_steps: int = 1
    shuffle_buffer_size: int = 100_000

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    wandb_project: str = "openvla"
    wandb_entity: str = None


class HuggingFaceDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: Any,
        image_transform: Callable[[Image.Image], torch.Tensor],
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

    def __iter__(self):
        for data in self.dataset:
            image = data["observation"]["image"]
            instruction = data["observation"]["task"]
            action = np.array(
                [data["relative_action"]["pose"][k] for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
                + [data["relative_action"]["grasp"]]
            )

            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {instruction}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)

            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
            pixel_values = self.image_transform(image).to(torch.bfloat16)

            labels[: -(len(action) + 1)] = IGNORE_INDEX

            yield dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class OpenVLATrainer(Trainer):
    def __init__(self, action_tokenizer: ActionTokenizer, **kwargs):
        super().__init__(**kwargs)
        self.action_tokenizer = action_tokenizer
        self.custom_metrics = {}

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(pixel_values=inputs.get("pixel_values"), input_ids=inputs.get("input_ids"), labels=labels)
        loss = outputs.loss

        # Compute Accuracy and L1 Loss for Logging
        action_logits = outputs.logits[:, model.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = labels[:, 1:].to(action_preds.device)
        mask = action_gt > self.action_tokenizer.action_token_begin_idx

        # Compute Accuracy
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # Compute L1 Loss on Predicted (Continuous) Actions
        continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
        )
        continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
        )
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

        # Store metrics for logging
        self.custom_metrics = {"action_accuracy": action_accuracy.item(), "l1_loss": action_l1_loss.item()}

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        logs.update(self.custom_metrics)  # Add custom metrics to the logs
        super().log(logs)


def main(cfg: FinetuneConfig):
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = HuggingFaceDataset(
        cfg.dataset_name,
        cfg.split,
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    training_args = TrainingArguments(
        output_dir=cfg.run_root_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        logging_dir=cfg.run_root_dir / "logs",
        logging_steps=10,
        report_to="wandb",
        run_name=f"ft+{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}",
    )

    collator = PaddedCollatorForActionPrediction(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side=processor.tokenizer.padding_side,
    )

    trainer = OpenVLATrainer(
        model=vla,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        action_tokenizer=action_tokenizer,
    )

    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity)
    trainer.train()


if __name__ == "__main__":
    cfg = FinetuneConfig()
    main(cfg)
