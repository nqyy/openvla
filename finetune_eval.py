"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

# CUDA problem on AWS
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/include:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type
import draccus
import json
import requests
from io import BytesIO

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

import tqdm
from PIL import Image
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import get_cosine_schedule_with_warmup

import wandb
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    # data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "mbodiai/xarm_7_6_delta"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    dataset_size: int = 600                                        # Size of fine-tuning dataset
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 50_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-6                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = None                        # Name of entity to log under

    # Evaluation Parameters
    eval_batch_size: int = 16             # Evaluation batch size
    eval_steps: int = 10                # Interval for running evaluation
    # fmt: on

    num_warmup_steps: int = 10


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class XarmDataset(Dataset):
    def __init__(
        self,
        dataset_split: Any,  # Use the appropriate type for your dataset
        action_tokenizer: Any,
        base_tokenizer: Any,
        image_transform: Callable[[Image.Image], torch.Tensor],
        prompt_builder_fn: Type[Any],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset = dataset_split
        self.data = list(self.dataset)  # Load the dataset into memory
        self.dataset_statistics = {
            "min": np.array([-0.07074, -0.10867, -0.08653, -6.28318977, -0.33160999, -0.33950999, 0.0]),
            "max": np.array([0.10836, 0.10925, 0.10709, 6.28318977, 0.14999001, 0.33197999, 1.0]),
            "mean": np.array([7.04869162e-03, -4.47613717e-03, -5.16619937e-03, -2.72256749e-05, -2.37738316e-03, -4.07056037e-04, 6.07476636e-01]),
            "q01": np.array([-0.0504155, -0.1054613, -0.08275, -6.28318977, -0.0226686, -0.33344679, 0.0]),
            "q99": np.array([0.0992138, 0.1063098, 0.0832235, 6.28318977, 0.0, 0.2959943, 1.0]),
            "std": np.array([0.02512313, 0.03886214, 0.03429312, 2.80552305, 0.02847977, 0.07015477, 0.48831217]),
        }

        self.transform = lambda x: x
        # Define your transformations / augmentations
        transform = [
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            ])
        ]
        transform.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.2, 2.0), value=0, inplace=False))
        self.transform = transforms.Compose(transform)


    def normalize_action(self, action):
        q01 = self.dataset_statistics['q01'][:-1]
        q99 = self.dataset_statistics['q99'][:-1]
        normalized_action = 2 * (action[:-1] - q01) / (q99 - q01 + 1e-8) - 1
        normalized_action = np.clip(normalized_action, -1, 1)
        # Append the grasp value without normalization
        normalized_action = np.append(normalized_action, action[-1])
        return normalized_action

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image = data["observation"]["image"]
        image = self.transform(image)
        image = transforms.ToPILImage()(image)
        instruction = data["observation"]["instruction"]
        action = np.array(
            [
                data["action"]["pose"]["x"],
                data["action"]["pose"]["y"],
                data["action"]["pose"]["z"],
                data["action"]["pose"]["roll"],
                data["action"]["pose"]["pitch"],
                data["action"]["pose"]["yaw"],
                data["action"]["grasp"],
            ]
        )
        normalized_action = self.normalize_action(action)

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {
                "from": "human",
                "value": f"What action should the robot take to {instruction}?",
            },
            {"from": "gpt", "value": self.action_tokenizer(normalized_action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image).to(torch.bfloat16)

        labels[: -(len(normalized_action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


def unnormalize_action(normalized_action, dataset_statistics):
    normalized_action = np.array(normalized_action)  # Convert to numpy array if it's a list
    q01 = dataset_statistics['q01'][:-1]
    q99 = dataset_statistics['q99'][:-1]
    action = (normalized_action[:-1] + 1) * (q99 - q01 + 1e-8) / 2 + q01
    # Append the grasp value without normalization
    action = np.append(action, normalized_action[-1])
    return action.tolist()

def evaluate_model(model, dataloader, device, step_idx, distributed_state, action_tokenizer, dataset_statistics):
    model.eval()
    total_loss = 0
    total_l1_loss = 0
    total_correct = 0
    total_mask = 0

    images_with_metadata = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"],
                )
                loss = output.loss
                total_loss += loss.item()

                # Compute Accuracy
                action_logits = output.logits[:, model.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                total_correct += correct_preds.sum().float().item()
                total_mask += mask.sum().float().item()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                total_l1_loss += l1_loss.item()

                continuous_actions_gt_list = continuous_actions_gt.tolist()
                continuous_actions_pred_list = continuous_actions_pred.tolist()

                nested_continuous_actions_gt_list = [
                    continuous_actions_gt_list[i : i + 7] for i in range(0, len(continuous_actions_gt_list), 7)
                ]
                nested_continuous_actions_pred_list = [
                    continuous_actions_pred_list[i : i + 7] for i in range(0, len(continuous_actions_pred_list), 7)
                ]
                unnormalized_continuous_actions_gt_list = [
                    unnormalize_action(action, dataset_statistics) for action in nested_continuous_actions_gt_list
                ]
                unnormalized_continuous_actions_pred_list = [
                    unnormalize_action(action, dataset_statistics) for action in nested_continuous_actions_pred_list
                ]
                # Log Images with Ground Truth and Predicted Actions as Metadata
                for img, gt_action, pred_action in zip(
                    batch["pixel_values"].to(torch.float32).cpu().numpy(),
                    unnormalized_continuous_actions_gt_list,
                    unnormalized_continuous_actions_pred_list,
                ):
                    image = img[:3].transpose(
                        1, 2, 0
                    )  # Select first 3 channels and transpose to [height, width, channels]
                    caption = f"Ground Truth: {gt_action}, Predicted: {pred_action}"
                    images_with_metadata.append(wandb.Image(image, caption=caption))

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_mask
    avg_l1_loss = total_l1_loss / len(dataloader)

    if distributed_state.is_main_process:
        wandb.log(
            {
                "eval_loss": avg_loss,
                "eval_accuracy": avg_accuracy,
                "eval_l1_loss": avg_l1_loss,
                "eval_images_actions": images_with_metadata,
            },
            step=step_idx,
        )

    return avg_loss, avg_accuracy, avg_l1_loss


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.num_warmup_steps,  # Adjust this value as needed
        num_training_steps=cfg.max_steps,
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # Load dataset and split it once
    dataset = load_dataset(cfg.dataset_name, streaming=False, trust_remote_code=True)
    split_data = dataset["train"].train_test_split(test_size=0.1)
    train_split = split_data["train"]
    test_split = split_data["test"]

    # Pass the split datasets to the XarmDataset class
    train_dataset = XarmDataset(
        dataset_split=train_split,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    test_dataset = XarmDataset(
        dataset_split=test_split,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    # if distributed_state.is_main_process:
    #     save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")
    # Calculate the number of epochs needed to achieve the max_steps
    steps_per_epoch = cfg.dataset_size // cfg.batch_size
    num_epochs = cfg.max_steps // steps_per_epoch + 1

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Train!
    step_idx = 0
    for epoch in range(num_epochs):
        with tqdm.tqdm(total=steps_per_epoch, leave=False) as progress:
            vla.train()
            optimizer.zero_grad()
            for batch in dataloader:
                if step_idx >= cfg.max_steps:
                    break

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Backward!
                loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Push Metrics to W&B (every 10 steps)
                if distributed_state.is_main_process and step_idx % 10 == 0:
                    wandb.log(
                        {
                            "learning_rate": scheduler.get_last_lr()[0],
                            "train_loss": loss,
                            "action_accuracy": action_accuracy,
                            "l1_loss": action_l1_loss,
                        },
                        step=step_idx,
                    )

                # Optimizer Step
                if (step_idx + 1) % cfg.grad_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if step_idx > 0 and step_idx % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if cfg.use_lora else run_dir

                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        vla.module.save_pretrained(save_dir)

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            merged_vla.save_pretrained(run_dir)

                    # Block on Main Process Checkpointing
                    dist.barrier()

                # Evaluate Model at Regular Intervals
                if step_idx > 0 and step_idx % cfg.eval_steps == 0:
                    evaluate_model(vla, eval_dataloader, device_id, step_idx, distributed_state, action_tokenizer, train_dataset.dataset_statistics)

                step_idx += 1
                if step_idx >= cfg.max_steps:
                    break


if __name__ == "__main__":
    finetune()
