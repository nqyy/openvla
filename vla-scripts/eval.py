import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, PreTrainedTokenizerBase
from PIL import Image
from typing import Callable, Type
from datasets import load_dataset

from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

IGNORE_INDEX = -100
# Define the evaluation dataset class
class HuggingFaceIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: Callable[[Image.Image], torch.Tensor],
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Load the dataset
        self.dataset = load_dataset(
            dataset_name, streaming=True, split=split, trust_remote_code=True
        )

    def __iter__(self):
        for data in self.dataset:
            # Load data from the dataset
            image = data["observation"]["image"]
            instruction = data["observation"]["task"]
            action = np.array(
                [data["relative_action"]["pose"][k] for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
                + [data["relative_action"]["grasp"]]
            )

            # Add instruction to VLA prompt
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {instruction}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            # Tokenize (w/ `base_tokenizer`)
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)

            # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
            #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_MODEL!
            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
            pixel_values = self.image_transform(image)

            # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            labels[: -(len(action) + 1)] = IGNORE_INDEX

            yield dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


# Define the evaluation function
def evaluate_robot_model(model, dataloader, action_tokenizer, device):
    model.eval()
    correct_actions = 0
    total_actions = 0
    l1_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'pixel_values': batch['pixel_values'].to(device, dtype=torch.float16),
                'labels': batch['labels'].to(device)
            }
            
            outputs = model(**inputs)
            action_logits = outputs.logits[:, model.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch['labels'][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            
            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            correct_actions += correct_preds.sum().item()
            total_actions += mask.sum().item()
            
            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            l1_losses.append(action_l1_loss.item())
            print("l1 loss:", action_l1_loss.item())
            print("correct actions / total:", correct_actions, "/", total_actions)


    # Calculate Metrics
    action_accuracy = correct_actions / total_actions
    mean_l1_loss = np.mean(l1_losses)

    metrics = {
        'action_accuracy': action_accuracy,
        'mean_l1_loss': mean_l1_loss
    }

    return metrics


# Define the main function for loading the model and dataset
def main():
    # Load the processor and model
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda")

    # Create the action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Define the dataset
    vla_dataset = HuggingFaceIterableDataset(
        "mbodiai/oxe_bridge",
        "all",
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
    )

    # Create the DataLoader
    dataloader = DataLoader(
        vla_dataset,
        batch_size=16,
        collate_fn=PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        ),
        num_workers=0
    )

    # Evaluate the model
    metrics = evaluate_robot_model(vla, dataloader, action_tokenizer, device='cuda')
    print(metrics)


if __name__ == "__main__":
    main()
