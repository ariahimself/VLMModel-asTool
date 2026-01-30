"""
vlm_dataset.py

Builds a simple "VQA-style" dataset from MNIST labels by generating
(question prompt, answer) pairs programmatically.

Each sample returns:
- image: (1,28,28) tensor (MNIST normalized)
- input_ids: token ids for the decoder input (teacher forcing)
- labels: next-token labels
- loss_mask: 1 where the loss should be applied (answer region), else 0

We use a character tokenizer so prompts can be free-form.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from vlm_tokenizer import CharTokenizer


_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}


@dataclass
class TaskSpec:
    name: str
    prompt_templates: List[str]


DEFAULT_TASKS: List[TaskSpec] = [
    TaskSpec(
        name="digit",
        prompt_templates=[
            "what digit is this?",
            "which digit is shown?",
            "what number is this?",
            "identify the digit."
        ],
    ),
    TaskSpec(
        name="parity",
        prompt_templates=[
            "is this even or odd?",
            "tell me if the digit is odd or even.",
            "parity?"
        ],
    ),
    TaskSpec(
        name="spell",
        prompt_templates=[
            "spell it out.",
            "write the digit in words.",
            "what is the digit in english?"
        ],
    ),
    TaskSpec(
        name="gt5",
        prompt_templates=[
            "is it greater than 5?",
            "is the digit > 5?",
            "greater than five?"
        ],
    ),
    TaskSpec(
        name="lt5",
        prompt_templates=[
            "is it less than 5?",
            "is the digit < 5?",
            "less than five?"
        ],
    ),
]


def _answer_for_task(task_name: str, y: int) -> str:
    if task_name == "digit":
        return str(y)
    if task_name == "parity":
        return "even" if (y % 2 == 0) else "odd"
    if task_name == "spell":
        return _WORDS[int(y)]
    if task_name == "gt5":
        return "yes" if y > 5 else "no"
    if task_name == "lt5":
        return "yes" if y < 5 else "no"
    # fallback
    return str(y)


class MNISTVQADataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        tokenizer: CharTokenizer,
        tasks: List[TaskSpec] = None,
        seed: int = 0,
        download: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tasks = tasks if tasks is not None else DEFAULT_TASKS

        # Normalize exactly like your ViT training and draw app
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.ds = datasets.MNIST(root=root, train=train, download=download, transform=tfm)

        # deterministic random for "train=False" (so val/test is stable)
        self.train = train
        self.rng = random.Random(seed if not train else None)

    def __len__(self):
        return len(self.ds)

    def _pick_task_and_prompt(self) -> Tuple[str, str]:
        task = (random.choice(self.tasks) if self.train else self.rng.choice(self.tasks))
        template = (random.choice(task.prompt_templates) if self.train else self.rng.choice(task.prompt_templates))
        return task.name, template

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, y = self.ds[idx]  # image: (1,28,28), y: int

        task_name, prompt = self._pick_task_and_prompt()
        answer = _answer_for_task(task_name, int(y))

        # We structure the text so generation is: given "question: ...\nanswer: " -> generate answer.
        input_text = f"question: {prompt.strip().lower()}\nanswer: "
        target_text = answer.strip().lower()

        # Full sequence includes prompt + answer, with BOS at start and EOS at end
        full_ids = self.tokenizer.encode(input_text + target_text, add_bos=True, add_eos=True)

        # Teacher forcing: model sees input_ids, predicts labels (next token)
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)

        # Loss should apply only to the answer region (including EOS).
        prompt_ids = self.tokenizer.encode(input_text, add_bos=True, add_eos=False)
        # labels index where the first answer token is predicted:
        start = len(prompt_ids) - 1
        loss_mask = torch.zeros_like(labels, dtype=torch.float32)
        loss_mask[start:] = 1.0

        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "meta": {"task": task_name, "prompt": prompt, "answer": answer, "label": int(y)},
        }


def collate_batch(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)

    max_len = max(int(b["input_ids"].shape[0]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), max_len), dtype=torch.float32)

    for i, b in enumerate(batch):
        L = int(b["input_ids"].shape[0])
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        loss_mask[i, :L] = b["loss_mask"]

    meta = [b["meta"] for b in batch]
    return {"images": images, "input_ids": input_ids, "labels": labels, "loss_mask": loss_mask, "meta": meta}
