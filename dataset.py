import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random


class WikiTextBidirectionalDataset(Dataset):
    def __init__(self, split, sequence_length, mask_probability, mask_token_id, tokenizer):
        self.sequence_length = sequence_length
        self.mask_probability = mask_probability
        self.mask_token_id = mask_token_id
        self.tokenizer = tokenizer

        raw_dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)

        all_text = " ".join([entry["text"] for entry in raw_dataset if entry["text"].strip()])
        all_token_ids = tokenizer.encode(all_text, add_special_tokens=False)

        self.chunks = [
            all_token_ids[start : start + sequence_length]
            for start in range(0, len(all_token_ids) - sequence_length, sequence_length)
        ]

    def __len__(self):
        return len(self.chunks) * 2

    def __getitem__(self, index):
        is_reversed = index >= len(self.chunks)
        chunk_index = index % len(self.chunks)

        token_ids = self.chunks[chunk_index]

        if is_reversed:
            token_ids = list(reversed(token_ids))

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        masked_token_ids, target_token_ids = self._apply_masking(token_ids)

        return {
            "masked_tokens": masked_token_ids,
            "target_tokens": target_token_ids,
            "is_reversed": torch.tensor(is_reversed, dtype=torch.bool)
        }

    def _apply_masking(self, token_ids):
        target_token_ids = token_ids.clone()
        masked_token_ids = token_ids.clone()

        mask_positions = torch.rand(token_ids.shape) < self.mask_probability
        masked_token_ids[mask_positions] = self.mask_token_id

        ignore_positions = ~mask_positions
        target_token_ids[ignore_positions] = -100

        return masked_token_ids, target_token_ids


def build_dataloaders(sequence_length, mask_probability, batch_size, num_workers=4):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    mask_token_id = tokenizer.mask_token_id
    vocabulary_size = tokenizer.vocab_size

    train_dataset = WikiTextBidirectionalDataset(
        split="train",
        sequence_length=sequence_length,
        mask_probability=mask_probability,
        mask_token_id=mask_token_id,
        tokenizer=tokenizer
    )

    validation_dataset = WikiTextBidirectionalDataset(
        split="validation",
        sequence_length=sequence_length,
        mask_probability=mask_probability,
        mask_token_id=mask_token_id,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, validation_loader, vocabulary_size, mask_token_id, tokenizer
