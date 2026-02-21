"""PyTorch Dataset for Dream SFT (Supervised Fine-Tuning) on NER data.

This module provides :class:`DreamNERDataset`, which tokenizes prompt/completion
pairs and pads completions to a fixed ``MAX_COMPLETION_LENGTH`` (default 128).
A custom :func:`random_truncation_collate_fn` implements Dream-Coder-style
random truncation: each completion in the batch is truncated to the length of
a randomly-chosen other example, eliminating most PAD tokens from the loss
computation.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# Dream-7B special token IDs
MASK_TOKEN_ID = 126336


class DreamNERDataset(Dataset):
    """Tokenized dataset for Dream-7B masked-diffusion SFT.

    Each element is a prompt/completion pair.  Prompts are tokenized without
    padding.  Completions are tokenized and right-padded with the tokenizer's
    ``pad_token_id`` to exactly ``max_completion_length`` tokens.

    Parameters
    ----------
    data : list[dict]
        List of dicts with ``"prompt"`` and ``"completion"`` string fields.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer for Dream-7B (Qwen2.5-based).
    max_seq_length : int, optional
        Maximum total sequence length (prompt + completion).  Prompts that
        exceed ``max_seq_length - max_completion_length`` are truncated from
        the left (keeping the end of the prompt which contains the passage
        text closest to the completion).  Default 512.
    max_completion_length : int, optional
        Fixed length for all completions after padding.  Default 128.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Any,
        max_seq_length: int = 512,
        max_completion_length: int = 128,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_completion_length = max_completion_length
        self.max_prompt_length = max_seq_length - max_completion_length

        # Resolve pad token id
        if tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            self.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id set."
            )

        # Pre-tokenize everything for faster __getitem__
        self._prompt_ids: List[List[int]] = []
        self._completion_ids: List[List[int]] = []
        self._completion_lengths: List[int] = []  # unpadded lengths

        for example in data:
            p_ids, c_ids, c_len = self._tokenize_example(example)
            self._prompt_ids.append(p_ids)
            self._completion_ids.append(c_ids)
            self._completion_lengths.append(c_len)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize_example(
        self, example: Dict[str, str]
    ) -> Tuple[List[int], List[int], int]:
        """Tokenize a single prompt/completion pair.

        Returns
        -------
        Tuple[List[int], List[int], int]
            ``(prompt_ids, padded_completion_ids, unpadded_completion_length)``
        """
        prompt_text = example["prompt"]
        completion_text = example["completion"]

        # Tokenize (no special tokens -- Dream handles this in the model)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(
            completion_text, add_special_tokens=False
        )

        # Truncate prompt from the LEFT if too long
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length :]

        # Record unpadded completion length before truncation/padding
        unpadded_len = min(len(completion_ids), self.max_completion_length)

        # Truncate completion if it exceeds max_completion_length
        if len(completion_ids) > self.max_completion_length:
            completion_ids = completion_ids[: self.max_completion_length]

        # Right-pad completion to max_completion_length
        pad_count = self.max_completion_length - len(completion_ids)
        completion_ids = completion_ids + [self.pad_token_id] * pad_count

        return prompt_ids, completion_ids, unpadded_len

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single tokenized example.

        Returns
        -------
        dict
            ``prompt_ids`` : LongTensor of shape ``(prompt_len,)``
            ``completion_ids`` : LongTensor of shape ``(max_completion_length,)``
            ``completion_length`` : int tensor -- unpadded completion length
            ``attention_mask`` : BoolTensor of shape ``(prompt_len + max_completion_length,)``
                True for real tokens, False for PAD positions.
        """
        prompt_ids = torch.tensor(self._prompt_ids[idx], dtype=torch.long)
        completion_ids = torch.tensor(self._completion_ids[idx], dtype=torch.long)
        comp_len = self._completion_lengths[idx]

        # Build attention mask: True for non-PAD positions
        prompt_mask = torch.ones(len(prompt_ids), dtype=torch.bool)
        completion_mask = torch.cat([
            torch.ones(comp_len, dtype=torch.bool),
            torch.zeros(self.max_completion_length - comp_len, dtype=torch.bool),
        ])
        attention_mask = torch.cat([prompt_mask, completion_mask])

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "completion_length": torch.tensor(comp_len, dtype=torch.long),
            "attention_mask": attention_mask,
        }

    @property
    def completion_lengths(self) -> List[int]:
        """Unpadded completion lengths for all examples (useful for analysis)."""
        return list(self._completion_lengths)


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def random_truncation_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate with Dream-Coder-style random truncation.

    For each example *i* in the batch, we randomly select another example *j*
    and truncate example *i*'s completion to the **unpadded** length of
    example *j*.  Tokens beyond that length are replaced with PAD, and the
    attention mask is updated accordingly.

    This removes most PAD tokens from the training signal, since each
    completion is effectively shortened to a realistic length drawn from the
    empirical length distribution of the batch.

    Parameters
    ----------
    batch : list[dict]
        List of dicts from :meth:`DreamNERDataset.__getitem__`.

    Returns
    -------
    dict
        Batched tensors with keys:
        - ``prompt_ids``: list of LongTensors (variable-length prompts)
        - ``completion_ids``: LongTensor ``(B, max_completion_length)``
        - ``completion_length``: LongTensor ``(B,)`` -- truncated lengths
        - ``attention_mask``: list of BoolTensors (variable total length)
        - ``original_completion_length``: LongTensor ``(B,)`` -- pre-truncation
    """
    batch_size = len(batch)
    comp_lengths = [item["completion_length"].item() for item in batch]

    # For each example, pick a random *other* example's length as the truncation target
    truncated_lengths: List[int] = []
    for i in range(batch_size):
        # Pick a random index (may be self, which is fine)
        j = random.randint(0, batch_size - 1)
        # Truncate to the minimum of own length and the donor's length
        trunc_len = min(comp_lengths[i], comp_lengths[j])
        truncated_lengths.append(trunc_len)

    # Determine the PAD token id from the first example's completion
    # (last token is guaranteed to be PAD if completion was shorter than max)
    pad_token_id = batch[0]["completion_ids"][-1].item()

    # Apply truncation
    new_completion_ids = []
    new_attention_masks = []
    prompt_ids_list = []
    max_completion_length = batch[0]["completion_ids"].shape[0]

    for i, item in enumerate(batch):
        trunc_len = truncated_lengths[i]
        comp_ids = item["completion_ids"].clone()

        # Zero out (PAD) everything beyond trunc_len
        if trunc_len < max_completion_length:
            comp_ids[trunc_len:] = pad_token_id

        new_completion_ids.append(comp_ids)
        prompt_ids_list.append(item["prompt_ids"])

        # Rebuild attention mask
        prompt_len = item["prompt_ids"].shape[0]
        prompt_mask = torch.ones(prompt_len, dtype=torch.bool)
        comp_mask = torch.cat([
            torch.ones(trunc_len, dtype=torch.bool),
            torch.zeros(max_completion_length - trunc_len, dtype=torch.bool),
        ])
        new_attention_masks.append(torch.cat([prompt_mask, comp_mask]))

    return {
        "prompt_ids": prompt_ids_list,  # list of variable-length tensors
        "completion_ids": torch.stack(new_completion_ids),  # (B, max_comp_len)
        "completion_length": torch.tensor(truncated_lengths, dtype=torch.long),
        "original_completion_length": torch.tensor(comp_lengths, dtype=torch.long),
        "attention_mask": new_attention_masks,  # list of variable-length tensors
    }


# ---------------------------------------------------------------------------
# Utility: pad prompts for batched forward pass
# ---------------------------------------------------------------------------

def pad_prompt_batch(
    prompt_ids_list: List[torch.Tensor],
    pad_token_id: int,
    padding_side: str = "left",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length prompt tensors into a batch.

    Parameters
    ----------
    prompt_ids_list : list[Tensor]
        Each tensor has shape ``(prompt_len_i,)``.
    pad_token_id : int
        Token ID used for padding.
    padding_side : str, optional
        ``"left"`` (default) or ``"right"``.

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(padded_prompt_ids, prompt_attention_mask)`` both of shape
        ``(B, max_prompt_len)``.
    """
    max_len = max(t.shape[0] for t in prompt_ids_list)
    batch_size = len(prompt_ids_list)
    padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, ids in enumerate(prompt_ids_list):
        length = ids.shape[0]
        if padding_side == "left":
            padded[i, max_len - length :] = ids
            mask[i, max_len - length :] = True
        else:
            padded[i, :length] = ids
            mask[i, :length] = True

    return padded, mask
