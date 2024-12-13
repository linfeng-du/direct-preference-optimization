from typing import Any
from collections.abc import Callable, Iterator

from transformers import PreTrainedTokenizerBase

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from preference_datasets import load_preference_dataset


class PreferenceDataset(Dataset):

    def __init__(
        self,
        dataset: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        max_prompt_length: int,
        **loader_kwargs: Any
    ) -> None:
        """Load the dataset split by its name and flatten to get the examples."""
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        data, self.truncation_mode = load_preference_dataset(dataset, split, **loader_kwargs)

        self.examples = []
        self.grouped_indices = []

        for prompt, prompt_data in data.items():
            indices = []
            responses = prompt_data.pop('responses')
            pairs = prompt_data.pop('pairs')

            for idx, (chosen, rejected) in enumerate(pairs):
                indices.append(len(self.examples))
                others = {key: value[idx] for key, value in prompt_data.items()}
                self.examples.append((
                    prompt,
                    responses[chosen],
                    responses[rejected],
                    others
                ))

            self.grouped_indices.append(indices)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str | list[int]]:
        prompt, chosen, rejected, others = self.examples[index]
        example = _tokenize_example(
            prompt,
            chosen,
            rejected,
            self.tokenizer,
            self.max_length,
            self.max_prompt_length,
            self.truncation_mode
        )
        example.update(others)
        return example


class PreferenceSampler(Sampler[int]):

    def __init__(
        self,
        dataset: PreferenceDataset,
        shuffle: bool,
        seed: int | None = None,
        n_epochs: int | None = None,
        n_examples: int | None = None
    ) -> None:
        assert n_epochs is not None or n_examples is not None, (
            'Must specify either n_epochs or n_examples'
        )

        self.dataset = dataset
        self.shuffle = shuffle
        self.n_epochs = n_epochs
        self.n_examples = n_examples

        if self.shuffle:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

        self.epoch_count = None
        self.example_count = None

    @property
    def num_samples(self) -> int:
        if self.n_epochs is None:
            return self.n_examples

        epoch_examples = self.n_epochs * len(self.dataset)

        if self.n_examples is None:
            return epoch_examples

        return min(epoch_examples, self.n_examples)

    def __iter__(self) -> Iterator[int]:
        self.epoch_count = 0
        self.example_count = 0

        while True:
            n_groups = len(self.dataset.grouped_indices)
            if self.shuffle:
                group_indices = torch.randperm(n_groups, generator=self.generator).tolist()
            else:
                group_indices = list(range(n_groups))

            for group_index in group_indices:
                for index in self.dataset.grouped_indices[group_index]:
                    yield index

                    self.example_count += 1
                    if self.n_examples is not None and self.example_count == self.n_examples:
                        self.epoch_count = 0
                        self.example_count = 0
                        return

            self.epoch_count += 1
            if self.n_epochs is not None and self.epoch_count == self.n_epochs:
                self.epoch_count = 0
                self.example_count = 0
                return

    def __len__(self) -> int:
        return self.num_samples


def get_collate_fn(
    tokenizer: PreTrainedTokenizerBase
) -> Callable[[list[dict[str, str | list[int]]]], dict[str, list[str] | torch.Tensor]]:
    """Get the collate function for the given tokenizer.

    The collate function takes a list of examples and returns a batch of examples.
    Examples are dicts, where values are strings [the original texts] or lists of ints [tokens].
    Strings are passed through; PyTorch tensors are padded to the length of the longest sequence.
    """
    def collate_fn(batch):
        padded_batch = {}
        for key in batch[0]:
            if key.endswith(('_input_ids', '_attention_mask', '_labels')):
                if 'prompt' in key:
                    # Reverse the prompt tokens before right padding
                    sequences = [torch.tensor(e[key][::-1], dtype=torch.long) for e in batch]
                else:
                    sequences = [torch.tensor(e[key], dtype=torch.long) for e in batch]

                if key.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif key.endswith('_attention_mask'):
                    padding_value = 0
                elif key.endswith('_labels'):
                    padding_value = -100

                padded_batch[key] = pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=padding_value,
                    padding_side='right'
                )

                if 'prompt' in key:
                    # Flip them back to place the padding on the left side
                    padded_batch[key] = padded_batch[key].flip(dims=[1])
            elif isinstance(batch[0][key], list):
                padded_batch[key] = torch.tensor([e[key] for e in batch])
            else:
                padded_batch[key] = [e[key] for e in batch]

        return padded_batch

    return collate_fn


def _tokenize_example(
    prompt: str,
    chosen: str,
    rejected: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str
) -> dict[str, str | list[int]]:
    """Tokenize a single example.

    Handle truncation when the prompt + chosen or prompt + rejected responses exceeds max_length.
    Truncate the prompt first, and if still too long, truncate the chosen and rejected responses.

    Create SFT labels for the chosen and rejected responses,
    with lengths equal to the sum of the prompt length and the respective chosen or rejected response.
    Labels for the prompt tokens are set to -100.
    """
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)
    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    # If the combined sequence exceeds max_length, truncate the prompt first
    prompt_length = len(prompt_tokens['input_ids'])
    chosen_length = len(chosen_tokens['input_ids'])
    rejected_length = len(rejected_tokens['input_ids'])
    longer_length = max(chosen_length, rejected_length)

    if prompt_length + longer_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # If still too long, truncate the responses
    prompt_length = len(prompt_tokens['input_ids'])

    if prompt_length + longer_length > max_length:
        max_response_length = max_length - max_prompt_length
        chosen_tokens = {k: v[:max_response_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_response_length] for k, v in rejected_tokens.items()}

    # Prepend the prompt to the responses
    chosen_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}

    # Create labels
    chosen_tokens['labels'] = chosen_tokens['input_ids'][:]
    chosen_tokens['labels'][:prompt_length] = [-100] * prompt_length
    rejected_tokens['labels'] = rejected_tokens['input_ids'][:]
    rejected_tokens['labels'][:prompt_length] = [-100] * prompt_length

    example = {
        'prompt': prompt,
        'chosen_response_only': chosen,
        'rejected_response_only': rejected
    }

    categories = {
        'prompt': prompt_tokens,
        'chosen': chosen_tokens,
        'rejected': rejected_tokens
    }
    for category, tokens in categories.items():
        for key, value in tokens.items():
            example[f'{category}_{key}'] = value

    return example
