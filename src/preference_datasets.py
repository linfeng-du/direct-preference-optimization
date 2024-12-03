import random
from typing import Any
from collections import defaultdict
from collections.abc import Callable, Iterator

import datasets
from transformers import PreTrainedTokenizerBase

import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from utils import TemporarilySeededRandom, log_main_process


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
        """Load dataset split by its name and flatten to get examples."""
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        loader, self.truncation_mode = self.get_dataset_config(dataset)

        self.examples = []
        self.grouped_indices = []

        for prompt, prompt_data in loader(self.split, **loader_kwargs).items():
            responses = prompt_data.pop('responses')
            pairs = prompt_data.pop('pairs')

            indices = []
            for idx, (chosen, rejected) in enumerate(pairs):
                others = {}
                for key, val in prompt_data.items():
                    if isinstance(val, str):
                        # Prompt-level property
                        others[key] = val
                    elif isinstance(val, list):
                        # Example-level property
                        others[key] = val[idx]

                indices.append(len(self.examples))
                self.examples.append((
                    prompt,
                    responses[chosen],
                    responses[rejected],
                    others
                ))

            self.grouped_indices.append(indices)

    @staticmethod
    def get_dataset_config(dataset: str) -> tuple[Callable, str]:
        """Get the loader and truncation mode of the dataset."""
        dataset_configs = {
            'persona': {'loader': _load_persona, 'truncation_mode': 'keep_end'}
        }
        config = dataset_configs[dataset]
        return config['loader'], config['truncation_mode']

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


class PreferenceSampler(Sampler):

    def __init__(
            self,
            dataset: PreferenceDataset,
            shuffle: bool,
            seed: int | None = None,
            n_epochs: int = -1,
            n_examples: int = -1
        ) -> None:
        assert n_epochs > 0 or n_examples > 0, \
               'Must specify either n_epochs or n_examples'

        self.shuffle = shuffle
        self.n_epochs = n_epochs
        self.n_examples = n_examples
        self.seed = seed

        self.split = dataset.split
        self.grouped_indices = dataset.grouped_indices[:]

        epoch_examples = self.n_epochs * len(dataset)
        if self.n_epochs == -1:
            self.sampler_len = self.n_examples
        elif self.n_examples == -1:
            self.sampler_len = epoch_examples
        else:
            self.sampler_len = min(self.n_examples, epoch_examples)

        self.epoch_idx = 0
        self.example_idx = 0

    def __iter__(self) -> Iterator[int]:
        while True:
            if self.shuffle:
                with TemporarilySeededRandom(self.seed):
                    random.shuffle(self.grouped_indices)
                    self.seed = random.randint(a=0, b=2**32)

            for indices in self.grouped_indices:
                for index in indices:
                    yield index

                    self.example_idx += 1
                    if self.example_idx == self.n_examples:
                        log_main_process(f'Finished generating {self.n_examples} examples ' \
                                         f'on {self.split} split')
                        return

            self.epoch_idx += 1
            if self.epoch_idx == self.n_epochs:
                log_main_process(f'Finished generating {self.n_epochs} epochs ' \
                                 f'on {self.split} split')
                return

    def __len__(self) -> int:
        return self.sampler_len


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
                    # Flip prompt tokens before right padding
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
                    # Flip back so padding is on the left side
                    padded_batch[key] = padded_batch[key].flip(dims=[1])
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

    Handle truncation when the prompt + chosen or prompt + rejected responses is/are too long.
    Truncate the prompt first; if still too long, truncate the chosen/rejected responses.

    Create SFT labels for the chosen/rejected responses, which are of length equal to
    the sum of the length of the prompt and the chosen/rejected response.
    Labels for the prompt tokens are set to -100.
    """
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    # If the combined sequence is too long, truncate the prompt
    prompt_length = len(prompt_tokens['input_ids'])
    chosen_length = len(chosen_tokens['input_ids'])
    rejected_length = len(rejected_tokens['input_ids'])
    longer_response_length = max(chosen_length, rejected_length)

    if prompt_length + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {
                k: v[:max_prompt_length]
                for k, v in prompt_tokens.items()
            }
        elif truncation_mode == 'keep_end':
            prompt_tokens = {
                k: v[-max_prompt_length:]
                for k, v in prompt_tokens.items()
            }
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # If that's still too long, truncate the response
    prompt_length = len(prompt_tokens['input_ids'])

    if prompt_length + longer_response_length > max_length:
        max_response_length = max_length - max_prompt_length
        chosen_tokens = {
            k: v[:max_response_length]
            for k, v in chosen_tokens.items()
        }
        rejected_tokens = {
            k: v[:max_response_length]
            for k, v in rejected_tokens.items()
        }

    chosen_sequence_tokens = {
        k: prompt_tokens[k] + chosen_tokens[k]
        for k in chosen_tokens
    }
    rejected_sequence_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k]
        for k in rejected_tokens
    }

    # Create labels
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:prompt_length] = [-100] * prompt_length
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:prompt_length] = [-100] * prompt_length

    example = {
        'prompt': prompt,
        'chosen': prompt + chosen,
        'rejected': prompt + rejected,
        'chosen_response_only': chosen,
        'rejected_response_only': rejected
    }

    categories = {
        'prompt': prompt_tokens,
        'chosen': chosen_sequence_tokens,
        'rejected': rejected_sequence_tokens
    }
    for category, tokens in categories.items():
        for tokens_key, tokens_val in tokens.items():
            if tokens_key == 'token_type_ids':
                continue

            example[f'{category}_{tokens_key}'] = tokens_val

    return example


def _load_persona(
    split: str,
    prepend_persona: bool
) -> dict[str, dict[str, str | list[str] | list[tuple[int, int]]]]:
    """Load the PERSONA dataset from Huggingface and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt1': {
            'responses': list[str],
            'pairs': list[tuple[int, int]],
            'persona': list[str]
        },
        ...
    }

    Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
    """
    def get_split_indices(split: str):
        """Create train/test/test_unseen splits for the PERSONA dataset.

        The dataset contains 1,000 personas, each with 100 training and testing examples.
        Randomly reserve 200 personas as the unseen test set.
        """
        if not hasattr(get_split_indices, '_splits'):
            with TemporarilySeededRandom(seed=42):
                unseen_personas = random.sample(range(1000), 200)
                seen_personas = [p for p in range(1000) if p not in unseen_personas]

            # For each persona, the first 100 examples are used for training,
            # and the following 100 examples are used for testing
            def get_indices(personas, start, end):
                return [
                    index
                    for p in personas
                    for index in range(p * 200 + start, p * 200 + end)
                ]

            train_indices = get_indices(seen_personas, start=0, end=100)
            test_indices = get_indices(seen_personas, start=100, end=200)
            test_unseen_indices = get_indices(unseen_personas, start=0, end=200)

            get_split_indices._splits = {
                'train': train_indices,
                'test': test_indices,
                'test_unseen': test_unseen_indices
            }

        return get_split_indices._splits[split]

    log_main_process(f'Loading PERSONA dataset ({split} split)')

    dataset = datasets.load_dataset('SynthLabsAI/PERSONA', split='train')
    data = defaultdict(lambda: defaultdict(list))

    for index in tqdm(get_split_indices(split), desc='Processing PERSONA'):
        example = dataset[index]

        prompt = example['instruction']
        chosen = example['data']
        rejected = example['original']
        persona = example['persona']

        if prepend_persona:
            persona_row = ', '.join(persona.strip().split('\n'))
            prompt = f'Here are your characteristics: {persona_row}. {prompt}'

        prompt = f'\n\nHuman: {prompt}\n\nAssistant:'
        responses = [f' {chosen}', f' {rejected}']
        n_responses = len(data[prompt]['responses'])

        data[prompt]['responses'].extend(responses)
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['persona'].append(persona)

    return data
