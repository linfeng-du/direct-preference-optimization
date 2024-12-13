import os
import json
import random
from time import perf_counter
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

from tqdm import tqdm
from omegaconf import DictConfig

from adapters import LoRAController, MixtureOfLoRAsController
from dataset import PreferenceDataset, PreferenceSampler, get_collate_fn
from utils import log_accelerate


class AccelerateTrainer:

    def __init__(
        self,
        config: DictConfig,
        policy: nn.Module,
        controller: LoRAController | MixtureOfLoRAsController,
        reference_model: nn.Module,
        accelerator: Accelerator
    ):
        """Trainer that leverages Hugging Face Accelerate for distributed training."""
        self.config = config
        self.controller = controller
        self.accelerator = accelerator

        self.example_count = None

        train_loader, test_loader, test_unseen_loader = self._load_dataset()
        optimizer = getattr(optim, self.config.optimizer)(
            [p for p in policy.parameters() if p.requires_grad],
            lr=self.config.lr
        )
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1., (step + 1) / (self.config.warmup_steps + 1))
        )
        (
            self.policy,
            self.reference_model,
            self.train_loader,
            self.test_loader,
            self.test_unseen_loader,
            self.optimizer,
            self.scheduler
        ) = self.accelerator.prepare(
            policy,
            reference_model,
            train_loader,
            test_loader,
            test_unseen_loader,
            optimizer,
            scheduler
        )

    def _load_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dataset_kwargs = {
            'tokenizer': tokenizer,
            'max_length': self.config.max_length,
            'max_prompt_length': self.config.max_prompt_length,
            **self.config.dataset
        }
        sampler_kwargs = {
            'shuffle': True,
            'seed': self.config.seed,
            'n_epochs': self.config.n_epochs,
            'n_examples': self.config.n_examples
        }
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'collate_fn': get_collate_fn(tokenizer)
        }

        train_dataset = PreferenceDataset(split='train', **dataset_kwargs)
        train_sampler = PreferenceSampler(train_dataset, **sampler_kwargs)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)

        sampler_kwargs.pop('seed')
        sampler_kwargs.update({
            'shuffle': False,
            'n_epochs': 1,
            'n_examples': self.config.n_eval_examples
        })
        loader_kwargs.update({'batch_size': self.config.eval_batch_size})

        test_dataset = PreferenceDataset(split='test', **dataset_kwargs)
        test_sampler = PreferenceSampler(test_dataset, **sampler_kwargs)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)

        test_unseen_dataset = PreferenceDataset(split='test_unseen', **dataset_kwargs)
        test_unseen_sampler = PreferenceSampler(test_unseen_dataset, **sampler_kwargs)
        test_unseen_loader = DataLoader(test_unseen_dataset, sampler=test_unseen_sampler, **loader_kwargs)

        return train_loader, test_loader, test_unseen_loader

    def train(self):
        """Begin preference training with periodic evaluation."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        self.reference_model.eval()
        local_example_count = 0
        local_example_count_train = None

        for batch in tqdm(self.train_loader, desc='Training'):
            self.example_count = sum(gather_object([local_example_count]))

            if self.example_count % self.config.eval_every == 0:
                if self.example_count > 0:
                    end_time = perf_counter()
                    throughput = local_example_count_train / (end_time - start_time)
                    log_accelerate(f'Training throughput: {throughput:.4g} examples/s')

                test_metrics = self.evaluate(self.test_loader, split='test')
                test_unseen_metrics = self.evaluate(self.test_unseen_loader, split='test_unseen')
                test_metrics.update(test_unseen_metrics)

                start_time = perf_counter()
                local_example_count_train = 0

            self.policy.train()

            loss, _ = self._compute_loss_and_metrics(batch, split='train')
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            local_example_count += batch['prompt_input_ids'].size(dim=0)
            local_example_count_train += batch['prompt_input_ids'].size(dim=0)

        test_metrics = self.evaluate(self.test_loader, split='test')
        test_unseen_metrics = self.evaluate(self.test_unseen_loader, split='test_unseen')
        test_metrics.update(test_unseen_metrics)
        self.save('LATEST', test_metrics)

    @torch.no_grad()
    def evaluate(self, eval_loader, split):
        self.policy.eval()
        local_example_count = 0
        start_time = perf_counter()
        all_metrics = defaultdict(list)

        for batch in tqdm(eval_loader, desc=f'Evaluating on {split} split'):
            _, eval_metrics = self._compute_loss_and_metrics(batch, split)
            local_example_count += batch['prompt_input_ids'].size(dim=0)

            for key, value in eval_metrics.items():
                all_metrics[key].extend(value)

        end_time = perf_counter()
        throughput = local_example_count / (end_time - start_time)
        log_accelerate(f'Inference throughput: {throughput:.4g} examples/s')

        all_metrics = {key: sum(value) / len(value) for key, value in all_metrics.items()}
        formatted_metrics = {key: f'{value:.5g}' for key, value in all_metrics.items()}
        log_accelerate(
            f'{split} split result after {self.example_count} training examples:\n'
            f'{json.dumps(formatted_metrics, indent=4)}'
        )
        return all_metrics

    def _compute_loss_and_metrics(
        self,
        batch: dict[str, list[str] | torch.Tensor],
        split: str
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute the preference loss and other metrics for the batch of examples."""
        policy_chosen_logps, policy_rejected_logps = \
            self._forward_concatenated_responses(self.policy, batch)

        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = \
                self._forward_concatenated_responses(self.reference_model, batch)

        losses, chosen_rewards, rejected_rewards = _compute_preference_losses_and_rewards(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            **self.config.loss
        )
        loss = losses.mean()

        losses = losses.detach()
        policy_chosen_logps = policy_chosen_logps.detach()
        policy_rejected_logps = policy_rejected_logps.detach()

        policy_chosen_logps = self.accelerator.gather(policy_chosen_logps)
        policy_rejected_logps = self.accelerator.gather(policy_rejected_logps)
        chosen_rewards = self.accelerator.gather(chosen_rewards)
        rejected_rewards = self.accelerator.gather(rejected_rewards)
        losses = self.accelerator.gather(losses)

        logp_accuracies = (policy_chosen_logps > policy_rejected_logps).float()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = chosen_rewards - rejected_rewards

        metrics = {
            f'logp_{split}/chosen': policy_chosen_logps.tolist(),
            f'logp_{split}/rejected': policy_rejected_logps.tolist(),
            f'logp_{split}/accuracy': logp_accuracies.tolist(),
            f'reward_{split}/chosen': chosen_rewards.tolist(),
            f'reward_{split}/rejected': rejected_rewards.tolist(),
            f'reward_{split}/accuracy': reward_accuracies.tolist(),
            f'reward_{split}/margin': reward_margins.tolist(),
            f'loss/{split}': losses.tolist()
        }
        return loss, metrics

    def _forward_concatenated_responses(
        self,
        model: nn.Module,
        batch: dict[str, list[str] | torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass on the model with concatenated chosen and rejected responses."""
        concatenated_batch = _concatenate_responses(batch)

        if self.config.dataset.n_clusters is not None:
            self.controller.update_lora_weights(concatenated_batch['proximities'])

        logits = model(
            concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask']
        ).logits.to(torch.float32)
        logps = _compute_response_logps(logits, concatenated_batch['concatenated_labels'])

        chosen_logps = logps[:batch['chosen_input_ids'].size(dim=0)]
        rejected_logps = logps[batch['chosen_input_ids'].size(dim=0):]
        return chosen_logps, rejected_logps

    def save(self, version: str, metrics: dict) -> None:
        """Save the policy, optimizer, and scheduler states to disk."""
        output_dir = os.path.join(self.config.run_dir, version)
        os.makedirs(output_dir, exist_ok=True)
        log_accelerate(f'Writing checkpoints to {output_dir}...')

        self.accelerator.save(
            {
                'step': self.example_count,
                'metrics': metrics,
                'state': self.accelerator.unwrap_model(self.policy).state_dict()
            },
            os.path.join(output_dir, 'policy.pt')
        )
        self.accelerator.save(
            {'state': self.optimizer.state_dict()},
            os.path.join(output_dir, 'optimizer.pt')
        )
        self.accelerator.save(
            {'state': self.scheduler.state_dict()},
            os.path.join(output_dir, 'scheduler.pt')
        )


def _concatenate_responses(batch: dict[str, list[str] | torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concatenate the chosen and rejected responses along the batch dimension.
    Duplicate any other tensors to match the doubled batch size.

    Args:
        batch: A batch of examples that should contain the following fields:
            chosen_input_ids: Token IDs of the prompt + chosen response
                Shape: (batch_size, chosen_sequence_length)
            chosen_attention_mask: Attention mask of the prompt + chosen response
                Shape: (batch_size, chosen_sequence_length)
            chosen_labels: Vocabulary IDs of the chosen response tokens
                Shape: (batch_size, chosen_sequence_length)

            rejected_input_ids: Token IDs of the prompt + rejected response
                Shape: (batch_size, rejected_sequence_length)
            rejected_attention_mask: Attention mask of the prompt + rejected response
                Shape: (batch_size, rejected_sequence_length)
            rejected_labels: Vocabulary IDs of the rejected response tokens
                Shape: (batch_size, rejected_sequence_length)
    """
    chosen_length = batch['chosen_input_ids'].size(dim=-1)
    rejected_length = batch['rejected_input_ids'].size(dim=-1)
    longer_length = max(chosen_length, rejected_length)

    concatenated_batch = {}
    for key in batch:
        if not isinstance(batch[key], torch.Tensor):
            continue

        if key.startswith('chosen_'):
            rejected_key = key.replace('chosen_', 'rejected_')
            padding_value = -100 if 'labels' in key else 0

            padded_chosen = F.pad(
                batch[key],
                (0, longer_length - chosen_length),
                mode='constant',
                value=padding_value
            )
            padded_rejected = F.pad(
                batch[rejected_key],
                (0, longer_length - rejected_length),
                mode='constant',
                value=padding_value
            )

            new_key = key.replace('chosen_', 'concatenated_')
            new_value = torch.cat((padded_chosen, padded_rejected), dim=0)
            concatenated_batch[new_key] = new_value
        elif not key.startswith('rejected_'):
            concatenated_batch[key] = torch.cat((batch[key], batch[key]), dim=0)

    return concatenated_batch


def _compute_response_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the log probability for each response indicated by the labels based on the logits.

    Args:
        logits: Unnormalized probabilities of the next tokens.
            Shape: (batch_size, sequence_length, vocab_size)
        labels: Vocabulary IDs of the response tokens. Prompt and padding tokens are labeled with -100.
            Shape: (batch_size, sequence_length)
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    token_mask = labels != -100

    logps = logits.log_softmax(dim=-1)
    index = labels.masked_fill(~token_mask, value=0).unsqueeze(dim=-1)
    token_logps = torch.gather(logps, dim=-1, index=index).squeeze(dim=-1)

    response_logps = token_logps.masked_fill_(~token_mask, value=0.).sum(dim=-1)
    return response_logps


def _compute_preference_losses_and_rewards(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    loss: str,
    reference_free: bool,
    beta: float,
    epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the preference loss and reward for each preference pair based on the log probabilities
    evaluated by the policy and reference models.

    General Args:
        policy_chosen_logps: Log probabilities of the chosen responses evaluated by the policy model.
            Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the rejected responses evaluated by the policy model.
            Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the chosen responses evaluated by the reference model.
            Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the rejected responses evaluated by the reference model.
            Shape: (batch_size,)
        loss: The type of loss function for computing the preference loss.
        reference_free: If True, use a reference model that assigns equal probability to all responses instead.
        beta: Temperature, typically in the range of 0.1 to 0.5. The reference model is ignored as beta approaches 0.
              Also known as tau in Eq. 17 of https://arxiv.org/pdf/2310.12036.

    DPO Args:
        epsilon: Conservativeness, assuming that the preference labels are flipped with a probability of epsilon.
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        reference_logratios = 0.

    # Also known as h_\pi(y_w,y_l) in Eq. 17 of https://arxiv.org/pdf/2310.12036
    logits = policy_logratios - reference_logratios

    if loss == 'dpo':
        # Eq. 3 of https://ericmitchell.ai/cdpo.pdf
        # epsilon = 0 gives the original DPO loss (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -(1 - epsilon) * F.logsigmoid(beta * logits) - epsilon * F.logsigmoid(-beta * logits)
    elif loss == 'ipo':
        # Eq. 17 of https://arxiv.org/pdf/2310.12036
        losses = torch.square(logits - 1 / (2 * beta))
    else:
        raise ValueError(f'Unknown preference loss: {loss}')

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
