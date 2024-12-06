import os
import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator

from tqdm import tqdm
from omegaconf import DictConfig

from adapters import Controller
from dataset import PreferenceDataset, PreferenceSampler, get_collate_fn
from utils import formatted_dict, pad_to_length, log_main_process


class AccelerateTrainer:

    def __init__(
        self,
        config: DictConfig,
        policy: nn.Module,
        controller: Controller | None,
        reference_model: nn.Module | None,
        accelerator: Accelerator
    ):
        """Trainer that leverages Hugging Face Accelerate for distributed training."""
        self.config = config
        self.policy = policy
        self.accelerator = accelerator
        self.controller = controller
        self.reference_model = reference_model

        log_main_process('Loading tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log_main_process('Loading dataset')
        self.train_dataset = PreferenceDataset(
            dataset=config.dataset,
            split='train',
            tokenizer=self.tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            prepend_persona=config.prepend_persona,
            n_clusters=config.n_clusters
        )
        train_sampler = PreferenceSampler(
            dataset=self.train_dataset,
            shuffle=True,
            seed=config.seed,
            n_epochs=config.n_epochs,
            n_examples=config.n_examples
        )
        self.train_iterator = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            collate_fn=get_collate_fn(self.tokenizer)
        )

        self.eval_dataset = PreferenceDataset(
            dataset=config.dataset,
            split='test',
            tokenizer=self.tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            prepend_persona=config.prepend_persona,
            n_clusters=config.n_clusters
        )
        eval_sampler = PreferenceSampler(self.eval_dataset, shuffle=False, n_epochs=1)
        self.eval_iterator = DataLoader(
            self.eval_dataset,
            batch_size=config.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=get_collate_fn(self.tokenizer)
        )

        self.test_dataset = PreferenceDataset(
            dataset=config.dataset,
            split='test_unseen',
            tokenizer=self.tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            prepend_persona=config.prepend_persona,
            n_clusters=config.n_clusters
        )
        test_sampler = PreferenceSampler(self.test_dataset, shuffle=False, n_epochs=1)
        self.test_iterator = DataLoader(
            self.eval_dataset,
            batch_size=config.eval_batch_size,
            sampler=test_sampler,
            collate_fn=get_collate_fn(self.tokenizer)
        )

        self.optimizer_non_loaded = getattr(
            torch.optim, self.config.optimizer
        )([p for p in self.policy.parameters() if p.requires_grad], lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_non_loaded,
            lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1))
        )

        self.policy, self.train_iterator, self.optimizer, self.scheduler = \
            accelerator.prepare(self.policy, self.train_iterator, self.optimizer_non_loaded, self.scheduler)
        self.reference_model = accelerator.prepare_model(self.reference_model) if self.reference_model is not None else None
        self.eval_iterator = accelerator.prepare(self.eval_iterator)
        self.test_iterator = accelerator.prepare(self.test_iterator)
    
    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if self.config.loss.loss in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0

        for batch in tqdm(self.train_iterator):
            #### BEGIN TRAINING ####
            self.policy.train()

            batch_metrics = defaultdict(list)

            for _ in range(self.config.gradient_accumulation_steps):
                loss, metrics = self.get_batch_metrics(batch, self.config.loss, train=True)
                self.first_pass = False
                l = (loss / self.config.gradient_accumulation_steps)
                self.accelerator.backward(l)
                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.example_counter += self.config.batch_size
            #### END TRAINING ####

            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0:
                log_main_process(f'Running evaluation after {self.example_counter} train examples')
                self.evaluate()
            #### END EVALUATION ####

        self.evaluate()

    def evaluate(self):
        self.policy.eval()
        all_eval_metrics = defaultdict(list)

        for eval_batch in tqdm(self.eval_iterator, desc='Computing eval metrics'):
            local_eval_batch = eval_batch
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        log_main_process(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

        for eval_batch in tqdm(self.test_iterator, desc='Computing test metrics'):
            local_eval_batch = eval_batch
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        mean_test_metrics = {f'unseen_{k}': sum(v) / len(v) for k, v in all_eval_metrics.items()}
        log_main_process(f'eval after {self.example_counter}: {formatted_dict(mean_test_metrics)}')

        mean_eval_metrics.update(mean_test_metrics)

        if self.example_counter > 0:
            output_dir = os.path.join(self.config.run_dir, f'step-{self.example_counter}')
            log_main_process(f'creating checkpoint to write to {output_dir}...')
            if self.config.save:
                self.save(output_dir, mean_eval_metrics)

    def get_batch_metrics(self, batch: dict[str, list | torch.Tensor], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""
        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.loss in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)

            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            losses, chosen_rewards, rejected_rewards = _compute_preference_losses(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                **loss_config
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = self.accelerator.gather(chosen_rewards)
            rejected_rewards = self.accelerator.gather(rejected_rewards)
            reward_accuracies = self.accelerator.gather(reward_accuracies)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            policy_rejected_logps = self.accelerator.gather(policy_rejected_logps.detach())
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        policy_chosen_logps = self.accelerator.gather(policy_chosen_logps.detach())
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()
        all_devices_losses = self.accelerator.gather(losses.detach())
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def concatenated_forward(self, model: nn.Module, batch: dict[str, list | torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = _concatenate_responses(batch)

        if self.config.n_clusters is not None:
            self.controller.update_lora_weights(concatenated_batch['proximities'])

        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _compute_response_logps(all_logits, concatenated_batch['concatenated_labels'])
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def save(self, output_dir: str | None = None, metrics: dict | None = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)

    def write_state_dict(self, step: int, state: dict[str, torch.Tensor], metrics: dict, filename: str, dir_name: str | None = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.config.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        log_main_process(f'writing checkpoint to {output_path}...')
        torch.save({
            'step': step,
            'metrics': metrics if metrics is not None else {},
            'state': state
        }, output_path)


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

            padded_chosen = pad_to_length(batch[key], longer_length, padding_value)
            padded_rejected = pad_to_length(batch[rejected_key], longer_length, padding_value)

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
    index = labels.masked_fill(~token_mask, 0).unsqueeze(dim=-1)
    token_logps = torch.gather(logps, dim=-1, index=index).squeeze(dim=-1)

    response_logps = token_logps.masked_fill_(~token_mask, 0.).sum(dim=-1)
    return response_logps


def _compute_preference_losses(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    loss: str,
    reference_free: bool,
    beta: float,
    epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the preference loss for each preference pair based on the log probabilities
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

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
