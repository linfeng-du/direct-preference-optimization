import torch
from transformers import AutoModelForCausalLM
from accelerate import Accelerator

import hydra
from omegaconf import DictConfig

from adapters import get_controller
from trainer import AccelerateTrainer
from utils import disable_dropout, log_main_process, log_all_processes


accelerator = Accelerator()
torch.backends.cuda.matmul.allow_tf32 = True


def accelerate_main(config, policy, controller, reference_model):
    log_all_processes(f'Creating trainer on process {accelerator.process_index} ' \
                      f'with world size {accelerator.num_processes}...')

    trainer = AccelerateTrainer(
        config,
        policy,
        controller,
        reference_model,
        accelerator,
    )
    accelerator.wait_for_everyone()
    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(config: DictConfig):
    """Main entry point for training. Validate config, initialize model(s), and kick off training."""
    if config.eval_every % config.batch_size != 0:
        eval_every = config.eval_every - config.eval_every % config.batch_size
        log_main_process(f'Setting eval_every to {eval_every}...', level='warning')
        config.eval_every = eval_every

    log_main_process('Building policy...')

    policy = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=getattr(torch, config.model.policy_dtype),
        low_cpu_mem_usage=True
    )
    disable_dropout(policy)

    if config.loss.loss in {'dpo', 'ipo'}:
        log_main_process('Building reference model...')

        reference_model = AutoModelForCausalLM.from_pretrained(
            config.model.model,
            torch_dtype=getattr(torch, config.model.reference_dtype),
            low_cpu_mem_usage=True
        )
        disable_dropout(reference_model)

    log_main_process(f'Inserting {config.adapter.adapter} adapters...')

    controller = get_controller(**config.adapter)
    controller.insert_adapters(policy, config.model.target_modules)

    accelerate_main(config, policy, controller, reference_model)


if __name__ == '__main__':
    main()
