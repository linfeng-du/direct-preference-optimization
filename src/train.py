import shutil

import torch
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

import hydra
from omegaconf import OmegaConf, DictConfig

accelerator = Accelerator()
torch.backends.cuda.matmul.allow_tf32 = True

from adapters import get_controller
from trainer import AccelerateTrainer
from utils import disable_dropout, count_trainable_parameters, log_accelerate


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(config: DictConfig):
    """Main entry point for training. Validate config, initialize models, and kick off training."""
    if accelerator.is_main_process:
        object_list = [config]
    else:
        shutil.rmtree(config.run_dir)
        object_list = [None]

    broadcast_object_list(object_list, from_process=0)
    config = object_list[0]

    missing_keys = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    if config.eval_every % config.batch_size != 0:
        eval_every = config.eval_every - config.eval_every % config.batch_size
        log_accelerate(
            f'Changing eval_every from {config.eval_every} to {eval_every} '
            f'to make it divisible by the batch size {config.batch_size}...',
            level='warning'
        )
        config.eval_every = eval_every

    log_accelerate('Building policy...')

    policy = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=getattr(torch, config.model.policy_dtype),
        low_cpu_mem_usage=True
    )
    disable_dropout(policy)

    log_accelerate(f'Inserting {config.adapter.adapter} adapters...')

    controller = get_controller(**config.adapter)
    controller.insert_adapters(policy, config.model.target_modules)

    num_params = count_trainable_parameters(policy)
    log_accelerate(f'Number of trainable parameters: {num_params}')

    log_accelerate('Building reference model...')

    reference_model = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=getattr(torch, config.model.reference_dtype),
        low_cpu_mem_usage=True
    )
    disable_dropout(reference_model)

    log_accelerate(
        f'Creating trainer on process {accelerator.process_index} '
        f'with world size {accelerator.num_processes}...',
        on_all_processes=True
    )
    trainer = AccelerateTrainer(
        config=config,
        policy=policy,
        controller=controller,
        reference_model=reference_model,
        accelerator=accelerator
    )
    trainer.train()


if __name__ == '__main__':
    main()
