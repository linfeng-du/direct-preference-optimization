import shutil

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

import hydra
from omegaconf import OmegaConf, DictConfig

accelerator = Accelerator()
torch.backends.cuda.matmul.allow_tf32 = True

from adapters import get_controller
from trainer import AccelerateTrainer
from utils import log_main_process


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
        log_main_process(f'Setting eval_every to {eval_every}...', level='warning')
        config.eval_every = eval_every

    log_main_process('Building policy...')

    policy = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=getattr(torch, config.model.policy_dtype),
        low_cpu_mem_usage=True
    )
    for module in policy.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.

    log_main_process(f'Inserting {config.adapter.adapter} adapters...')

    controller = get_controller(**config.adapter)
    controller.insert_adapters(policy, config.model.target_modules)

    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if num_params >= 1e9:
        num_params = f'{num_params / 1e9:.2f}B'
    elif num_params >= 1e6:
        num_params = f'{num_params / 1e6:.2f}M'
    else:
        num_params = f'{num_params:,}'

    log_main_process(f'Number of trainable parameters: {num_params}')

    log_main_process('Building reference model...')

    reference_model = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=getattr(torch, config.model.reference_dtype),
        low_cpu_mem_usage=True
    )
    for module in reference_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.

    trainer = AccelerateTrainer(config, policy, controller, reference_model, accelerator)
    trainer.train()


if __name__ == '__main__':
    main()
