import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import hydra
from omegaconf import OmegaConf, DictConfig

from adapters import get_controller
from trainer import AccelerateTrainer
from utils import log_main_process


torch.backends.cuda.matmul.allow_tf32 = True


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(config: DictConfig):
    """Main entry point for training. Validate config, initialize models, and kick off training."""
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

    log_main_process('Building reference model...')

    reference_model = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=getattr(torch, config.model.reference_dtype),
        low_cpu_mem_usage=True
    )
    for module in reference_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.

    log_main_process(f'Inserting {config.adapter.adapter} adapters...')

    controller = get_controller(**config.adapter)
    controller.insert_adapters(policy, config.model.target_modules)

    trainer = AccelerateTrainer(config, policy, controller, reference_model)
    trainer.train()


if __name__ == '__main__':
    main()
