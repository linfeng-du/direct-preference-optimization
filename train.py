import os
import json
import socket
import resource
from typing import Optional, Set

import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import torch.multiprocessing as mp
import transformers
from peft import LoraConfig, get_peft_model

from accelerate import Accelerator


import trainers
from hnet.utils import get_module
from hnet.hypernet import PolicyWrapper
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port


scratch_dir = os.getenv('SCRATCH')
os.environ['TRANSFORMERS_CACHE'] = f'{scratch_dir}/models'
os.environ['HF_HOME'] = f'{scratch_dir}/models'
os.environ['HF_DATASETS_CACHE'] = f'{scratch_dir}/models'
os.environ['TORCH_HOME'] = f'{scratch_dir}/models'
cache_dir = f'{scratch_dir}/models'


def print_trainable_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e9:
        formatted_params = f'{num_params / 1e9:.2f}B'
    elif num_params >= 1e6:
        formatted_params = f'{num_params / 1e6:.2f}M'
    else:
        formatted_params = f'{num_params:,}'
    
    print(f'Number of trainable parameters: {formatted_params}')


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(accelerator,rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if config.use_hnet and not config.train_reward: 
            configuration = policy.config
            config.hnet.n_layers = configuration.num_hidden_layers

            hypernet,controller = get_module(config.hnet_type,config)
            controller.augmentLLM(policy)

            controller.freezeParams(policy, False)
            policy = PolicyWrapper(hypernet,policy)

            print_trainable_params(policy)
    else:
        controller = None
        hypernet = None

    print_trainable_params(policy)

    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = cache_dir
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    # THIS WILL BE BASIC TRAINER ALWAYS 
    TrainerClass = getattr(trainers, config.trainer)

    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(accelerator, policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size,hnet_controller = controller)

    #accelerator wait for all process 
    accelerator.wait_for_everyone()

    trainer.train()
    if config.save:
        trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    accelerator = Accelerator()
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)
    #=====================IGNORE=====================
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port
    #=====================IGNORE END=====================

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    #=====================IGNORE=====================

    if config.use_lora:
        device_map = {'device_map': 'balanced'}
    else:
        device_map = {'device_map': 'balanced'}
    model_kwargs = device_map if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    
    #=====================IGNORE END=====================

    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=cache_dir, low_cpu_mem_usage=True, torch_dtype=policy_dtype, device_map=None)

    # Print number of parameters in policy
    print_trainable_params(policy)
    if config.use_lora:
        lora_config = LoraConfig(
                        r=config.lora.r,
                        lora_alpha=32,
                        target_modules=config.model.target_modules,
                        lora_dropout=0.,
                        bias="none",
                        task_type="CAUSAL_LM"
                        )
        policy = get_peft_model(policy, lora_config)
        print('Using Lora')

    disable_dropout(policy)
    #policy forward pass base llm + adaptors
    #reference policy is being loaded into memory on its own
    #reference policy = policy - adaptors
    #reference policy = policy with adaptors disabled

    if config.loss.name in {'dpo', 'ipo'} :
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=cache_dir, low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, device_map=None)
        disable_dropout(reference_model)        
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    #=====================IGNORE=====================
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.set_start_method('forkserver')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    #=====================IGNORE END=====================
 
    else:
        print('starting single-process worker')
        worker_main(accelerator,accelerator.device.index, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
