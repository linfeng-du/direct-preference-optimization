from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import peft
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
from hnet.lora import LoRA_controller, LoRA
from transformers import LlamaModel, LlamaConfig,LlamaForCausalLM
from hnet.hypernet import HyperNetController, HyperNet, HyperNetLinear,PolicyWrapper

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


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""

    if config.use_hnet: 
            kvq = 3 
            a_b = 2 
            configuration = policy.config
            d_model = configuration.hidden_size
            output_dim = configuration.num_hidden_layers * (config.hnet.d_a *  config.hnet.d_A ) * a_b   *  kvq
            hypernet,controller = get
            hypernet = HyperNet(config.hnet.alpha, config.hnet.dropout, config.hnet.d_A,config.hnet.d_a,config.hnet.d_hnet,output_dim,config.hnet.d_emb,  config.hnet.n_transformer_layers,configuration.num_hidden_layers ,config.hnet.n_transformer_heads)
            controller = HyperNetController(config,hypernet,target_modules = config.model.target_modules, 
                                            d_model = d_model,
                                            A = config.hnet.d_A,
                                            a = config.hnet.d_a)
            controller.augmentLLM(policy)
            controller.freezeParams(policy, False)
            policy = PolicyWrapper(hypernet,policy)
            #print number of trainable parameters 

            print_trainable_params(hypernet)
    elif config.use_lora:
        controller = LoRA_controller(config.lora,config.lora.r,target_modules = config.model.target_modules)
        

    else:
        controller = None
        hypernet = None
    print_trainable_params(policy)
    
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = cache_dir
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)

    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size,hnet_controller = controller)
    if config.use_lora:
        controller.augmentLLM(trainer.policy)
        controller.freezeParams(trainer.policy, False)
        print('Using Lora')

    
    trainer.train()
    if config.save:
        trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)
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

    print(OmegaConf.to_yaml(config))
    # assert config.use_hnet != config.use_lora, 'Cannot use both Hnet and Lora at the same time, siable use_lora or use_hnet'

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    if config.use_lora:
        device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}
    else:
        device_map = {'device_map': 'balanced'}
    model_kwargs = device_map if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    if config.use_hnet and config.debug:
        

        # Initializing a LLaMA llama-7b style configuration
        configuration = LlamaConfig()
        configuration.num_hidden_layers = 2
        configuration.num_attention_heads = 2
        configuration.num_key_value_heads = 2
        policy = LlamaForCausalLM(configuration)
        d_model = configuration.hidden_size
        kvq = 3 
        a_b = 2 
        d_model = configuration.hidden_size
        output_dim = configuration.num_hidden_layers * (config.hnet.d_a *  config.hnet.d_A ) * a_b   *  kvq

        hypernet = HyperNet(config.hnet.alpha, config.hnet.dropout, config.hnet.d_A,config.hnet.d_a,output_dim,config.hnet.d_emb,  config.hnet.n_transformer_layers,configuration.num_hidden_layers ,config.hnet.n_transformer_heads)
        controller = HyperNetController(hypernet,target_modules =config.model.target_modules, 
                                        d_model = d_model,
                                        A = config.hnet.d_A,
                                        a = config.hnet.d_a)
        controller.augmentLLM(policy)
        controller.freezeParams(policy, False)
        # user_embeddings = np.load('/home/mila/e/emiliano.penaloza/direct-preference-optimization/notebooks/data/user_embeddings.npy').tolist()
        # config.user_embeddings = user_embeddings


    else:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=cache_dir, low_cpu_mem_usage=True, torch_dtype=policy_dtype,device_map= 'balanced')
        #print number of parameters in policy
        print_trainable_params(policy)
        # if config.use_lora:
        #     lora_config = LoraConfig(
        #                     r=config.lora.r,
        #                     lora_alpha=32,
        #                     target_modules=config.model.target_modules,
        #                     lora_dropout=0.,
        #                     bias="none",
        #                     task_type="CAUSAL_LM"
        #                     )
        #     policy = get_peft_model(policy, lora_config)
        #     print('Using Lora')
        # if config.use_hnet: 
        #     kvq = 3 
        #     a_b = 2 
        #     configuration = policy.config
        #     d_model = configuration.hidden_size
        #     output_dim = configuration.num_hidden_layers * (config.hnet.d_a *  config.hnet.d_A ) * a_b   *  kvq
            
        #     hypernet = HyperNet(config.hnet.alpha, config.hnet.dropout, config.hnet.d_A,config.hnet.d_a,output_dim,config.hnet.d_emb,  config.hnet.n_transformer_layers,configuration.num_hidden_layers ,config.hnet.n_transformer_heads)
        #     controller = HyperNetController(hypernet,target_modules = ['query_key_value'], 
        #                                     d_model = d_model,
        #                                     A = config.hnet.d_A,
        #                                     a = config.hnet.d_a)
        #     controller.augmentLLM(policy)
        #     controller.freezeParams(policy, False)

            
    disable_dropout(policy)
    #policy forward pass base llm + adaptors
    #reference policy is being loaded into memory on its own
    #reference policy = policy - adaptors
    #reference policy = policy with adaptors disabled

    if config.loss.name in {'dpo', 'ipo'} :
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=cache_dir, low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, device_map= 'balanced')
        disable_dropout(reference_model)
    elif config.debug:
        
        configuration.num_hidden_layers = 2
        configuration.num_attention_heads = 16
        configuration.num_key_value_heads = 2
        reference_model = LlamaForCausalLM(configuration)
        
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
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.set_start_method('forkserver')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()