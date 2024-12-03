import json

import torch
from transformers import AutoModelForCausalLM
from accelerate import Accelerator, PartialState

import hydra
from omegaconf import DictConfig

from trainers import AccelerateTrainer
from adaptors import get_controller_class
from utils import disable_dropout, log_main_process, log_all_processes


torch.backends.cuda.matmul.allow_tf32 = True
accelerator = Accelerator()


def accelerate_main(controller, config, reference_model=None):
    log_all_processes(f'Creating trainer on process {accelerator.process_index} ' \
                      f'with world size {accelerator.num_processes}...')

    trainer = AccelerateTrainer(
        accelerator,
        controller,
        config,
        reference_model=reference_model
    )
    accelerator.wait_for_everyone()
    trainer.train()
    trainer.save()


@PartialState().on_main_process
@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    """Main entry point for training.

    Validate config, create/initialize model(s), and kick off training.
    """
    if config.eval_every % config.batch_size != 0:
        eval_every = config.eval_every - config.eval_every % config.batch_size
        log_main_process(f'Setting eval_every to {eval_every}...', level='warning')
        config.eval_every = eval_every

    log_main_process('Building policy...')

    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=policy_dtype,
        low_cpu_mem_usage=True
    )
    disable_dropout(policy)

    if config.loss.name in {'dpo', 'ipo'}:
        log_main_process('Building reference model...')

        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=reference_model_dtype,
            low_cpu_mem_usage=True
        )
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        log_main_process(f'Loading pre-trained weights ' \
                         f'at step {step} from {config.model.archive} ' \
                         f'with metrics {json.dumps(metrics)}...')

        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])

    log_main_process(f'Inserting {config.adaptor.name} adaptors...')

    ControllerClass = get_controller_class(config.adaptor.name)
    controller = ControllerClass(policy)
    controller.insert_adaptors(config.model.target_modules, **config.adaptor.args)

    accelerate_main(controller, config, reference_model=reference_model)


if __name__ == '__main__':
    main()
