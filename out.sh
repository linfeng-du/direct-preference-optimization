[2024-07-10 16:17:18,804] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 16:17:18,837] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 16:17:18,842] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 16:17:18,843] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 16:17:21,708] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 16:17:21,710][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 2
[2024-07-10 16:17:21,802] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 16:17:21,803][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
[2024-07-10 16:17:21,829] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 16:17:21,830][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 3
[2024-07-10 16:17:21,833] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 16:17:21,833] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[2024-07-10 16:17:21,834][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2024-07-10 16:17:21,834][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2024-07-10 16:17:21,841][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2024-07-10 16:17:21,842][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2024-07-10 16:17:21,844][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
seed: 0
exp_name: hnet_dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: true
wandb:
  enabled: true
  entity: null
  project: direct-preference-optimization
local_dirs:
- /scr-ssd
- /scr
- .cache
sample_during_eval: false
n_eval_model_samples: 0
do_first_eval: true
local_run_dir: .cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_886373
lr: 0.05
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 2
n_examples: null
n_eval_examples: 256
trainer: BasicTrainer
optimizer: AdamW
warmup_steps: 150
activation_checkpointing: false
eval_every: 4960
minimum_log_interval_secs: 0.0
save: false
model:
  name_or_path: vicgalle/gpt2-open-instruct-v1
  tokenizer_name_or_path: vicgalle/gpt2-open-instruct-v1
  archive: null
  transpose_AB: false
  block_name: GPT2Block
  policy_dtype: float32
  fsdp_policy_mp: null
  reference_dtype: float16
  target_modules:
  - c_attn
loss:
  name: dpo
  beta: 0.3
  label_smoothing: 0
  reference_free: false
hnet:
  d_a: 16
  alpha: 16
  d_A: 64
  n_transformer_heads: 2
  n_transformer_layers: 8
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_886373
================================================================================
building policy
WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
seed: 0
exp_name: hnet_dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: true
wandb:
  enabled: true
  entity: null
  project: direct-preference-optimization
local_dirs:
- /scr-ssd
- /scr
- .cache
sample_during_eval: false
n_eval_model_samples: 0
do_first_eval: true
local_run_dir: .cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_899398
lr: 0.05
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 2
n_examples: null
n_eval_examples: 256
trainer: BasicTrainer
optimizer: AdamW
warmup_steps: 150
activation_checkpointing: false
eval_every: 4960
minimum_log_interval_secs: 0.0
save: false
model:
  name_or_path: vicgalle/gpt2-open-instruct-v1
  tokenizer_name_or_path: vicgalle/gpt2-open-instruct-v1
  archive: null
  transpose_AB: false
  block_name: GPT2Block
  policy_dtype: float32
  fsdp_policy_mp: null
  reference_dtype: float16
  target_modules:
  - c_attn
loss:
  name: dpo
  beta: 0.3
  label_smoothing: 0
  reference_free: false
hnet:
  d_a: 16
  alpha: 16
  d_A: 64
  n_transformer_heads: 2
  n_transformer_layers: 8
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
seed: 0
exp_name: hnet_dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: true
wandb:
  enabled: true
  entity: null
  project: direct-preference-optimization
local_dirs:
- /scr-ssd
- /scr
- .cache
sample_during_eval: false
n_eval_model_samples: 0
do_first_eval: true
local_run_dir: .cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_903328
lr: 0.05
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 2
n_examples: null
n_eval_examples: 256
trainer: BasicTrainer
optimizer: AdamW
warmup_steps: 150
activation_checkpointing: false
eval_every: 4960
minimum_log_interval_secs: 0.0
save: false
model:
  name_or_path: vicgalle/gpt2-open-instruct-v1
  tokenizer_name_or_path: vicgalle/gpt2-open-instruct-v1
  archive: null
  transpose_AB: false
  block_name: GPT2Block
  policy_dtype: float32
  fsdp_policy_mp: null
  reference_dtype: float16
  target_modules:
  - c_attn
loss:
  name: dpo
  beta: 0.3
  label_smoothing: 0
  reference_free: false
hnet:
  d_a: 16
  alpha: 16
  d_A: 64
  n_transformer_heads: 2
  n_transformer_layers: 8
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

seed: 0
exp_name: hnet_dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: true
wandb:
  enabled: true
  entity: null
  project: direct-preference-optimization
local_dirs:
- /scr-ssd
- /scr
- .cache
sample_during_eval: false
n_eval_model_samples: 0
do_first_eval: true
local_run_dir: .cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_903832
lr: 0.05
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 2
n_examples: null
n_eval_examples: 256
trainer: BasicTrainer
optimizer: AdamW
warmup_steps: 150
activation_checkpointing: false
eval_every: 4960
minimum_log_interval_secs: 0.0
save: false
model:
  name_or_path: vicgalle/gpt2-open-instruct-v1
  tokenizer_name_or_path: vicgalle/gpt2-open-instruct-v1
  archive: null
  transpose_AB: false
  block_name: GPT2Block
  policy_dtype: float32
  fsdp_policy_mp: null
  reference_dtype: float16
  target_modules:
  - c_attn
loss:
  name: dpo
  beta: 0.3
  label_smoothing: 0
  reference_free: false
hnet:
  d_a: 16
  alpha: 16
  d_A: 64
  n_transformer_heads: 2
  n_transformer_layers: 8
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_899398
================================================================================
building policy
================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_903328
================================================================================
building policy
================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_903832
================================================================================
building policy
Number of trainable parameters: 124.44M
building reference model
Number of trainable parameters: 124.44M
building reference model
Number of trainable parameters: 124.44M
building reference model
Number of trainable parameters: 124.44M
building reference model
starting single-process worker
key='transformer.h.0.attn.c_attn'
key='transformer.h.1.attn.c_attn'
key='transformer.h.2.attn.c_attn'
key='transformer.h.3.attn.c_attn'
key='transformer.h.4.attn.c_attn'
key='transformer.h.5.attn.c_attn'
key='transformer.h.6.attn.c_attn'
key='transformer.h.7.attn.c_attn'
starting single-process worker
key='transformer.h.8.attn.c_attn'
starting single-process worker
starting single-process worker
key='transformer.h.0.attn.c_attn'
key='transformer.h.0.attn.c_attn'
key='transformer.h.0.attn.c_attn'
key='transformer.h.9.attn.c_attn'
key='transformer.h.10.attn.c_attn'
key='transformer.h.11.attn.c_attn'
Number of trainable parameters: 3.26M
Number of trainable parameters: 3.26M
Creating trainer on process 3 with world size 1
name='prism'
key='transformer.h.1.attn.c_attn'
key='transformer.h.1.attn.c_attn'
key='transformer.h.1.attn.c_attn'
key='transformer.h.2.attn.c_attn'
key='transformer.h.2.attn.c_attn'
key='transformer.h.2.attn.c_attn'
key='transformer.h.3.attn.c_attn'
key='transformer.h.3.attn.c_attn'
key='transformer.h.3.attn.c_attn'
key='transformer.h.4.attn.c_attn'
key='transformer.h.4.attn.c_attn'
key='transformer.h.4.attn.c_attn'
key='transformer.h.5.attn.c_attn'
key='transformer.h.5.attn.c_attn'
key='transformer.h.5.attn.c_attn'
key='transformer.h.6.attn.c_attn'
key='transformer.h.6.attn.c_attn'
key='transformer.h.6.attn.c_attn'
key='transformer.h.7.attn.c_attn'
key='transformer.h.7.attn.c_attn'
key='transformer.h.7.attn.c_attn'
key='transformer.h.8.attn.c_attn'
key='transformer.h.8.attn.c_attn'
key='transformer.h.8.attn.c_attn'
key='transformer.h.9.attn.c_attn'
key='transformer.h.9.attn.c_attn'
key='transformer.h.9.attn.c_attn'
key='transformer.h.10.attn.c_attn'
key='transformer.h.10.attn.c_attn'
key='transformer.h.10.attn.c_attn'
key='transformer.h.11.attn.c_attn'
key='transformer.h.11.attn.c_attn'
key='transformer.h.11.attn.c_attn'
Number of trainable parameters: 3.26M
Number of trainable parameters: 3.26M
Creating trainer on process 1 with world size 1
Number of trainable parameters: 3.26M
Number of trainable parameters: 3.26M
Creating trainer on process 2 with world size 1
Number of trainable parameters: 3.26M
Number of trainable parameters: 3.26M
name='prism'
name='prism'
Creating trainer on process 0 with world size 1
Loading tokenizer vicgalle/gpt2-open-instruct-v1
Loaded train data iterator
name='prism'
Finished generating 2 epochs on train split
name='prism'
Finished generating 2 epochs on train split
name='prism'
Finished generating 2 epochs on train split
name='prism'
Finished generating 2 epochs on train split
name='prism'
Finished generating 1 epochs on test split
Finished generating 1 epochs on test split
Finished generating 1 epochs on test split
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
ninja: no work to do.
Time to load cpu_adam op: 2.592151403427124 seconds
Finished generating 1 epochs on test split
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.050000, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
[2024-07-10 16:17:58,180][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 3
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
ninja: no work to do.
Time to load cpu_adam op: 2.5654172897338867 seconds
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
ninja: no work to do.
Time to load cpu_adam op: 2.5103812217712402 seconds
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.050000, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
[2024-07-10 16:17:59,579][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 2
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.050000, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
[2024-07-10 16:17:59,845][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 1
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
ninja: no work to do.
Time to load cpu_adam op: 2.6140623092651367 seconds
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.050000, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
[2024-07-10 16:18:01,485] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed info: version=0.14.0, git-hash=unknown, git-branch=unknown
[2024-07-10 16:18:01,731][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 0
[2024-07-10 16:18:01,731][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 16:18:01,731][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 16:18:01,734][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 16:18:01,735][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 16:18:02,550] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2024-07-10 16:18:02,551] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2024-07-10 16:18:02,552] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2024-07-10 16:18:02,555] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = DeepSpeedCPUAdam
[2024-07-10 16:18:02,555] [INFO] [utils.py:56:is_zero_supported_optimizer] Checking ZeRO support for optimizer=DeepSpeedCPUAdam type=<class 'deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam'>
[2024-07-10 16:18:02,555] [INFO] [logging.py:96:log_dist] [Rank 0] Creating torch.bfloat16 ZeRO stage 2 optimizer
[2024-07-10 16:18:02,556] [INFO] [stage_1_and_2.py:149:__init__] Reduce bucket size 500,000,000
[2024-07-10 16:18:02,556] [INFO] [stage_1_and_2.py:150:__init__] Allgather bucket size 500,000,000
[2024-07-10 16:18:02,556] [INFO] [stage_1_and_2.py:151:__init__] CPU Offload: True
[2024-07-10 16:18:02,556] [INFO] [stage_1_and_2.py:152:__init__] Round robin gradient partitioning: False
[2024-07-10 16:18:03,254] [INFO] [utils.py:800:see_memory_usage] Before initializing optimizer states
[2024-07-10 16:18:03,256] [INFO] [utils.py:801:see_memory_usage] MA 0.26 GB         Max_MA 0.26 GB         CA 0.28 GB         Max_CA 0 GB 
[2024-07-10 16:18:03,256] [INFO] [utils.py:808:see_memory_usage] CPU Virtual Memory:  used = 40.3 GB, percent = 4.0%
[2024-07-10 16:18:03,819] [INFO] [utils.py:800:see_memory_usage] After initializing optimizer states
[2024-07-10 16:18:03,820] [INFO] [utils.py:801:see_memory_usage] MA 0.26 GB         Max_MA 0.26 GB         CA 0.28 GB         Max_CA 0 GB 
[2024-07-10 16:18:03,820] [INFO] [utils.py:808:see_memory_usage] CPU Virtual Memory:  used = 40.3 GB, percent = 4.0%
[2024-07-10 16:18:03,820] [INFO] [stage_1_and_2.py:542:__init__] optimizer state initialized
[2024-07-10 16:18:04,380] [INFO] [utils.py:800:see_memory_usage] After initializing ZeRO optimizer
[2024-07-10 16:18:04,381] [INFO] [utils.py:801:see_memory_usage] MA 0.26 GB         Max_MA 0.26 GB         CA 0.28 GB         Max_CA 0 GB 
[2024-07-10 16:18:04,381] [INFO] [utils.py:808:see_memory_usage] CPU Virtual Memory:  used = 40.31 GB, percent = 4.0%
[2024-07-10 16:18:04,383] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = DeepSpeedCPUAdam
[2024-07-10 16:18:04,383] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2024-07-10 16:18:04,383] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2024-07-10 16:18:04,383] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.00033112582781456954], mom=[(0.9, 0.999)]
[2024-07-10 16:18:04,384] [INFO] [config.py:996:print] DeepSpeedEngine configuration:
[2024-07-10 16:18:04,384] [INFO] [config.py:1000:print]   activation_checkpointing_config  {
    "partition_activations": false, 
    "contiguous_memory_optimization": false, 
    "cpu_checkpointing": false, 
    "number_checkpoints": null, 
    "synchronize_checkpoint_boundary": false, 
    "profile": false
}
[2024-07-10 16:18:04,384] [INFO] [config.py:1000:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2024-07-10 16:18:04,384] [INFO] [config.py:1000:print]   amp_enabled .................. False
[2024-07-10 16:18:04,384] [INFO] [config.py:1000:print]   amp_params ................... False
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   autotuning_config ............ {
    "enabled": false, 
    "start_step": null, 
    "end_step": null, 
    "metric_path": null, 
    "arg_mappings": null, 
    "metric": "throughput", 
    "model_info": null, 
    "results_dir": "autotuning_results", 
    "exps_dir": "autotuning_exps", 
    "overwrite": true, 
    "fast": true, 
    "start_profile_step": 3, 
    "end_profile_step": 5, 
    "tuner_type": "gridsearch", 
    "tuner_early_stopping": 5, 
    "tuner_num_trials": 50, 
    "model_info_path": null, 
    "mp_size": 1, 
    "max_train_batch_size": null, 
    "min_train_batch_size": 1, 
    "max_train_micro_batch_size_per_gpu": 1.024000e+03, 
    "min_train_micro_batch_size_per_gpu": 1, 
    "num_tuning_micro_batch_sizes": 3
}
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   bfloat16_enabled ............. True
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   bfloat16_immediate_grad_update  False
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   checkpoint_parallel_write_pipeline  False
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   checkpoint_tag_validation_enabled  True
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   checkpoint_tag_validation_fail  False
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7f7d416893c0>
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   communication_data_type ...... None
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   compile_config ............... enabled=False backend='inductor' kwargs={}
[2024-07-10 16:18:04,385] [INFO] [config.py:1000:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   curriculum_enabled_legacy .... False
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   curriculum_params_legacy ..... False
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   data_efficiency_enabled ...... False
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   dataloader_drop_last ......... False
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   disable_allgather ............ False
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   dump_state ................... False
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   dynamic_loss_scale_args ...... None
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   eigenvalue_enabled ........... False
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   eigenvalue_gas_boundary_resolution  1
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   eigenvalue_layer_num ......... 0
[2024-07-10 16:18:04,386] [INFO] [config.py:1000:print]   eigenvalue_max_iter .......... 100
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   eigenvalue_stability ......... 1e-06
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   eigenvalue_tol ............... 0.01
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   eigenvalue_verbose ........... False
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   elasticity_enabled ........... False
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   flops_profiler_config ........ {
    "enabled": false, 
    "recompute_fwd_factor": 0.0, 
    "profile_step": 1, 
    "module_depth": -1, 
    "top_modules": 1, 
    "detailed": true, 
    "output_file": null
}
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   fp16_auto_cast ............... None
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   fp16_enabled ................. False
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   fp16_master_weights_and_gradients  False
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   global_rank .................. 0
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   grad_accum_dtype ............. None
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   gradient_accumulation_steps .. 1
[2024-07-10 16:18:04,387] [INFO] [config.py:1000:print]   gradient_clipping ............ 10.0
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   gradient_predivide_factor .... 1.0
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   graph_harvesting ............. False
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   initial_dynamic_scale ........ 1
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   load_universal_checkpoint .... False
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   loss_scale ................... 1.0
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   memory_breakdown ............. False
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   mics_hierarchial_params_gather  False
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   mics_shard_size .............. -1
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') enabled=False
[2024-07-10 16:18:04,388] [INFO] [config.py:1000:print]   nebula_config ................ {
    "enabled": false, 
    "persistent_storage_path": null, 
    "persistent_time_interval": 100, 
    "num_of_version_in_retention": 2, 
    "enable_nebula_load": true, 
    "load_path": null
}
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   optimizer_legacy_fusion ...... False
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   optimizer_name ............... None
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   optimizer_params ............. None
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   pld_enabled .................. False
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   pld_params ................... False
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   prescale_gradients ........... False
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   scheduler_name ............... None
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   scheduler_params ............. None
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   seq_parallel_communication_data_type  torch.float32
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   sparse_attention ............. None
[2024-07-10 16:18:04,389] [INFO] [config.py:1000:print]   sparse_gradients_enabled ..... False
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   steps_per_print .............. inf
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   train_batch_size ............. 128
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   train_micro_batch_size_per_gpu  32
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   use_data_before_expert_parallel_  False
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   use_node_local_storage ....... False
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   wall_clock_breakdown ......... False
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   weight_quantization_config ... None
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   world_size ................... 4
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   zero_allow_untested_optimizer  True
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   zero_config .................. stage=2 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500,000,000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500,000,000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=DeepSpeedZeroOffloadParamConfig(device='cpu', nvme_path=None, buffer_count=5, buffer_size=100,000,000, max_in_cpu=1,000,000,000, pin_memory=False) offload_optimizer=DeepSpeedZeroOffloadOptimizerConfig(device='cpu', nvme_path=None, buffer_count=4, pin_memory=False, pipeline=False, pipeline_read=False, pipeline_write=False, fast_init=False, ratio=1.0) sub_group_size=1,000,000,000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50,000,000 param_persistence_threshold=100,000 model_persistence_threshold=sys.maxsize max_live_parameters=1,000,000,000 max_reuse_distance=1,000,000,000 gather_16bit_weights_on_model_save=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   zero_enabled ................. True
[2024-07-10 16:18:04,390] [INFO] [config.py:1000:print]   zero_force_ds_cpu_optimizer .. True
[2024-07-10 16:18:04,391] [INFO] [config.py:1000:print]   zero_optimization_stage ...... 2
[2024-07-10 16:18:04,391] [INFO] [config.py:986:print_user_config]   json = {
    "train_batch_size": 128, 
    "train_micro_batch_size_per_gpu": 32, 
    "gradient_accumulation_steps": 1, 
    "zero_optimization": {
        "stage": 2, 
        "offload_optimizer": {
            "device": "cpu", 
            "nvme_path": null
        }, 
        "offload_param": {
            "device": "cpu", 
            "nvme_path": null
        }, 
        "stage3_gather_16bit_weights_on_model_save": false
    }, 
    "gradient_clipping": 10.0, 
    "steps_per_print": inf, 
    "bf16": {
        "enabled": true
    }, 
    "fp16": {
        "enabled": false
    }, 
    "zero_allow_untested_optimizer": true
}
Loaded 1 eval batches of size 64
Using AdamW optimizer
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 371, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 296, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 339, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 423, 2304])
size_out=torch.Size([128, 423, 2304])
eval after 0: {'rewards_eval/chosen': '-0.056337', 'rewards_eval/rejected': '-0.035957', 'rewards_eval/accuracies': '0.41016', 'rewards_eval/margins': '-0.02038', 'logps_eval/rejected': '-346.03', 'logps_eval/chosen': '-355.51', 'loss/eval': '0.70663'}
==================================================
==================================================
Running evaluation after 0 train examples
==================================================
==================================================
train stats after 32 examples: {'rewards_train/chosen': '-0.030011', 'rewards_train/rejected': '-0.019089', 'rewards_train/accuracies': '0.46094', 'rewards_train/margins': '-0.010922', 'logps_train/rejected': '-284.89', 'logps_train/chosen': '-319.18', 'loss/train': '0.70236', 'examples_per_second': '52.133', 'grad_norm': '0', 'counters/examples': 32, 'counters/updates': 1}
train stats after 64 examples: {'rewards_train/chosen': '-0.051801', 'rewards_train/rejected': '-0.048854', 'rewards_train/accuracies': '0.44531', 'rewards_train/margins': '-0.0029465', 'logps_train/rejected': '-309.57', 'logps_train/chosen': '-328.2', 'loss/train': '0.69782', 'examples_per_second': '57.079', 'grad_norm': '0', 'counters/examples': 64, 'counters/updates': 2}
train stats after 96 examples: {'rewards_train/chosen': '-0.03548', 'rewards_train/rejected': '-0.028475', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '-0.0070046', 'logps_train/rejected': '-303.33', 'logps_train/chosen': '-328.62', 'loss/train': '0.6993', 'examples_per_second': '57.097', 'grad_norm': '0', 'counters/examples': 96, 'counters/updates': 3}
train stats after 128 examples: {'rewards_train/chosen': '-0.0056819', 'rewards_train/rejected': '-0.0019002', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '-0.0037816', 'logps_train/rejected': '-300.86', 'logps_train/chosen': '-308.35', 'loss/train': '0.69747', 'examples_per_second': '57.781', 'grad_norm': '0', 'counters/examples': 128, 'counters/updates': 4}
train stats after 160 examples: {'rewards_train/chosen': '-0.025781', 'rewards_train/rejected': '-0.026142', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.00036083', 'logps_train/rejected': '-283.88', 'logps_train/chosen': '-308.45', 'loss/train': '0.69818', 'examples_per_second': '63.325', 'grad_norm': '0', 'counters/examples': 160, 'counters/updates': 5}
train stats after 192 examples: {'rewards_train/chosen': '0.022983', 'rewards_train/rejected': '-0.017363', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.040347', 'logps_train/rejected': '-295.9', 'logps_train/chosen': '-310.73', 'loss/train': '0.68379', 'examples_per_second': '49.275', 'grad_norm': '0', 'counters/examples': 192, 'counters/updates': 6}
train stats after 224 examples: {'rewards_train/chosen': '-0.20084', 'rewards_train/rejected': '-0.19296', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '-0.0078827', 'logps_train/rejected': '-291.87', 'logps_train/chosen': '-323.75', 'loss/train': '0.73358', 'examples_per_second': '67.247', 'grad_norm': '0', 'counters/examples': 224, 'counters/updates': 7}
train stats after 256 examples: {'rewards_train/chosen': '-0.067317', 'rewards_train/rejected': '-0.053631', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '-0.013686', 'logps_train/rejected': '-275.2', 'logps_train/chosen': '-305.07', 'loss/train': '0.70431', 'examples_per_second': '64.516', 'grad_norm': '0', 'counters/examples': 256, 'counters/updates': 8}
train stats after 288 examples: {'rewards_train/chosen': '-0.0073763', 'rewards_train/rejected': '-0.042224', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.034848', 'logps_train/rejected': '-300.71', 'logps_train/chosen': '-302.4', 'loss/train': '0.67914', 'examples_per_second': '65.15', 'grad_norm': '0', 'counters/examples': 288, 'counters/updates': 9}
train stats after 320 examples: {'rewards_train/chosen': '-0.16267', 'rewards_train/rejected': '-0.1499', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.012776', 'logps_train/rejected': '-271.29', 'logps_train/chosen': '-313.76', 'loss/train': '0.72171', 'examples_per_second': '51.886', 'grad_norm': '0', 'counters/examples': 320, 'counters/updates': 10}
train stats after 352 examples: {'rewards_train/chosen': '-0.048031', 'rewards_train/rejected': '-0.024856', 'rewards_train/accuracies': '0.44531', 'rewards_train/margins': '-0.023175', 'logps_train/rejected': '-296.58', 'logps_train/chosen': '-347.85', 'loss/train': '0.71053', 'examples_per_second': '47.22', 'grad_norm': '0', 'counters/examples': 352, 'counters/updates': 11}
train stats after 384 examples: {'rewards_train/chosen': '-0.014899', 'rewards_train/rejected': '-0.034762', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.019863', 'logps_train/rejected': '-304.1', 'logps_train/chosen': '-305.89', 'loss/train': '0.68633', 'examples_per_second': '54.583', 'grad_norm': '0', 'counters/examples': 384, 'counters/updates': 12}
train stats after 416 examples: {'rewards_train/chosen': '-0.022019', 'rewards_train/rejected': '-0.02963', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.0076106', 'logps_train/rejected': '-328.54', 'logps_train/chosen': '-327.04', 'loss/train': '0.6927', 'examples_per_second': '55.457', 'grad_norm': '0', 'counters/examples': 416, 'counters/updates': 13}
train stats after 448 examples: {'rewards_train/chosen': '-0.019516', 'rewards_train/rejected': '-0.026067', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.006551', 'logps_train/rejected': '-282.16', 'logps_train/chosen': '-321.86', 'loss/train': '0.69348', 'examples_per_second': '53.593', 'grad_norm': '0', 'counters/examples': 448, 'counters/updates': 14}
train stats after 480 examples: {'rewards_train/chosen': '-0.016447', 'rewards_train/rejected': '-0.014245', 'rewards_train/accuracies': '0.44531', 'rewards_train/margins': '-0.0022027', 'logps_train/rejected': '-323.11', 'logps_train/chosen': '-338.04', 'loss/train': '0.69804', 'examples_per_second': '55.823', 'grad_norm': '0', 'counters/examples': 480, 'counters/updates': 15}
train stats after 512 examples: {'rewards_train/chosen': '-0.034779', 'rewards_train/rejected': '-0.019205', 'rewards_train/accuracies': '0.44531', 'rewards_train/margins': '-0.015574', 'logps_train/rejected': '-276.09', 'logps_train/chosen': '-311.15', 'loss/train': '0.70484', 'examples_per_second': '47.324', 'grad_norm': '0', 'counters/examples': 512, 'counters/updates': 16}
train stats after 544 examples: {'rewards_train/chosen': '-0.0022203', 'rewards_train/rejected': '-0.018724', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.016504', 'logps_train/rejected': '-279.39', 'logps_train/chosen': '-304.65', 'loss/train': '0.68973', 'examples_per_second': '54.582', 'grad_norm': '0', 'counters/examples': 544, 'counters/updates': 17}
train stats after 576 examples: {'rewards_train/chosen': '0.0011811', 'rewards_train/rejected': '-0.070172', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.071353', 'logps_train/rejected': '-305.39', 'logps_train/chosen': '-332.69', 'loss/train': '0.67913', 'examples_per_second': '45.745', 'grad_norm': '0', 'counters/examples': 576, 'counters/updates': 18}
train stats after 608 examples: {'rewards_train/chosen': '-0.51505', 'rewards_train/rejected': '-0.46307', 'rewards_train/accuracies': '0.46094', 'rewards_train/margins': '-0.051982', 'logps_train/rejected': '-299.13', 'logps_train/chosen': '-309.8', 'loss/train': '0.83077', 'examples_per_second': '60.993', 'grad_norm': '0', 'counters/examples': 608, 'counters/updates': 19}
train stats after 640 examples: {'rewards_train/chosen': '0.037565', 'rewards_train/rejected': '-0.028661', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.066226', 'logps_train/rejected': '-261.91', 'logps_train/chosen': '-322.28', 'loss/train': '0.67637', 'examples_per_second': '52.492', 'grad_norm': '0', 'counters/examples': 640, 'counters/updates': 20}
train stats after 672 examples: {'rewards_train/chosen': '-0.022634', 'rewards_train/rejected': '0.0063657', 'rewards_train/accuracies': '0.39844', 'rewards_train/margins': '-0.028999', 'logps_train/rejected': '-283.67', 'logps_train/chosen': '-300.3', 'loss/train': '0.71513', 'examples_per_second': '53.581', 'grad_norm': '0', 'counters/examples': 672, 'counters/updates': 21}
train stats after 704 examples: {'rewards_train/chosen': '-0.026417', 'rewards_train/rejected': '-0.031371', 'rewards_train/accuracies': '0.46094', 'rewards_train/margins': '0.0049539', 'logps_train/rejected': '-270.83', 'logps_train/chosen': '-298.19', 'loss/train': '0.6935', 'examples_per_second': '67.242', 'grad_norm': '0', 'counters/examples': 704, 'counters/updates': 22}
train stats after 736 examples: {'rewards_train/chosen': '-0.055536', 'rewards_train/rejected': '-0.033631', 'rewards_train/accuracies': '0.40625', 'rewards_train/margins': '-0.021906', 'logps_train/rejected': '-294.17', 'logps_train/chosen': '-310.43', 'loss/train': '0.70726', 'examples_per_second': '53.273', 'grad_norm': '0', 'counters/examples': 736, 'counters/updates': 23}
train stats after 768 examples: {'rewards_train/chosen': '-0.026213', 'rewards_train/rejected': '-0.031771', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.0055584', 'logps_train/rejected': '-315.86', 'logps_train/chosen': '-315.47', 'loss/train': '0.69409', 'examples_per_second': '44.963', 'grad_norm': '0', 'counters/examples': 768, 'counters/updates': 24}
train stats after 800 examples: {'rewards_train/chosen': '-0.029973', 'rewards_train/rejected': '-0.018528', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '-0.011445', 'logps_train/rejected': '-297.8', 'logps_train/chosen': '-322.03', 'loss/train': '0.70201', 'examples_per_second': '61.371', 'grad_norm': '0', 'counters/examples': 800, 'counters/updates': 25}
train stats after 832 examples: {'rewards_train/chosen': '-0.024246', 'rewards_train/rejected': '-0.012941', 'rewards_train/accuracies': '0.4375', 'rewards_train/margins': '-0.011305', 'logps_train/rejected': '-289.91', 'logps_train/chosen': '-300.31', 'loss/train': '0.70254', 'examples_per_second': '67.294', 'grad_norm': '0', 'counters/examples': 832, 'counters/updates': 26}
train stats after 864 examples: {'rewards_train/chosen': '-0.043229', 'rewards_train/rejected': '-0.034179', 'rewards_train/accuracies': '0.46094', 'rewards_train/margins': '-0.0090505', 'logps_train/rejected': '-295.95', 'logps_train/chosen': '-303.97', 'loss/train': '0.70264', 'examples_per_second': '70.197', 'grad_norm': '0', 'counters/examples': 864, 'counters/updates': 27}
train stats after 896 examples: {'rewards_train/chosen': '-0.10239', 'rewards_train/rejected': '-0.12872', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.026329', 'logps_train/rejected': '-283.05', 'logps_train/chosen': '-321.55', 'loss/train': '0.70103', 'examples_per_second': '44.723', 'grad_norm': '0', 'counters/examples': 896, 'counters/updates': 28}
train stats after 928 examples: {'rewards_train/chosen': '-0.046513', 'rewards_train/rejected': '-0.065382', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.018869', 'logps_train/rejected': '-327.73', 'logps_train/chosen': '-340.22', 'loss/train': '0.69501', 'examples_per_second': '55.323', 'grad_norm': '0', 'counters/examples': 928, 'counters/updates': 29}
train stats after 960 examples: {'rewards_train/chosen': '-0.027051', 'rewards_train/rejected': '-0.015146', 'rewards_train/accuracies': '0.42188', 'rewards_train/margins': '-0.011905', 'logps_train/rejected': '-297.35', 'logps_train/chosen': '-315.3', 'loss/train': '0.70412', 'examples_per_second': '53.523', 'grad_norm': '0', 'counters/examples': 960, 'counters/updates': 30}
train stats after 992 examples: {'rewards_train/chosen': '-0.020485', 'rewards_train/rejected': '-0.026736', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.0062505', 'logps_train/rejected': '-327.49', 'logps_train/chosen': '-329.79', 'loss/train': '0.69339', 'examples_per_second': '49.502', 'grad_norm': '0', 'counters/examples': 992, 'counters/updates': 31}
train stats after 1024 examples: {'rewards_train/chosen': '-0.017968', 'rewards_train/rejected': '-0.048056', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.030087', 'logps_train/rejected': '-273.49', 'logps_train/chosen': '-287.25', 'loss/train': '0.68148', 'examples_per_second': '49.748', 'grad_norm': '0', 'counters/examples': 1024, 'counters/updates': 32}
train stats after 1056 examples: {'rewards_train/chosen': '-0.040353', 'rewards_train/rejected': '-0.036022', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.0043312', 'logps_train/rejected': '-300.19', 'logps_train/chosen': '-314.01', 'loss/train': '0.6989', 'examples_per_second': '62.772', 'grad_norm': '0', 'counters/examples': 1056, 'counters/updates': 33}
train stats after 1088 examples: {'rewards_train/chosen': '-0.032549', 'rewards_train/rejected': '-0.018494', 'rewards_train/accuracies': '0.42969', 'rewards_train/margins': '-0.014056', 'logps_train/rejected': '-304.28', 'logps_train/chosen': '-309.35', 'loss/train': '0.7034', 'examples_per_second': '70.619', 'grad_norm': '0', 'counters/examples': 1088, 'counters/updates': 34}
train stats after 1120 examples: {'rewards_train/chosen': '-0.058282', 'rewards_train/rejected': '-0.025891', 'rewards_train/accuracies': '0.42188', 'rewards_train/margins': '-0.032391', 'logps_train/rejected': '-287.92', 'logps_train/chosen': '-309.27', 'loss/train': '0.71285', 'examples_per_second': '47.883', 'grad_norm': '0', 'counters/examples': 1120, 'counters/updates': 35}
train stats after 1152 examples: {'rewards_train/chosen': '-0.037852', 'rewards_train/rejected': '-0.035751', 'rewards_train/accuracies': '0.47656', 'rewards_train/margins': '-0.0021008', 'logps_train/rejected': '-275.05', 'logps_train/chosen': '-292.99', 'loss/train': '0.6974', 'examples_per_second': '60.364', 'grad_norm': '0', 'counters/examples': 1152, 'counters/updates': 36}
train stats after 1184 examples: {'rewards_train/chosen': '-0.017735', 'rewards_train/rejected': '-0.024404', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.0066695', 'logps_train/rejected': '-270.35', 'logps_train/chosen': '-301.73', 'loss/train': '0.69298', 'examples_per_second': '50.471', 'grad_norm': '0', 'counters/examples': 1184, 'counters/updates': 37}
train stats after 1216 examples: {'rewards_train/chosen': '-0.020537', 'rewards_train/rejected': '-0.0083074', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '-0.01223', 'logps_train/rejected': '-285.54', 'logps_train/chosen': '-313.05', 'loss/train': '0.70767', 'examples_per_second': '51.475', 'grad_norm': '0', 'counters/examples': 1216, 'counters/updates': 38}
train stats after 1248 examples: {'rewards_train/chosen': '-0.039308', 'rewards_train/rejected': '-0.024652', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.014656', 'logps_train/rejected': '-321.18', 'logps_train/chosen': '-329.69', 'loss/train': '0.70588', 'examples_per_second': '66.06', 'grad_norm': '0', 'counters/examples': 1248, 'counters/updates': 39}
train stats after 1280 examples: {'rewards_train/chosen': '-0.0198', 'rewards_train/rejected': '-0.028286', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.008486', 'logps_train/rejected': '-296.95', 'logps_train/chosen': '-316.37', 'loss/train': '0.69113', 'examples_per_second': '65.931', 'grad_norm': '0', 'counters/examples': 1280, 'counters/updates': 40}
train stats after 1312 examples: {'rewards_train/chosen': '-0.027085', 'rewards_train/rejected': '-0.045688', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.018603', 'logps_train/rejected': '-288.1', 'logps_train/chosen': '-316.55', 'loss/train': '0.68943', 'examples_per_second': '50.567', 'grad_norm': '0', 'counters/examples': 1312, 'counters/updates': 41}
train stats after 1344 examples: {'rewards_train/chosen': '-0.043076', 'rewards_train/rejected': '-0.036463', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.0066134', 'logps_train/rejected': '-300.52', 'logps_train/chosen': '-336.62', 'loss/train': '0.70083', 'examples_per_second': '48.283', 'grad_norm': '0', 'counters/examples': 1344, 'counters/updates': 42}
train stats after 1376 examples: {'rewards_train/chosen': '-0.055729', 'rewards_train/rejected': '-0.063172', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.0074426', 'logps_train/rejected': '-306.08', 'logps_train/chosen': '-315.71', 'loss/train': '0.69385', 'examples_per_second': '46.279', 'grad_norm': '0', 'counters/examples': 1376, 'counters/updates': 43}
train stats after 1408 examples: {'rewards_train/chosen': '-0.025873', 'rewards_train/rejected': '-0.03767', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.011798', 'logps_train/rejected': '-310.06', 'logps_train/chosen': '-332.34', 'loss/train': '0.69132', 'examples_per_second': '31.718', 'grad_norm': '0', 'counters/examples': 1408, 'counters/updates': 44}
train stats after 1440 examples: {'rewards_train/chosen': '-0.043739', 'rewards_train/rejected': '-0.034727', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '-0.0090121', 'logps_train/rejected': '-286.67', 'logps_train/chosen': '-315.48', 'loss/train': '0.70098', 'examples_per_second': '50.932', 'grad_norm': '0', 'counters/examples': 1440, 'counters/updates': 45}
train stats after 1472 examples: {'rewards_train/chosen': '-0.023861', 'rewards_train/rejected': '-0.031414', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.007553', 'logps_train/rejected': '-265.01', 'logps_train/chosen': '-305.16', 'loss/train': '0.69217', 'examples_per_second': '61.82', 'grad_norm': '0', 'counters/examples': 1472, 'counters/updates': 46}
train stats after 1504 examples: {'rewards_train/chosen': '-0.045206', 'rewards_train/rejected': '-0.045045', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.00016076', 'logps_train/rejected': '-290.34', 'logps_train/chosen': '-305.68', 'loss/train': '0.69803', 'examples_per_second': '46.107', 'grad_norm': '0', 'counters/examples': 1504, 'counters/updates': 47}
train stats after 1536 examples: {'rewards_train/chosen': '-0.037041', 'rewards_train/rejected': '-0.031673', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '-0.0053678', 'logps_train/rejected': '-270.86', 'logps_train/chosen': '-317.6', 'loss/train': '0.69891', 'examples_per_second': '54.646', 'grad_norm': '0', 'counters/examples': 1536, 'counters/updates': 48}
train stats after 1568 examples: {'rewards_train/chosen': '-0.039689', 'rewards_train/rejected': '-0.033437', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '-0.0062524', 'logps_train/rejected': '-285.2', 'logps_train/chosen': '-311.14', 'loss/train': '0.69993', 'examples_per_second': '68.712', 'grad_norm': '0', 'counters/examples': 1568, 'counters/updates': 49}
train stats after 1600 examples: {'rewards_train/chosen': '-0.036236', 'rewards_train/rejected': '-0.019684', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '-0.016552', 'logps_train/rejected': '-308.19', 'logps_train/chosen': '-329.74', 'loss/train': '0.7058', 'examples_per_second': '59.216', 'grad_norm': '0', 'counters/examples': 1600, 'counters/updates': 50}
train stats after 1632 examples: {'rewards_train/chosen': '-0.043377', 'rewards_train/rejected': '-0.025888', 'rewards_train/accuracies': '0.44531', 'rewards_train/margins': '-0.017488', 'logps_train/rejected': '-288.12', 'logps_train/chosen': '-306.65', 'loss/train': '0.70553', 'examples_per_second': '49.968', 'grad_norm': '0', 'counters/examples': 1632, 'counters/updates': 51}
train stats after 1664 examples: {'rewards_train/chosen': '-0.026299', 'rewards_train/rejected': '-0.013289', 'rewards_train/accuracies': '0.41406', 'rewards_train/margins': '-0.01301', 'logps_train/rejected': '-290.76', 'logps_train/chosen': '-301.96', 'loss/train': '0.70257', 'examples_per_second': '65.544', 'grad_norm': '0', 'counters/examples': 1664, 'counters/updates': 52}
train stats after 1696 examples: {'rewards_train/chosen': '-0.022658', 'rewards_train/rejected': '-0.020967', 'rewards_train/accuracies': '0.45312', 'rewards_train/margins': '-0.0016915', 'logps_train/rejected': '-288.2', 'logps_train/chosen': '-289.77', 'loss/train': '0.69712', 'examples_per_second': '73.828', 'grad_norm': '0', 'counters/examples': 1696, 'counters/updates': 53}
train stats after 1728 examples: {'rewards_train/chosen': '-0.037028', 'rewards_train/rejected': '-0.016521', 'rewards_train/accuracies': '0.42188', 'rewards_train/margins': '-0.020507', 'logps_train/rejected': '-321.97', 'logps_train/chosen': '-329', 'loss/train': '0.70622', 'examples_per_second': '50.925', 'grad_norm': '0', 'counters/examples': 1728, 'counters/updates': 54}
train stats after 1760 examples: {'rewards_train/chosen': '-0.017886', 'rewards_train/rejected': '-0.028981', 'rewards_train/accuracies': '0.45312', 'rewards_train/margins': '0.011095', 'logps_train/rejected': '-299.6', 'logps_train/chosen': '-322.68', 'loss/train': '0.69228', 'examples_per_second': '64.82', 'grad_norm': '0', 'counters/examples': 1760, 'counters/updates': 55}
train stats after 1792 examples: {'rewards_train/chosen': '-0.018613', 'rewards_train/rejected': '-0.03628', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.017667', 'logps_train/rejected': '-286.41', 'logps_train/chosen': '-303.32', 'loss/train': '0.68729', 'examples_per_second': '50.955', 'grad_norm': '0', 'counters/examples': 1792, 'counters/updates': 56}
train stats after 1824 examples: {'rewards_train/chosen': '-0.03564', 'rewards_train/rejected': '-0.035235', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '-0.00040526', 'logps_train/rejected': '-315.65', 'logps_train/chosen': '-347.3', 'loss/train': '0.69706', 'examples_per_second': '47.141', 'grad_norm': '0', 'counters/examples': 1824, 'counters/updates': 57}
train stats after 1856 examples: {'rewards_train/chosen': '-0.037271', 'rewards_train/rejected': '-0.027704', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.0095671', 'logps_train/rejected': '-287.55', 'logps_train/chosen': '-301.64', 'loss/train': '0.7016', 'examples_per_second': '71.438', 'grad_norm': '0', 'counters/examples': 1856, 'counters/updates': 58}
train stats after 1888 examples: {'rewards_train/chosen': '-0.014138', 'rewards_train/rejected': '-0.017733', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.0035954', 'logps_train/rejected': '-261.87', 'logps_train/chosen': '-289.24', 'loss/train': '0.69406', 'examples_per_second': '65.007', 'grad_norm': '0', 'counters/examples': 1888, 'counters/updates': 59}
train stats after 1920 examples: {'rewards_train/chosen': '-0.03111', 'rewards_train/rejected': '-0.025352', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '-0.0057589', 'logps_train/rejected': '-288.65', 'logps_train/chosen': '-308.91', 'loss/train': '0.69957', 'examples_per_second': '67.885', 'grad_norm': '0', 'counters/examples': 1920, 'counters/updates': 60}
train stats after 1952 examples: {'rewards_train/chosen': '-0.019839', 'rewards_train/rejected': '-0.05138', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.03154', 'logps_train/rejected': '-307.82', 'logps_train/chosen': '-335.72', 'loss/train': '0.68109', 'examples_per_second': '58.284', 'grad_norm': '0', 'counters/examples': 1952, 'counters/updates': 61}
train stats after 1984 examples: {'rewards_train/chosen': '-0.032222', 'rewards_train/rejected': '-0.042827', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.010605', 'logps_train/rejected': '-293.87', 'logps_train/chosen': '-315.78', 'loss/train': '0.69174', 'examples_per_second': '51.819', 'grad_norm': '0', 'counters/examples': 1984, 'counters/updates': 62}
train stats after 2016 examples: {'rewards_train/chosen': '-0.018309', 'rewards_train/rejected': '-0.030465', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.012156', 'logps_train/rejected': '-300.23', 'logps_train/chosen': '-320.63', 'loss/train': '0.69071', 'examples_per_second': '67.659', 'grad_norm': '0', 'counters/examples': 2016, 'counters/updates': 63}
train stats after 2048 examples: {'rewards_train/chosen': '-0.026476', 'rewards_train/rejected': '-0.0034268', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '-0.023049', 'logps_train/rejected': '-308.67', 'logps_train/chosen': '-338.59', 'loss/train': '0.70898', 'examples_per_second': '63.017', 'grad_norm': '0', 'counters/examples': 2048, 'counters/updates': 64}
train stats after 2080 examples: {'rewards_train/chosen': '-0.0057422', 'rewards_train/rejected': '-0.02542', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.019677', 'logps_train/rejected': '-302.38', 'logps_train/chosen': '-326.59', 'loss/train': '0.68687', 'examples_per_second': '54.314', 'grad_norm': '0', 'counters/examples': 2080, 'counters/updates': 65}
train stats after 2112 examples: {'rewards_train/chosen': '-0.031599', 'rewards_train/rejected': '0.012139', 'rewards_train/accuracies': '0.35938', 'rewards_train/margins': '-0.043737', 'logps_train/rejected': '-308.41', 'logps_train/chosen': '-335.26', 'loss/train': '0.71835', 'examples_per_second': '46.532', 'grad_norm': '0', 'counters/examples': 2112, 'counters/updates': 66}
train stats after 2144 examples: {'rewards_train/chosen': '-0.024394', 'rewards_train/rejected': '-0.026981', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.0025865', 'logps_train/rejected': '-292.98', 'logps_train/chosen': '-292.84', 'loss/train': '0.69612', 'examples_per_second': '51.579', 'grad_norm': '0', 'counters/examples': 2144, 'counters/updates': 67}
train stats after 2176 examples: {'rewards_train/chosen': '-0.0052069', 'rewards_train/rejected': '-0.030232', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.025025', 'logps_train/rejected': '-294.9', 'logps_train/chosen': '-331.47', 'loss/train': '0.68412', 'examples_per_second': '58.173', 'grad_norm': '0', 'counters/examples': 2176, 'counters/updates': 68}
train stats after 2208 examples: {'rewards_train/chosen': '-0.013371', 'rewards_train/rejected': '-0.010046', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '-0.0033253', 'logps_train/rejected': '-265.12', 'logps_train/chosen': '-292.81', 'loss/train': '0.69797', 'examples_per_second': '51.711', 'grad_norm': '0', 'counters/examples': 2208, 'counters/updates': 69}
train stats after 2240 examples: {'rewards_train/chosen': '-0.012919', 'rewards_train/rejected': '-0.024939', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.01202', 'logps_train/rejected': '-269.71', 'logps_train/chosen': '-326.24', 'loss/train': '0.69149', 'examples_per_second': '51.346', 'grad_norm': '0', 'counters/examples': 2240, 'counters/updates': 70}
train stats after 2272 examples: {'rewards_train/chosen': '-0.017182', 'rewards_train/rejected': '-0.027776', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.010594', 'logps_train/rejected': '-290.89', 'logps_train/chosen': '-321.84', 'loss/train': '0.69125', 'examples_per_second': '59.32', 'grad_norm': '0', 'counters/examples': 2272, 'counters/updates': 71}
train stats after 2304 examples: {'rewards_train/chosen': '-0.031014', 'rewards_train/rejected': '-0.021174', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.0098401', 'logps_train/rejected': '-311.17', 'logps_train/chosen': '-320.6', 'loss/train': '0.70193', 'examples_per_second': '69.822', 'grad_norm': '0', 'counters/examples': 2304, 'counters/updates': 72}
train stats after 2336 examples: {'rewards_train/chosen': '-0.014062', 'rewards_train/rejected': '-0.022386', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.0083236', 'logps_train/rejected': '-254.28', 'logps_train/chosen': '-296.51', 'loss/train': '0.69391', 'examples_per_second': '66.72', 'grad_norm': '0', 'counters/examples': 2336, 'counters/updates': 73}
train stats after 2368 examples: {'rewards_train/chosen': '-0.0108', 'rewards_train/rejected': '-0.034256', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.023456', 'logps_train/rejected': '-327.08', 'logps_train/chosen': '-325.28', 'loss/train': '0.68512', 'examples_per_second': '45.689', 'grad_norm': '0', 'counters/examples': 2368, 'counters/updates': 74}
train stats after 2400 examples: {'rewards_train/chosen': '-0.032684', 'rewards_train/rejected': '-0.03673', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.0040463', 'logps_train/rejected': '-298.5', 'logps_train/chosen': '-313.37', 'loss/train': '0.69545', 'examples_per_second': '67.649', 'grad_norm': '0', 'counters/examples': 2400, 'counters/updates': 75}
train stats after 2432 examples: {'rewards_train/chosen': '-0.011289', 'rewards_train/rejected': '-0.024033', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.012744', 'logps_train/rejected': '-284.84', 'logps_train/chosen': '-314.34', 'loss/train': '0.69046', 'examples_per_second': '58.832', 'grad_norm': '0', 'counters/examples': 2432, 'counters/updates': 76}
train stats after 2464 examples: {'rewards_train/chosen': '-0.013812', 'rewards_train/rejected': '-0.044028', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.030215', 'logps_train/rejected': '-315.09', 'logps_train/chosen': '-333.45', 'loss/train': '0.68411', 'examples_per_second': '50.617', 'grad_norm': '0', 'counters/examples': 2464, 'counters/updates': 77}
train stats after 2496 examples: {'rewards_train/chosen': '0.0066112', 'rewards_train/rejected': '-0.023715', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.030326', 'logps_train/rejected': '-274.36', 'logps_train/chosen': '-300.1', 'loss/train': '0.68174', 'examples_per_second': '57.325', 'grad_norm': '0', 'counters/examples': 2496, 'counters/updates': 78}
train stats after 2528 examples: {'rewards_train/chosen': '0.00060454', 'rewards_train/rejected': '-0.048061', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.048666', 'logps_train/rejected': '-305.02', 'logps_train/chosen': '-319.64', 'loss/train': '0.67364', 'examples_per_second': '57.005', 'grad_norm': '0', 'counters/examples': 2528, 'counters/updates': 79}
train stats after 2560 examples: {'rewards_train/chosen': '-0.013223', 'rewards_train/rejected': '-0.029132', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.015909', 'logps_train/rejected': '-293.77', 'logps_train/chosen': '-327.01', 'loss/train': '0.68992', 'examples_per_second': '58.444', 'grad_norm': '0', 'counters/examples': 2560, 'counters/updates': 80}
train stats after 2592 examples: {'rewards_train/chosen': '-0.01862', 'rewards_train/rejected': '-0.048226', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.029606', 'logps_train/rejected': '-297.11', 'logps_train/chosen': '-294.5', 'loss/train': '0.68358', 'examples_per_second': '51.473', 'grad_norm': '0', 'counters/examples': 2592, 'counters/updates': 81}
train stats after 2624 examples: {'rewards_train/chosen': '-0.0036149', 'rewards_train/rejected': '0.0023709', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.0059858', 'logps_train/rejected': '-288.09', 'logps_train/chosen': '-304.79', 'loss/train': '0.69998', 'examples_per_second': '67.905', 'grad_norm': '0', 'counters/examples': 2624, 'counters/updates': 82}
train stats after 2656 examples: {'rewards_train/chosen': '-0.022965', 'rewards_train/rejected': '-0.01722', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '-0.0057443', 'logps_train/rejected': '-329.89', 'logps_train/chosen': '-328.92', 'loss/train': '0.70293', 'examples_per_second': '46.986', 'grad_norm': '0', 'counters/examples': 2656, 'counters/updates': 83}
train stats after 2688 examples: {'rewards_train/chosen': '-0.0034549', 'rewards_train/rejected': '-0.0092575', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.0058027', 'logps_train/rejected': '-292.4', 'logps_train/chosen': '-318.42', 'loss/train': '0.69632', 'examples_per_second': '47.655', 'grad_norm': '0', 'counters/examples': 2688, 'counters/updates': 84}
train stats after 2720 examples: {'rewards_train/chosen': '-0.040223', 'rewards_train/rejected': '-0.020061', 'rewards_train/accuracies': '0.4375', 'rewards_train/margins': '-0.020162', 'logps_train/rejected': '-270.81', 'logps_train/chosen': '-277.57', 'loss/train': '0.70795', 'examples_per_second': '72.022', 'grad_norm': '0', 'counters/examples': 2720, 'counters/updates': 85}
train stats after 2752 examples: {'rewards_train/chosen': '0.0012197', 'rewards_train/rejected': '-0.0066957', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.0079154', 'logps_train/rejected': '-308.33', 'logps_train/chosen': '-337.1', 'loss/train': '0.69518', 'examples_per_second': '61.521', 'grad_norm': '0', 'counters/examples': 2752, 'counters/updates': 86}
train stats after 2784 examples: {'rewards_train/chosen': '0.011276', 'rewards_train/rejected': '0.0082321', 'rewards_train/accuracies': '0.47656', 'rewards_train/margins': '0.0030442', 'logps_train/rejected': '-302.75', 'logps_train/chosen': '-325.01', 'loss/train': '0.69701', 'examples_per_second': '51.415', 'grad_norm': '0', 'counters/examples': 2784, 'counters/updates': 87}
train stats after 2816 examples: {'rewards_train/chosen': '0.01152', 'rewards_train/rejected': '-5.2209e-05', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.011572', 'logps_train/rejected': '-296.5', 'logps_train/chosen': '-334.42', 'loss/train': '0.69284', 'examples_per_second': '53.596', 'grad_norm': '0', 'counters/examples': 2816, 'counters/updates': 88}
train stats after 2848 examples: {'rewards_train/chosen': '-0.01916', 'rewards_train/rejected': '-0.019796', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.00063513', 'logps_train/rejected': '-302.99', 'logps_train/chosen': '-314.32', 'loss/train': '0.6984', 'examples_per_second': '65.965', 'grad_norm': '0', 'counters/examples': 2848, 'counters/updates': 89}
train stats after 2880 examples: {'rewards_train/chosen': '-0.003773', 'rewards_train/rejected': '-0.012435', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.0086622', 'logps_train/rejected': '-289.64', 'logps_train/chosen': '-333.74', 'loss/train': '0.69427', 'examples_per_second': '55.601', 'grad_norm': '0', 'counters/examples': 2880, 'counters/updates': 90}
train stats after 2912 examples: {'rewards_train/chosen': '-0.0072168', 'rewards_train/rejected': '-0.025792', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.018575', 'logps_train/rejected': '-289.5', 'logps_train/chosen': '-304.56', 'loss/train': '0.68985', 'examples_per_second': '50.142', 'grad_norm': '0', 'counters/examples': 2912, 'counters/updates': 91}
train stats after 2944 examples: {'rewards_train/chosen': '0.012082', 'rewards_train/rejected': '-0.008359', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.020441', 'logps_train/rejected': '-302.82', 'logps_train/chosen': '-327.25', 'loss/train': '0.68897', 'examples_per_second': '48.193', 'grad_norm': '0', 'counters/examples': 2944, 'counters/updates': 92}
train stats after 2976 examples: {'rewards_train/chosen': '0.0072166', 'rewards_train/rejected': '0.0018843', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.0053323', 'logps_train/rejected': '-290.53', 'logps_train/chosen': '-297.94', 'loss/train': '0.69692', 'examples_per_second': '62.854', 'grad_norm': '0', 'counters/examples': 2976, 'counters/updates': 93}
train stats after 3008 examples: {'rewards_train/chosen': '0.0035912', 'rewards_train/rejected': '-0.032558', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.036149', 'logps_train/rejected': '-279.95', 'logps_train/chosen': '-315.48', 'loss/train': '0.68225', 'examples_per_second': '69.878', 'grad_norm': '0', 'counters/examples': 3008, 'counters/updates': 94}
train stats after 3040 examples: {'rewards_train/chosen': '-0.018806', 'rewards_train/rejected': '-0.045474', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.026668', 'logps_train/rejected': '-292.09', 'logps_train/chosen': '-306.05', 'loss/train': '0.6882', 'examples_per_second': '53.302', 'grad_norm': '0', 'counters/examples': 3040, 'counters/updates': 95}
train stats after 3072 examples: {'rewards_train/chosen': '0.026836', 'rewards_train/rejected': '0.012198', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.014638', 'logps_train/rejected': '-310.77', 'logps_train/chosen': '-305.25', 'loss/train': '0.69188', 'examples_per_second': '52.126', 'grad_norm': '0', 'counters/examples': 3072, 'counters/updates': 96}
train stats after 3104 examples: {'rewards_train/chosen': '0.021191', 'rewards_train/rejected': '0.0033757', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.017815', 'logps_train/rejected': '-331.49', 'logps_train/chosen': '-353.45', 'loss/train': '0.69061', 'examples_per_second': '49.412', 'grad_norm': '0', 'counters/examples': 3104, 'counters/updates': 97}
train stats after 3136 examples: {'rewards_train/chosen': '0.0058031', 'rewards_train/rejected': '-0.023133', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.028936', 'logps_train/rejected': '-296.73', 'logps_train/chosen': '-318.47', 'loss/train': '0.68445', 'examples_per_second': '63.745', 'grad_norm': '0', 'counters/examples': 3136, 'counters/updates': 98}
train stats after 3168 examples: {'rewards_train/chosen': '-0.010373', 'rewards_train/rejected': '-0.016418', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.006045', 'logps_train/rejected': '-282.91', 'logps_train/chosen': '-304.92', 'loss/train': '0.69868', 'examples_per_second': '64.136', 'grad_norm': '0', 'counters/examples': 3168, 'counters/updates': 99}
train stats after 3200 examples: {'rewards_train/chosen': '-0.0097254', 'rewards_train/rejected': '-0.002771', 'rewards_train/accuracies': '0.45312', 'rewards_train/margins': '-0.0069545', 'logps_train/rejected': '-292.26', 'logps_train/chosen': '-324.04', 'loss/train': '0.70208', 'examples_per_second': '52.91', 'grad_norm': '0', 'counters/examples': 3200, 'counters/updates': 100}
train stats after 3232 examples: {'rewards_train/chosen': '-0.019971', 'rewards_train/rejected': '-0.022202', 'rewards_train/accuracies': '0.44531', 'rewards_train/margins': '0.0022308', 'logps_train/rejected': '-283.45', 'logps_train/chosen': '-313.52', 'loss/train': '0.69932', 'examples_per_second': '47.493', 'grad_norm': '0', 'counters/examples': 3232, 'counters/updates': 101}
train stats after 3264 examples: {'rewards_train/chosen': '0.017933', 'rewards_train/rejected': '-0.014265', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.032198', 'logps_train/rejected': '-275.85', 'logps_train/chosen': '-289.25', 'loss/train': '0.68649', 'examples_per_second': '51.331', 'grad_norm': '0', 'counters/examples': 3264, 'counters/updates': 102}
train stats after 3296 examples: {'rewards_train/chosen': '0.03036', 'rewards_train/rejected': '0.0088197', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.02154', 'logps_train/rejected': '-327.7', 'logps_train/chosen': '-352.1', 'loss/train': '0.68935', 'examples_per_second': '57.632', 'grad_norm': '0', 'counters/examples': 3296, 'counters/updates': 103}
train stats after 3328 examples: {'rewards_train/chosen': '0.019791', 'rewards_train/rejected': '-0.016928', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.036719', 'logps_train/rejected': '-331.9', 'logps_train/chosen': '-339.19', 'loss/train': '0.68189', 'examples_per_second': '55.614', 'grad_norm': '0', 'counters/examples': 3328, 'counters/updates': 104}
train stats after 3360 examples: {'rewards_train/chosen': '0.0030914', 'rewards_train/rejected': '-0.031076', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.034168', 'logps_train/rejected': '-267.49', 'logps_train/chosen': '-307.1', 'loss/train': '0.68238', 'examples_per_second': '46.7', 'grad_norm': '0', 'counters/examples': 3360, 'counters/updates': 105}
train stats after 3392 examples: {'rewards_train/chosen': '0.031475', 'rewards_train/rejected': '0.012427', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.019048', 'logps_train/rejected': '-303.75', 'logps_train/chosen': '-289.75', 'loss/train': '0.69175', 'examples_per_second': '48.477', 'grad_norm': '0', 'counters/examples': 3392, 'counters/updates': 106}
train stats after 3424 examples: {'rewards_train/chosen': '0.02821', 'rewards_train/rejected': '-0.0057227', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.033932', 'logps_train/rejected': '-304.94', 'logps_train/chosen': '-299.32', 'loss/train': '0.68327', 'examples_per_second': '63.953', 'grad_norm': '0', 'counters/examples': 3424, 'counters/updates': 107}
train stats after 3456 examples: {'rewards_train/chosen': '0.047114', 'rewards_train/rejected': '0.0059605', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.041153', 'logps_train/rejected': '-269.06', 'logps_train/chosen': '-307.69', 'loss/train': '0.68129', 'examples_per_second': '62.83', 'grad_norm': '0', 'counters/examples': 3456, 'counters/updates': 108}
train stats after 3488 examples: {'rewards_train/chosen': '0.026815', 'rewards_train/rejected': '-0.0073108', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.034126', 'logps_train/rejected': '-260.15', 'logps_train/chosen': '-298.05', 'loss/train': '0.68426', 'examples_per_second': '45.338', 'grad_norm': '0', 'counters/examples': 3488, 'counters/updates': 109}
train stats after 3520 examples: {'rewards_train/chosen': '-0.01897', 'rewards_train/rejected': '-0.0057937', 'rewards_train/accuracies': '0.4375', 'rewards_train/margins': '-0.013176', 'logps_train/rejected': '-300.09', 'logps_train/chosen': '-328.95', 'loss/train': '0.70988', 'examples_per_second': '64.529', 'grad_norm': '0', 'counters/examples': 3520, 'counters/updates': 110}
train stats after 3552 examples: {'rewards_train/chosen': '0.01366', 'rewards_train/rejected': '0.0087478', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.0049124', 'logps_train/rejected': '-281.84', 'logps_train/chosen': '-330.92', 'loss/train': '0.69969', 'examples_per_second': '46.128', 'grad_norm': '0', 'counters/examples': 3552, 'counters/updates': 111}
train stats after 3584 examples: {'rewards_train/chosen': '0.037495', 'rewards_train/rejected': '0.0026403', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.034855', 'logps_train/rejected': '-298.21', 'logps_train/chosen': '-307.55', 'loss/train': '0.68752', 'examples_per_second': '49.088', 'grad_norm': '0', 'counters/examples': 3584, 'counters/updates': 112}
train stats after 3616 examples: {'rewards_train/chosen': '-0.015836', 'rewards_train/rejected': '-0.021857', 'rewards_train/accuracies': '0.4375', 'rewards_train/margins': '0.0060216', 'logps_train/rejected': '-276.99', 'logps_train/chosen': '-316.62', 'loss/train': '0.70125', 'examples_per_second': '67.462', 'grad_norm': '0', 'counters/examples': 3616, 'counters/updates': 113}
train stats after 3648 examples: {'rewards_train/chosen': '0.0057562', 'rewards_train/rejected': '-0.022581', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.028338', 'logps_train/rejected': '-281.54', 'logps_train/chosen': '-290.44', 'loss/train': '0.68762', 'examples_per_second': '48.332', 'grad_norm': '0', 'counters/examples': 3648, 'counters/updates': 114}
train stats after 3680 examples: {'rewards_train/chosen': '-0.021541', 'rewards_train/rejected': '-0.045786', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.024246', 'logps_train/rejected': '-294.97', 'logps_train/chosen': '-306.59', 'loss/train': '0.69278', 'examples_per_second': '53.833', 'grad_norm': '0', 'counters/examples': 3680, 'counters/updates': 115}
train stats after 3712 examples: {'rewards_train/chosen': '0.0050301', 'rewards_train/rejected': '0.0031212', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.0019089', 'logps_train/rejected': '-266.62', 'logps_train/chosen': '-305.11', 'loss/train': '0.70096', 'examples_per_second': '65.374', 'grad_norm': '0', 'counters/examples': 3712, 'counters/updates': 116}
train stats after 3744 examples: {'rewards_train/chosen': '-0.0036482', 'rewards_train/rejected': '-0.0080281', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.0043799', 'logps_train/rejected': '-304.76', 'logps_train/chosen': '-321.07', 'loss/train': '0.70095', 'examples_per_second': '70.117', 'grad_norm': '0', 'counters/examples': 3744, 'counters/updates': 117}
train stats after 3776 examples: {'rewards_train/chosen': '0.0079619', 'rewards_train/rejected': '-0.0051969', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.013159', 'logps_train/rejected': '-286.79', 'logps_train/chosen': '-316.6', 'loss/train': '0.69849', 'examples_per_second': '45.33', 'grad_norm': '0', 'counters/examples': 3776, 'counters/updates': 118}
train stats after 3808 examples: {'rewards_train/chosen': '0.0065508', 'rewards_train/rejected': '0.012811', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.0062603', 'logps_train/rejected': '-282.94', 'logps_train/chosen': '-317.97', 'loss/train': '0.7053', 'examples_per_second': '46.338', 'grad_norm': '0', 'counters/examples': 3808, 'counters/updates': 119}
train stats after 3840 examples: {'rewards_train/chosen': '-0.0075705', 'rewards_train/rejected': '-0.0020772', 'rewards_train/accuracies': '0.46094', 'rewards_train/margins': '-0.0054933', 'logps_train/rejected': '-293.89', 'logps_train/chosen': '-322.14', 'loss/train': '0.70538', 'examples_per_second': '47.578', 'grad_norm': '0', 'counters/examples': 3840, 'counters/updates': 120}
train stats after 3872 examples: {'rewards_train/chosen': '0.020909', 'rewards_train/rejected': '0.012732', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.0081772', 'logps_train/rejected': '-307.28', 'logps_train/chosen': '-313.34', 'loss/train': '0.69828', 'examples_per_second': '50.93', 'grad_norm': '0', 'counters/examples': 3872, 'counters/updates': 121}
train stats after 3904 examples: {'rewards_train/chosen': '0.0020206', 'rewards_train/rejected': '-0.02873', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.03075', 'logps_train/rejected': '-283.56', 'logps_train/chosen': '-301.56', 'loss/train': '0.68894', 'examples_per_second': '51.005', 'grad_norm': '0', 'counters/examples': 3904, 'counters/updates': 122}
train stats after 3936 examples: {'rewards_train/chosen': '0.011393', 'rewards_train/rejected': '-0.015877', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.02727', 'logps_train/rejected': '-317.64', 'logps_train/chosen': '-322.99', 'loss/train': '0.69017', 'examples_per_second': '67.092', 'grad_norm': '0', 'counters/examples': 3936, 'counters/updates': 123}
train stats after 3968 examples: {'rewards_train/chosen': '-0.021956', 'rewards_train/rejected': '-0.061885', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.039929', 'logps_train/rejected': '-299.06', 'logps_train/chosen': '-332.96', 'loss/train': '0.68357', 'examples_per_second': '48.502', 'grad_norm': '0', 'counters/examples': 3968, 'counters/updates': 124}
train stats after 4000 examples: {'rewards_train/chosen': '0.034866', 'rewards_train/rejected': '-0.025587', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.060453', 'logps_train/rejected': '-299.07', 'logps_train/chosen': '-312.33', 'loss/train': '0.67276', 'examples_per_second': '45.008', 'grad_norm': '0', 'counters/examples': 4000, 'counters/updates': 125}
train stats after 4032 examples: {'rewards_train/chosen': '0.0045054', 'rewards_train/rejected': '-0.035712', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.040218', 'logps_train/rejected': '-318.6', 'logps_train/chosen': '-335.24', 'loss/train': '0.68571', 'examples_per_second': '57.319', 'grad_norm': '0', 'counters/examples': 4032, 'counters/updates': 126}
train stats after 4064 examples: {'rewards_train/chosen': '0.0096973', 'rewards_train/rejected': '0.013187', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '-0.0034893', 'logps_train/rejected': '-305.46', 'logps_train/chosen': '-328.38', 'loss/train': '0.70623', 'examples_per_second': '74.338', 'grad_norm': '0', 'counters/examples': 4064, 'counters/updates': 127}
train stats after 4096 examples: {'rewards_train/chosen': '-0.029901', 'rewards_train/rejected': '-0.019006', 'rewards_train/accuracies': '0.47656', 'rewards_train/margins': '-0.010894', 'logps_train/rejected': '-290.65', 'logps_train/chosen': '-313.86', 'loss/train': '0.71264', 'examples_per_second': '46.615', 'grad_norm': '0', 'counters/examples': 4096, 'counters/updates': 128}
train stats after 4128 examples: {'rewards_train/chosen': '-0.00031048', 'rewards_train/rejected': '-0.0064255', 'rewards_train/accuracies': '0.44531', 'rewards_train/margins': '0.0061151', 'logps_train/rejected': '-292.86', 'logps_train/chosen': '-318.6', 'loss/train': '0.70174', 'examples_per_second': '51.448', 'grad_norm': '0', 'counters/examples': 4128, 'counters/updates': 129}
train stats after 4160 examples: {'rewards_train/chosen': '0.038715', 'rewards_train/rejected': '-0.01603', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.054744', 'logps_train/rejected': '-302.79', 'logps_train/chosen': '-339.03', 'loss/train': '0.68927', 'examples_per_second': '45.578', 'grad_norm': '0', 'counters/examples': 4160, 'counters/updates': 130}
train stats after 4192 examples: {'rewards_train/chosen': '0.074151', 'rewards_train/rejected': '0.0023736', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.071777', 'logps_train/rejected': '-284.45', 'logps_train/chosen': '-302.55', 'loss/train': '0.67108', 'examples_per_second': '51.204', 'grad_norm': '0', 'counters/examples': 4192, 'counters/updates': 131}
train stats after 4224 examples: {'rewards_train/chosen': '0.013667', 'rewards_train/rejected': '-0.020578', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.034245', 'logps_train/rejected': '-273.22', 'logps_train/chosen': '-304.19', 'loss/train': '0.68882', 'examples_per_second': '72.195', 'grad_norm': '0', 'counters/examples': 4224, 'counters/updates': 132}
train stats after 4256 examples: {'rewards_train/chosen': '0.074454', 'rewards_train/rejected': '-0.013063', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.087517', 'logps_train/rejected': '-282.18', 'logps_train/chosen': '-312', 'loss/train': '0.66402', 'examples_per_second': '71.336', 'grad_norm': '0', 'counters/examples': 4256, 'counters/updates': 133}
train stats after 4288 examples: {'rewards_train/chosen': '0.039701', 'rewards_train/rejected': '-0.015781', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.055481', 'logps_train/rejected': '-290.52', 'logps_train/chosen': '-313.46', 'loss/train': '0.67711', 'examples_per_second': '48.902', 'grad_norm': '0', 'counters/examples': 4288, 'counters/updates': 134}
train stats after 4320 examples: {'rewards_train/chosen': '0.04043', 'rewards_train/rejected': '0.03334', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.0070895', 'logps_train/rejected': '-289.77', 'logps_train/chosen': '-310.75', 'loss/train': '0.70201', 'examples_per_second': '56.924', 'grad_norm': '0', 'counters/examples': 4320, 'counters/updates': 135}
train stats after 4352 examples: {'rewards_train/chosen': '0.054278', 'rewards_train/rejected': '0.034771', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.019507', 'logps_train/rejected': '-281.98', 'logps_train/chosen': '-312.04', 'loss/train': '0.69505', 'examples_per_second': '55.93', 'grad_norm': '0', 'counters/examples': 4352, 'counters/updates': 136}
train stats after 4384 examples: {'rewards_train/chosen': '0.017783', 'rewards_train/rejected': '-0.030812', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.048595', 'logps_train/rejected': '-290.93', 'logps_train/chosen': '-319.59', 'loss/train': '0.68876', 'examples_per_second': '65.608', 'grad_norm': '0', 'counters/examples': 4384, 'counters/updates': 137}
train stats after 4416 examples: {'rewards_train/chosen': '0.062529', 'rewards_train/rejected': '0.014391', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.048138', 'logps_train/rejected': '-304.83', 'logps_train/chosen': '-333.64', 'loss/train': '0.68402', 'examples_per_second': '63.589', 'grad_norm': '0', 'counters/examples': 4416, 'counters/updates': 138}
train stats after 4448 examples: {'rewards_train/chosen': '0.031915', 'rewards_train/rejected': '-0.022529', 'rewards_train/accuracies': '0.625', 'rewards_train/margins': '0.054443', 'logps_train/rejected': '-300.3', 'logps_train/chosen': '-313.99', 'loss/train': '0.68004', 'examples_per_second': '63.504', 'grad_norm': '0', 'counters/examples': 4448, 'counters/updates': 139}
train stats after 4480 examples: {'rewards_train/chosen': '0.1062', 'rewards_train/rejected': '0.039937', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.066265', 'logps_train/rejected': '-337.6', 'logps_train/chosen': '-363.89', 'loss/train': '0.67029', 'examples_per_second': '50.313', 'grad_norm': '0', 'counters/examples': 4480, 'counters/updates': 140}
train stats after 4512 examples: {'rewards_train/chosen': '-0.00032717', 'rewards_train/rejected': '-0.0076947', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.0073676', 'logps_train/rejected': '-292.49', 'logps_train/chosen': '-311.67', 'loss/train': '0.70671', 'examples_per_second': '44.311', 'grad_norm': '0', 'counters/examples': 4512, 'counters/updates': 141}
train stats after 4544 examples: {'rewards_train/chosen': '0.025369', 'rewards_train/rejected': '-0.0099071', 'rewards_train/accuracies': '0.46094', 'rewards_train/margins': '0.035276', 'logps_train/rejected': '-287.25', 'logps_train/chosen': '-315', 'loss/train': '0.68973', 'examples_per_second': '53.884', 'grad_norm': '0', 'counters/examples': 4544, 'counters/updates': 142}
train stats after 4576 examples: {'rewards_train/chosen': '0.050203', 'rewards_train/rejected': '-0.012748', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.062951', 'logps_train/rejected': '-289.74', 'logps_train/chosen': '-333.23', 'loss/train': '0.68282', 'examples_per_second': '45.655', 'grad_norm': '0', 'counters/examples': 4576, 'counters/updates': 143}
train stats after 4608 examples: {'rewards_train/chosen': '0.040925', 'rewards_train/rejected': '0.022144', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.01878', 'logps_train/rejected': '-305.43', 'logps_train/chosen': '-330.04', 'loss/train': '0.70046', 'examples_per_second': '64.848', 'grad_norm': '0', 'counters/examples': 4608, 'counters/updates': 144}
train stats after 4640 examples: {'rewards_train/chosen': '0.072947', 'rewards_train/rejected': '-0.016243', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.08919', 'logps_train/rejected': '-282.02', 'logps_train/chosen': '-326.27', 'loss/train': '0.66406', 'examples_per_second': '57.29', 'grad_norm': '0', 'counters/examples': 4640, 'counters/updates': 145}
train stats after 4672 examples: {'rewards_train/chosen': '0.063479', 'rewards_train/rejected': '0.064285', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.00080648', 'logps_train/rejected': '-303.91', 'logps_train/chosen': '-320.93', 'loss/train': '0.70859', 'examples_per_second': '49.029', 'grad_norm': '0', 'counters/examples': 4672, 'counters/updates': 146}
train stats after 4704 examples: {'rewards_train/chosen': '0.038186', 'rewards_train/rejected': '0.033325', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.0048601', 'logps_train/rejected': '-256.39', 'logps_train/chosen': '-299.82', 'loss/train': '0.70504', 'examples_per_second': '49.422', 'grad_norm': '0', 'counters/examples': 4704, 'counters/updates': 147}
train stats after 4736 examples: {'rewards_train/chosen': '0.038504', 'rewards_train/rejected': '-0.026307', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.064811', 'logps_train/rejected': '-260.8', 'logps_train/chosen': '-304.15', 'loss/train': '0.67609', 'examples_per_second': '65.539', 'grad_norm': '0', 'counters/examples': 4736, 'counters/updates': 148}
train stats after 4768 examples: {'rewards_train/chosen': '0.044279', 'rewards_train/rejected': '-0.0039838', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.048263', 'logps_train/rejected': '-313.41', 'logps_train/chosen': '-334.14', 'loss/train': '0.68666', 'examples_per_second': '52.222', 'grad_norm': '0', 'counters/examples': 4768, 'counters/updates': 149}
train stats after 4800 examples: {'rewards_train/chosen': '0.013851', 'rewards_train/rejected': '0.023848', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '-0.0099974', 'logps_train/rejected': '-289.84', 'logps_train/chosen': '-308.59', 'loss/train': '0.71302', 'examples_per_second': '54.17', 'grad_norm': '0', 'counters/examples': 4800, 'counters/updates': 150}
train stats after 4832 examples: {'rewards_train/chosen': '0.071844', 'rewards_train/rejected': '0.039478', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.032366', 'logps_train/rejected': '-319.21', 'logps_train/chosen': '-331.74', 'loss/train': '0.69434', 'examples_per_second': '56.446', 'grad_norm': '0', 'counters/examples': 4832, 'counters/updates': 151}
train stats after 4864 examples: {'rewards_train/chosen': '0.056875', 'rewards_train/rejected': '-0.00034', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.057215', 'logps_train/rejected': '-267.71', 'logps_train/chosen': '-311.2', 'loss/train': '0.67934', 'examples_per_second': '46.513', 'grad_norm': '0', 'counters/examples': 4864, 'counters/updates': 152}
train stats after 4896 examples: {'rewards_train/chosen': '-0.0047409', 'rewards_train/rejected': '0.012001', 'rewards_train/accuracies': '0.47656', 'rewards_train/margins': '-0.016742', 'logps_train/rejected': '-283.4', 'logps_train/chosen': '-313.3', 'loss/train': '0.71592', 'examples_per_second': '49.761', 'grad_norm': '0', 'counters/examples': 4896, 'counters/updates': 153}
train stats after 4928 examples: {'rewards_train/chosen': '0.087053', 'rewards_train/rejected': '-0.001988', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.089041', 'logps_train/rejected': '-264.97', 'logps_train/chosen': '-289.4', 'loss/train': '0.66555', 'examples_per_second': '48.373', 'grad_norm': '0', 'counters/examples': 4928, 'counters/updates': 154}
train stats after 4960 examples: {'rewards_train/chosen': '0.029023', 'rewards_train/rejected': '-0.029658', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.058681', 'logps_train/rejected': '-285.71', 'logps_train/chosen': '-308.11', 'loss/train': '0.6809', 'examples_per_second': '52.094', 'grad_norm': '0', 'counters/examples': 4960, 'counters/updates': 155}
==================================================
==================================================
Running evaluation after 4960 train examples
==================================================
==================================================
train stats after 4992 examples: {'rewards_train/chosen': '0.026424', 'rewards_train/rejected': '0.016803', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.0096209', 'logps_train/rejected': '-294.51', 'logps_train/chosen': '-316.09', 'loss/train': '0.70172', 'examples_per_second': '58.509', 'grad_norm': '0', 'counters/examples': 4992, 'counters/updates': 156}
train stats after 5024 examples: {'rewards_train/chosen': '-0.0029086', 'rewards_train/rejected': '0.011088', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.013997', 'logps_train/rejected': '-302.93', 'logps_train/chosen': '-332.47', 'loss/train': '0.71665', 'examples_per_second': '69.638', 'grad_norm': '0', 'counters/examples': 5024, 'counters/updates': 157}
train stats after 5056 examples: {'rewards_train/chosen': '0.01289', 'rewards_train/rejected': '-0.040417', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.053307', 'logps_train/rejected': '-316.37', 'logps_train/chosen': '-347.06', 'loss/train': '0.68334', 'examples_per_second': '54.188', 'grad_norm': '0', 'counters/examples': 5056, 'counters/updates': 158}
train stats after 5088 examples: {'rewards_train/chosen': '0.060686', 'rewards_train/rejected': '-0.0061166', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.066803', 'logps_train/rejected': '-277.7', 'logps_train/chosen': '-301.86', 'loss/train': '0.67588', 'examples_per_second': '56.532', 'grad_norm': '0', 'counters/examples': 5088, 'counters/updates': 159}
train stats after 5120 examples: {'rewards_train/chosen': '0.016199', 'rewards_train/rejected': '0.017603', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '-0.001404', 'logps_train/rejected': '-289.9', 'logps_train/chosen': '-331.02', 'loss/train': '0.71029', 'examples_per_second': '49.502', 'grad_norm': '0', 'counters/examples': 5120, 'counters/updates': 160}
train stats after 5152 examples: {'rewards_train/chosen': '0.064413', 'rewards_train/rejected': '0.022312', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.042101', 'logps_train/rejected': '-304.83', 'logps_train/chosen': '-333.33', 'loss/train': '0.6866', 'examples_per_second': '55.991', 'grad_norm': '0', 'counters/examples': 5152, 'counters/updates': 161}
train stats after 5184 examples: {'rewards_train/chosen': '0.07942', 'rewards_train/rejected': '0.018568', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.060852', 'logps_train/rejected': '-303.44', 'logps_train/chosen': '-321.38', 'loss/train': '0.67817', 'examples_per_second': '50.116', 'grad_norm': '0', 'counters/examples': 5184, 'counters/updates': 162}
train stats after 5216 examples: {'rewards_train/chosen': '0.040839', 'rewards_train/rejected': '0.047522', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '-0.0066832', 'logps_train/rejected': '-282.76', 'logps_train/chosen': '-310.92', 'loss/train': '0.70739', 'examples_per_second': '63.061', 'grad_norm': '0', 'counters/examples': 5216, 'counters/updates': 163}
train stats after 5248 examples: {'rewards_train/chosen': '0.086064', 'rewards_train/rejected': '0.074518', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.011546', 'logps_train/rejected': '-306.09', 'logps_train/chosen': '-323.62', 'loss/train': '0.7007', 'examples_per_second': '66.715', 'grad_norm': '0', 'counters/examples': 5248, 'counters/updates': 164}
train stats after 5280 examples: {'rewards_train/chosen': '-0.018834', 'rewards_train/rejected': '0.011713', 'rewards_train/accuracies': '0.45312', 'rewards_train/margins': '-0.030547', 'logps_train/rejected': '-308.84', 'logps_train/chosen': '-299.9', 'loss/train': '0.72012', 'examples_per_second': '59.459', 'grad_norm': '0', 'counters/examples': 5280, 'counters/updates': 165}
train stats after 5312 examples: {'rewards_train/chosen': '0.050721', 'rewards_train/rejected': '0.033919', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '0.016802', 'logps_train/rejected': '-303.22', 'logps_train/chosen': '-322.21', 'loss/train': '0.69938', 'examples_per_second': '55.939', 'grad_norm': '0', 'counters/examples': 5312, 'counters/updates': 166}
train stats after 5344 examples: {'rewards_train/chosen': '0.039804', 'rewards_train/rejected': '0.004442', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.035362', 'logps_train/rejected': '-328.85', 'logps_train/chosen': '-325.51', 'loss/train': '0.68943', 'examples_per_second': '47.928', 'grad_norm': '0', 'counters/examples': 5344, 'counters/updates': 167}
train stats after 5376 examples: {'rewards_train/chosen': '0.051442', 'rewards_train/rejected': '-0.043489', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.094931', 'logps_train/rejected': '-301.72', 'logps_train/chosen': '-321.05', 'loss/train': '0.65993', 'examples_per_second': '56.595', 'grad_norm': '0', 'counters/examples': 5376, 'counters/updates': 168}
train stats after 5408 examples: {'rewards_train/chosen': '0.0059959', 'rewards_train/rejected': '-0.01421', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.020206', 'logps_train/rejected': '-278.82', 'logps_train/chosen': '-303.87', 'loss/train': '0.70058', 'examples_per_second': '63.484', 'grad_norm': '0', 'counters/examples': 5408, 'counters/updates': 169}
train stats after 5440 examples: {'rewards_train/chosen': '0.019633', 'rewards_train/rejected': '0.0075408', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.012093', 'logps_train/rejected': '-280.53', 'logps_train/chosen': '-288.46', 'loss/train': '0.6991', 'examples_per_second': '68.732', 'grad_norm': '0', 'counters/examples': 5440, 'counters/updates': 170}
train stats after 5472 examples: {'rewards_train/chosen': '0.029738', 'rewards_train/rejected': '-0.018558', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.048296', 'logps_train/rejected': '-308.1', 'logps_train/chosen': '-322.6', 'loss/train': '0.68262', 'examples_per_second': '70.123', 'grad_norm': '0', 'counters/examples': 5472, 'counters/updates': 171}
train stats after 5504 examples: {'rewards_train/chosen': '0.051727', 'rewards_train/rejected': '0.0060893', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.045638', 'logps_train/rejected': '-253.64', 'logps_train/chosen': '-279.93', 'loss/train': '0.6852', 'examples_per_second': '53.583', 'grad_norm': '0', 'counters/examples': 5504, 'counters/updates': 172}
train stats after 5536 examples: {'rewards_train/chosen': '0.03741', 'rewards_train/rejected': '0.0151', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.02231', 'logps_train/rejected': '-304.34', 'logps_train/chosen': '-301.05', 'loss/train': '0.6948', 'examples_per_second': '46.865', 'grad_norm': '0', 'counters/examples': 5536, 'counters/updates': 173}
train stats after 5568 examples: {'rewards_train/chosen': '0.0031375', 'rewards_train/rejected': '-0.039677', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.042815', 'logps_train/rejected': '-306.06', 'logps_train/chosen': '-325.7', 'loss/train': '0.68736', 'examples_per_second': '52.923', 'grad_norm': '0', 'counters/examples': 5568, 'counters/updates': 174}
train stats after 5600 examples: {'rewards_train/chosen': '0.018199', 'rewards_train/rejected': '-0.035583', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.053781', 'logps_train/rejected': '-289.14', 'logps_train/chosen': '-304.92', 'loss/train': '0.68238', 'examples_per_second': '63.047', 'grad_norm': '0', 'counters/examples': 5600, 'counters/updates': 175}
train stats after 5632 examples: {'rewards_train/chosen': '0.019203', 'rewards_train/rejected': '-0.006051', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '0.025254', 'logps_train/rejected': '-294.97', 'logps_train/chosen': '-334.5', 'loss/train': '0.69518', 'examples_per_second': '60.23', 'grad_norm': '0', 'counters/examples': 5632, 'counters/updates': 176}
train stats after 5664 examples: {'rewards_train/chosen': '0.0064482', 'rewards_train/rejected': '-0.010665', 'rewards_train/accuracies': '0.42969', 'rewards_train/margins': '0.017113', 'logps_train/rejected': '-268.02', 'logps_train/chosen': '-291.99', 'loss/train': '0.70069', 'examples_per_second': '70.907', 'grad_norm': '0', 'counters/examples': 5664, 'counters/updates': 177}
train stats after 5696 examples: {'rewards_train/chosen': '0.01436', 'rewards_train/rejected': '0.018028', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '-0.0036673', 'logps_train/rejected': '-305.87', 'logps_train/chosen': '-328.6', 'loss/train': '0.71313', 'examples_per_second': '47.011', 'grad_norm': '0', 'counters/examples': 5696, 'counters/updates': 178}
train stats after 5728 examples: {'rewards_train/chosen': '-0.035465', 'rewards_train/rejected': '-0.018285', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '-0.017181', 'logps_train/rejected': '-297.75', 'logps_train/chosen': '-310.51', 'loss/train': '0.71846', 'examples_per_second': '51.409', 'grad_norm': '0', 'counters/examples': 5728, 'counters/updates': 179}
train stats after 5760 examples: {'rewards_train/chosen': '0.015739', 'rewards_train/rejected': '-0.015643', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.031383', 'logps_train/rejected': '-298.03', 'logps_train/chosen': '-293.7', 'loss/train': '0.6951', 'examples_per_second': '70.991', 'grad_norm': '0', 'counters/examples': 5760, 'counters/updates': 180}
train stats after 5792 examples: {'rewards_train/chosen': '0.01985', 'rewards_train/rejected': '0.040511', 'rewards_train/accuracies': '0.45312', 'rewards_train/margins': '-0.020662', 'logps_train/rejected': '-310.58', 'logps_train/chosen': '-334.42', 'loss/train': '0.71906', 'examples_per_second': '47.468', 'grad_norm': '0', 'counters/examples': 5792, 'counters/updates': 181}
train stats after 5824 examples: {'rewards_train/chosen': '0.016654', 'rewards_train/rejected': '0.01264', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.0040143', 'logps_train/rejected': '-289.63', 'logps_train/chosen': '-304.29', 'loss/train': '0.70625', 'examples_per_second': '45.724', 'grad_norm': '0', 'counters/examples': 5824, 'counters/updates': 182}
train stats after 5856 examples: {'rewards_train/chosen': '0.045011', 'rewards_train/rejected': '0.0080489', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.036963', 'logps_train/rejected': '-287.64', 'logps_train/chosen': '-290.16', 'loss/train': '0.68697', 'examples_per_second': '54.202', 'grad_norm': '0', 'counters/examples': 5856, 'counters/updates': 183}
train stats after 5888 examples: {'rewards_train/chosen': '0.026335', 'rewards_train/rejected': '-0.021527', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.047862', 'logps_train/rejected': '-281.92', 'logps_train/chosen': '-325.71', 'loss/train': '0.68029', 'examples_per_second': '67.839', 'grad_norm': '0', 'counters/examples': 5888, 'counters/updates': 184}
train stats after 5920 examples: {'rewards_train/chosen': '0.096785', 'rewards_train/rejected': '-0.035371', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.13216', 'logps_train/rejected': '-324.57', 'logps_train/chosen': '-335.68', 'loss/train': '0.6487', 'examples_per_second': '60.276', 'grad_norm': '0', 'counters/examples': 5920, 'counters/updates': 185}
train stats after 5952 examples: {'rewards_train/chosen': '0.01808', 'rewards_train/rejected': '-0.012369', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.030449', 'logps_train/rejected': '-293.57', 'logps_train/chosen': '-290.55', 'loss/train': '0.69514', 'examples_per_second': '51.792', 'grad_norm': '0', 'counters/examples': 5952, 'counters/updates': 186}
train stats after 5984 examples: {'rewards_train/chosen': '0.0077515', 'rewards_train/rejected': '-0.043937', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.051688', 'logps_train/rejected': '-265.54', 'logps_train/chosen': '-280.7', 'loss/train': '0.67911', 'examples_per_second': '55.105', 'grad_norm': '0', 'counters/examples': 5984, 'counters/updates': 187}
train stats after 6016 examples: {'rewards_train/chosen': '0.025617', 'rewards_train/rejected': '-0.032335', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.057952', 'logps_train/rejected': '-305.83', 'logps_train/chosen': '-322.55', 'loss/train': '0.68138', 'examples_per_second': '69.727', 'grad_norm': '0', 'counters/examples': 6016, 'counters/updates': 188}
train stats after 6048 examples: {'rewards_train/chosen': '0.043768', 'rewards_train/rejected': '0.012432', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.031336', 'logps_train/rejected': '-301.02', 'logps_train/chosen': '-338.17', 'loss/train': '0.69186', 'examples_per_second': '46.476', 'grad_norm': '0', 'counters/examples': 6048, 'counters/updates': 189}
train stats after 6080 examples: {'rewards_train/chosen': '0.044459', 'rewards_train/rejected': '-0.022028', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.066487', 'logps_train/rejected': '-299.33', 'logps_train/chosen': '-314.64', 'loss/train': '0.67366', 'examples_per_second': '53.89', 'grad_norm': '0', 'counters/examples': 6080, 'counters/updates': 190}
train stats after 6112 examples: {'rewards_train/chosen': '0.0042833', 'rewards_train/rejected': '0.02043', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '-0.016147', 'logps_train/rejected': '-273.93', 'logps_train/chosen': '-305.5', 'loss/train': '0.71676', 'examples_per_second': '49.125', 'grad_norm': '0', 'counters/examples': 6112, 'counters/updates': 191}
train stats after 6144 examples: {'rewards_train/chosen': '0.035701', 'rewards_train/rejected': '-0.042825', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.078526', 'logps_train/rejected': '-342.79', 'logps_train/chosen': '-326.15', 'loss/train': '0.67283', 'examples_per_second': '54.173', 'grad_norm': '0', 'counters/examples': 6144, 'counters/updates': 192}
train stats after 6176 examples: {'rewards_train/chosen': '0.044381', 'rewards_train/rejected': '-0.014523', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.058905', 'logps_train/rejected': '-291.93', 'logps_train/chosen': '-305.89', 'loss/train': '0.68014', 'examples_per_second': '69.531', 'grad_norm': '0', 'counters/examples': 6176, 'counters/updates': 193}
train stats after 6208 examples: {'rewards_train/chosen': '0.011475', 'rewards_train/rejected': '-0.040433', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.051908', 'logps_train/rejected': '-279.14', 'logps_train/chosen': '-314.95', 'loss/train': '0.68276', 'examples_per_second': '51.582', 'grad_norm': '0', 'counters/examples': 6208, 'counters/updates': 194}
train stats after 6240 examples: {'rewards_train/chosen': '-0.0053428', 'rewards_train/rejected': '-0.0040495', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '-0.0012933', 'logps_train/rejected': '-276.56', 'logps_train/chosen': '-303.71', 'loss/train': '0.70972', 'examples_per_second': '65.989', 'grad_norm': '0', 'counters/examples': 6240, 'counters/updates': 195}
train stats after 6272 examples: {'rewards_train/chosen': '0.026051', 'rewards_train/rejected': '-0.033183', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.059234', 'logps_train/rejected': '-300.16', 'logps_train/chosen': '-323.05', 'loss/train': '0.68332', 'examples_per_second': '60.433', 'grad_norm': '0', 'counters/examples': 6272, 'counters/updates': 196}
train stats after 6304 examples: {'rewards_train/chosen': '0.037688', 'rewards_train/rejected': '-0.023166', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.060854', 'logps_train/rejected': '-271.91', 'logps_train/chosen': '-291.28', 'loss/train': '0.67753', 'examples_per_second': '65.881', 'grad_norm': '0', 'counters/examples': 6304, 'counters/updates': 197}
train stats after 6336 examples: {'rewards_train/chosen': '0.075708', 'rewards_train/rejected': '0.0065531', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.069155', 'logps_train/rejected': '-335.2', 'logps_train/chosen': '-340.4', 'loss/train': '0.67303', 'examples_per_second': '51.823', 'grad_norm': '0', 'counters/examples': 6336, 'counters/updates': 198}
train stats after 6368 examples: {'rewards_train/chosen': '0.033982', 'rewards_train/rejected': '0.03028', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.0037023', 'logps_train/rejected': '-344.12', 'logps_train/chosen': '-363.45', 'loss/train': '0.70647', 'examples_per_second': '46.88', 'grad_norm': '0', 'counters/examples': 6368, 'counters/updates': 199}
train stats after 6400 examples: {'rewards_train/chosen': '0.058253', 'rewards_train/rejected': '0.045385', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.012868', 'logps_train/rejected': '-312.79', 'logps_train/chosen': '-342.55', 'loss/train': '0.70294', 'examples_per_second': '30.319', 'grad_norm': '0', 'counters/examples': 6400, 'counters/updates': 200}
train stats after 6432 examples: {'rewards_train/chosen': '0.07794', 'rewards_train/rejected': '0.033737', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.044203', 'logps_train/rejected': '-323.29', 'logps_train/chosen': '-325.37', 'loss/train': '0.68833', 'examples_per_second': '44.758', 'grad_norm': '0', 'counters/examples': 6432, 'counters/updates': 201}
train stats after 6464 examples: {'rewards_train/chosen': '0.039711', 'rewards_train/rejected': '0.033198', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.0065135', 'logps_train/rejected': '-316.92', 'logps_train/chosen': '-330', 'loss/train': '0.70933', 'examples_per_second': '48.348', 'grad_norm': '0', 'counters/examples': 6464, 'counters/updates': 202}
train stats after 6496 examples: {'rewards_train/chosen': '0.049881', 'rewards_train/rejected': '0.032381', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.0175', 'logps_train/rejected': '-288.68', 'logps_train/chosen': '-316.19', 'loss/train': '0.69876', 'examples_per_second': '48.718', 'grad_norm': '0', 'counters/examples': 6496, 'counters/updates': 203}
train stats after 6528 examples: {'rewards_train/chosen': '0.10268', 'rewards_train/rejected': '0.041995', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.060682', 'logps_train/rejected': '-290.26', 'logps_train/chosen': '-307.99', 'loss/train': '0.67679', 'examples_per_second': '54.722', 'grad_norm': '0', 'counters/examples': 6528, 'counters/updates': 204}
train stats after 6560 examples: {'rewards_train/chosen': '0.044902', 'rewards_train/rejected': '0.038996', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.0059054', 'logps_train/rejected': '-299.66', 'logps_train/chosen': '-319.8', 'loss/train': '0.70348', 'examples_per_second': '71.522', 'grad_norm': '0', 'counters/examples': 6560, 'counters/updates': 205}
train stats after 6592 examples: {'rewards_train/chosen': '0.059964', 'rewards_train/rejected': '0.075093', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '-0.015129', 'logps_train/rejected': '-256.08', 'logps_train/chosen': '-280.94', 'loss/train': '0.71535', 'examples_per_second': '51.276', 'grad_norm': '0', 'counters/examples': 6592, 'counters/updates': 206}
train stats after 6624 examples: {'rewards_train/chosen': '0.054491', 'rewards_train/rejected': '-0.02363', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.07812', 'logps_train/rejected': '-284.02', 'logps_train/chosen': '-306.43', 'loss/train': '0.66973', 'examples_per_second': '56.662', 'grad_norm': '0', 'counters/examples': 6624, 'counters/updates': 207}
train stats after 6656 examples: {'rewards_train/chosen': '0.077583', 'rewards_train/rejected': '0.063312', 'rewards_train/accuracies': '0.47656', 'rewards_train/margins': '0.014271', 'logps_train/rejected': '-275.48', 'logps_train/chosen': '-314.96', 'loss/train': '0.70509', 'examples_per_second': '45.747', 'grad_norm': '0', 'counters/examples': 6656, 'counters/updates': 208}
train stats after 6688 examples: {'rewards_train/chosen': '0.0099447', 'rewards_train/rejected': '-0.010955', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.0209', 'logps_train/rejected': '-302.61', 'logps_train/chosen': '-313.68', 'loss/train': '0.70209', 'examples_per_second': '69.034', 'grad_norm': '0', 'counters/examples': 6688, 'counters/updates': 209}
train stats after 6720 examples: {'rewards_train/chosen': '0.065334', 'rewards_train/rejected': '0.057952', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.0073817', 'logps_train/rejected': '-317.2', 'logps_train/chosen': '-346.2', 'loss/train': '0.70966', 'examples_per_second': '46.331', 'grad_norm': '0', 'counters/examples': 6720, 'counters/updates': 210}
train stats after 6752 examples: {'rewards_train/chosen': '0.081176', 'rewards_train/rejected': '0.055471', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.025705', 'logps_train/rejected': '-308.43', 'logps_train/chosen': '-314.98', 'loss/train': '0.69217', 'examples_per_second': '48.158', 'grad_norm': '0', 'counters/examples': 6752, 'counters/updates': 211}
train stats after 6784 examples: {'rewards_train/chosen': '0.068842', 'rewards_train/rejected': '-0.01432', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.083162', 'logps_train/rejected': '-310.46', 'logps_train/chosen': '-345.49', 'loss/train': '0.67361', 'examples_per_second': '50.473', 'grad_norm': '0', 'counters/examples': 6784, 'counters/updates': 212}
train stats after 6816 examples: {'rewards_train/chosen': '0.053224', 'rewards_train/rejected': '0.044719', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.0085057', 'logps_train/rejected': '-304.83', 'logps_train/chosen': '-324.74', 'loss/train': '0.70447', 'examples_per_second': '57.006', 'grad_norm': '0', 'counters/examples': 6816, 'counters/updates': 213}
train stats after 6848 examples: {'rewards_train/chosen': '0.10211', 'rewards_train/rejected': '0.014761', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.087354', 'logps_train/rejected': '-287.85', 'logps_train/chosen': '-312.56', 'loss/train': '0.6638', 'examples_per_second': '49.109', 'grad_norm': '0', 'counters/examples': 6848, 'counters/updates': 214}
train stats after 6880 examples: {'rewards_train/chosen': '0.037533', 'rewards_train/rejected': '-0.03642', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.073953', 'logps_train/rejected': '-265', 'logps_train/chosen': '-293.93', 'loss/train': '0.669', 'examples_per_second': '61.503', 'grad_norm': '0', 'counters/examples': 6880, 'counters/updates': 215}
train stats after 6912 examples: {'rewards_train/chosen': '0.090591', 'rewards_train/rejected': '0.056135', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.034455', 'logps_train/rejected': '-296.35', 'logps_train/chosen': '-310.58', 'loss/train': '0.69509', 'examples_per_second': '67.947', 'grad_norm': '0', 'counters/examples': 6912, 'counters/updates': 216}
train stats after 6944 examples: {'rewards_train/chosen': '0.044343', 'rewards_train/rejected': '0.022128', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.022216', 'logps_train/rejected': '-330.27', 'logps_train/chosen': '-339.58', 'loss/train': '0.70423', 'examples_per_second': '61.653', 'grad_norm': '0', 'counters/examples': 6944, 'counters/updates': 217}
train stats after 6976 examples: {'rewards_train/chosen': '0.027407', 'rewards_train/rejected': '0.024691', 'rewards_train/accuracies': '0.4375', 'rewards_train/margins': '0.0027162', 'logps_train/rejected': '-303.31', 'logps_train/chosen': '-330.21', 'loss/train': '0.71022', 'examples_per_second': '53.443', 'grad_norm': '0', 'counters/examples': 6976, 'counters/updates': 218}
train stats after 7008 examples: {'rewards_train/chosen': '0.018115', 'rewards_train/rejected': '-0.016721', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.034836', 'logps_train/rejected': '-272.08', 'logps_train/chosen': '-296', 'loss/train': '0.69718', 'examples_per_second': '49.61', 'grad_norm': '0', 'counters/examples': 7008, 'counters/updates': 219}
train stats after 7040 examples: {'rewards_train/chosen': '0.044359', 'rewards_train/rejected': '0.022799', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.02156', 'logps_train/rejected': '-290.21', 'logps_train/chosen': '-322.21', 'loss/train': '0.70168', 'examples_per_second': '64.321', 'grad_norm': '0', 'counters/examples': 7040, 'counters/updates': 220}
train stats after 7072 examples: {'rewards_train/chosen': '0.082883', 'rewards_train/rejected': '-0.0021295', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.085012', 'logps_train/rejected': '-286.89', 'logps_train/chosen': '-324.1', 'loss/train': '0.67184', 'examples_per_second': '53.127', 'grad_norm': '0', 'counters/examples': 7072, 'counters/updates': 221}
train stats after 7104 examples: {'rewards_train/chosen': '0.047186', 'rewards_train/rejected': '0.083989', 'rewards_train/accuracies': '0.4375', 'rewards_train/margins': '-0.036803', 'logps_train/rejected': '-319.3', 'logps_train/chosen': '-330.03', 'loss/train': '0.72905', 'examples_per_second': '66.372', 'grad_norm': '0', 'counters/examples': 7104, 'counters/updates': 222}
train stats after 7136 examples: {'rewards_train/chosen': '0.019188', 'rewards_train/rejected': '-0.043242', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.06243', 'logps_train/rejected': '-307.02', 'logps_train/chosen': '-318.89', 'loss/train': '0.67935', 'examples_per_second': '47.432', 'grad_norm': '0', 'counters/examples': 7136, 'counters/updates': 223}
train stats after 7168 examples: {'rewards_train/chosen': '0.094628', 'rewards_train/rejected': '0.046489', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.048139', 'logps_train/rejected': '-298.5', 'logps_train/chosen': '-308.94', 'loss/train': '0.68651', 'examples_per_second': '52.015', 'grad_norm': '0', 'counters/examples': 7168, 'counters/updates': 224}
train stats after 7200 examples: {'rewards_train/chosen': '0.041585', 'rewards_train/rejected': '0.026716', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.014869', 'logps_train/rejected': '-281.63', 'logps_train/chosen': '-298.25', 'loss/train': '0.70202', 'examples_per_second': '50.163', 'grad_norm': '0', 'counters/examples': 7200, 'counters/updates': 225}
train stats after 7232 examples: {'rewards_train/chosen': '0.058124', 'rewards_train/rejected': '-0.0049248', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.063049', 'logps_train/rejected': '-267.83', 'logps_train/chosen': '-296.86', 'loss/train': '0.67942', 'examples_per_second': '71.416', 'grad_norm': '0', 'counters/examples': 7232, 'counters/updates': 226}
train stats after 7264 examples: {'rewards_train/chosen': '0.065196', 'rewards_train/rejected': '0.038738', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.026458', 'logps_train/rejected': '-307.95', 'logps_train/chosen': '-300.94', 'loss/train': '0.69614', 'examples_per_second': '66.418', 'grad_norm': '0', 'counters/examples': 7264, 'counters/updates': 227}
train stats after 7296 examples: {'rewards_train/chosen': '0.084081', 'rewards_train/rejected': '-0.032385', 'rewards_train/accuracies': '0.625', 'rewards_train/margins': '0.11647', 'logps_train/rejected': '-292.85', 'logps_train/chosen': '-331.77', 'loss/train': '0.65505', 'examples_per_second': '57.198', 'grad_norm': '0', 'counters/examples': 7296, 'counters/updates': 228}
train stats after 7328 examples: {'rewards_train/chosen': '0.067681', 'rewards_train/rejected': '0.038738', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.028942', 'logps_train/rejected': '-316.24', 'logps_train/chosen': '-334.79', 'loss/train': '0.69848', 'examples_per_second': '50.436', 'grad_norm': '0', 'counters/examples': 7328, 'counters/updates': 229}
train stats after 7360 examples: {'rewards_train/chosen': '0.047856', 'rewards_train/rejected': '0.019532', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.028324', 'logps_train/rejected': '-305.89', 'logps_train/chosen': '-295.63', 'loss/train': '0.69652', 'examples_per_second': '61.967', 'grad_norm': '0', 'counters/examples': 7360, 'counters/updates': 230}
train stats after 7392 examples: {'rewards_train/chosen': '0.16075', 'rewards_train/rejected': '0.070161', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.090588', 'logps_train/rejected': '-293.75', 'logps_train/chosen': '-323.69', 'loss/train': '0.66784', 'examples_per_second': '57.214', 'grad_norm': '0', 'counters/examples': 7392, 'counters/updates': 231}
train stats after 7424 examples: {'rewards_train/chosen': '0.12394', 'rewards_train/rejected': '0.053521', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.070415', 'logps_train/rejected': '-280.75', 'logps_train/chosen': '-320.5', 'loss/train': '0.67414', 'examples_per_second': '63.586', 'grad_norm': '0', 'counters/examples': 7424, 'counters/updates': 232}
train stats after 7456 examples: {'rewards_train/chosen': '0.11863', 'rewards_train/rejected': '0.02022', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.098412', 'logps_train/rejected': '-291.15', 'logps_train/chosen': '-301.51', 'loss/train': '0.66059', 'examples_per_second': '70.339', 'grad_norm': '0', 'counters/examples': 7456, 'counters/updates': 233}
train stats after 7488 examples: {'rewards_train/chosen': '0.044217', 'rewards_train/rejected': '0.07375', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.029533', 'logps_train/rejected': '-310.12', 'logps_train/chosen': '-346.91', 'loss/train': '0.72582', 'examples_per_second': '72.389', 'grad_norm': '0', 'counters/examples': 7488, 'counters/updates': 234}
train stats after 7520 examples: {'rewards_train/chosen': '0.089809', 'rewards_train/rejected': '0.065076', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.024733', 'logps_train/rejected': '-336.37', 'logps_train/chosen': '-324.03', 'loss/train': '0.70248', 'examples_per_second': '47.443', 'grad_norm': '0', 'counters/examples': 7520, 'counters/updates': 235}
train stats after 7552 examples: {'rewards_train/chosen': '0.022339', 'rewards_train/rejected': '-0.011648', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.033987', 'logps_train/rejected': '-280.93', 'logps_train/chosen': '-316.35', 'loss/train': '0.69615', 'examples_per_second': '51.589', 'grad_norm': '0', 'counters/examples': 7552, 'counters/updates': 236}
train stats after 7584 examples: {'rewards_train/chosen': '0.097672', 'rewards_train/rejected': '0.033567', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.064106', 'logps_train/rejected': '-292.27', 'logps_train/chosen': '-322.77', 'loss/train': '0.68252', 'examples_per_second': '47.466', 'grad_norm': '0', 'counters/examples': 7584, 'counters/updates': 237}
train stats after 7616 examples: {'rewards_train/chosen': '0.06247', 'rewards_train/rejected': '0.011758', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.050712', 'logps_train/rejected': '-278.71', 'logps_train/chosen': '-284.73', 'loss/train': '0.69391', 'examples_per_second': '73.518', 'grad_norm': '0', 'counters/examples': 7616, 'counters/updates': 238}
train stats after 7648 examples: {'rewards_train/chosen': '0.10661', 'rewards_train/rejected': '0.02623', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.080384', 'logps_train/rejected': '-257.73', 'logps_train/chosen': '-303.37', 'loss/train': '0.67237', 'examples_per_second': '56.667', 'grad_norm': '0', 'counters/examples': 7648, 'counters/updates': 239}
train stats after 7680 examples: {'rewards_train/chosen': '0.13687', 'rewards_train/rejected': '0.096078', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.040788', 'logps_train/rejected': '-290.29', 'logps_train/chosen': '-323.76', 'loss/train': '0.69296', 'examples_per_second': '69.163', 'grad_norm': '0', 'counters/examples': 7680, 'counters/updates': 240}
train stats after 7712 examples: {'rewards_train/chosen': '0.080908', 'rewards_train/rejected': '-0.0054996', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.086407', 'logps_train/rejected': '-306.77', 'logps_train/chosen': '-321.21', 'loss/train': '0.67086', 'examples_per_second': '47.502', 'grad_norm': '0', 'counters/examples': 7712, 'counters/updates': 241}
train stats after 7744 examples: {'rewards_train/chosen': '0.059138', 'rewards_train/rejected': '0.044571', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.014566', 'logps_train/rejected': '-304.34', 'logps_train/chosen': '-313.95', 'loss/train': '0.70599', 'examples_per_second': '66.137', 'grad_norm': '0', 'counters/examples': 7744, 'counters/updates': 242}
train stats after 7776 examples: {'rewards_train/chosen': '0.093911', 'rewards_train/rejected': '0.11065', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '-0.016736', 'logps_train/rejected': '-304.39', 'logps_train/chosen': '-329.6', 'loss/train': '0.71609', 'examples_per_second': '47.54', 'grad_norm': '0', 'counters/examples': 7776, 'counters/updates': 243}
train stats after 7808 examples: {'rewards_train/chosen': '0.053381', 'rewards_train/rejected': '0.024879', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.028502', 'logps_train/rejected': '-309.54', 'logps_train/chosen': '-293.73', 'loss/train': '0.69571', 'examples_per_second': '59.09', 'grad_norm': '0', 'counters/examples': 7808, 'counters/updates': 244}
train stats after 7840 examples: {'rewards_train/chosen': '0.14593', 'rewards_train/rejected': '0.028165', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.11776', 'logps_train/rejected': '-320.44', 'logps_train/chosen': '-323.62', 'loss/train': '0.66152', 'examples_per_second': '48.442', 'grad_norm': '0', 'counters/examples': 7840, 'counters/updates': 245}
train stats after 7872 examples: {'rewards_train/chosen': '0.09149', 'rewards_train/rejected': '0.033557', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.057934', 'logps_train/rejected': '-296.85', 'logps_train/chosen': '-326.1', 'loss/train': '0.68333', 'examples_per_second': '58.106', 'grad_norm': '0', 'counters/examples': 7872, 'counters/updates': 246}
train stats after 7904 examples: {'rewards_train/chosen': '0.095801', 'rewards_train/rejected': '0.0079366', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.087864', 'logps_train/rejected': '-293.84', 'logps_train/chosen': '-315.11', 'loss/train': '0.66635', 'examples_per_second': '54.207', 'grad_norm': '0', 'counters/examples': 7904, 'counters/updates': 247}
train stats after 7936 examples: {'rewards_train/chosen': '0.055227', 'rewards_train/rejected': '0.029161', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.026066', 'logps_train/rejected': '-287.74', 'logps_train/chosen': '-304.98', 'loss/train': '0.69741', 'examples_per_second': '71.252', 'grad_norm': '0', 'counters/examples': 7936, 'counters/updates': 248}
train stats after 7968 examples: {'rewards_train/chosen': '0.049832', 'rewards_train/rejected': '-0.0043334', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.054166', 'logps_train/rejected': '-309.09', 'logps_train/chosen': '-314.55', 'loss/train': '0.68723', 'examples_per_second': '50.487', 'grad_norm': '0', 'counters/examples': 7968, 'counters/updates': 249}
train stats after 8000 examples: {'rewards_train/chosen': '0.047673', 'rewards_train/rejected': '0.038166', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.0095068', 'logps_train/rejected': '-291.06', 'logps_train/chosen': '-299.57', 'loss/train': '0.70493', 'examples_per_second': '48.008', 'grad_norm': '0', 'counters/examples': 8000, 'counters/updates': 250}
train stats after 8032 examples: {'rewards_train/chosen': '0.017236', 'rewards_train/rejected': '0.033743', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '-0.016506', 'logps_train/rejected': '-330.38', 'logps_train/chosen': '-332.84', 'loss/train': '0.71908', 'examples_per_second': '64.021', 'grad_norm': '0', 'counters/examples': 8032, 'counters/updates': 251}
train stats after 8064 examples: {'rewards_train/chosen': '0.09108', 'rewards_train/rejected': '0.030844', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.060236', 'logps_train/rejected': '-314.7', 'logps_train/chosen': '-322.39', 'loss/train': '0.67759', 'examples_per_second': '46.372', 'grad_norm': '0', 'counters/examples': 8064, 'counters/updates': 252}
train stats after 8096 examples: {'rewards_train/chosen': '0.014946', 'rewards_train/rejected': '0.0081828', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.0067632', 'logps_train/rejected': '-309.78', 'logps_train/chosen': '-331.61', 'loss/train': '0.71125', 'examples_per_second': '48.831', 'grad_norm': '0', 'counters/examples': 8096, 'counters/updates': 253}
train stats after 8128 examples: {'rewards_train/chosen': '0.066032', 'rewards_train/rejected': '0.011702', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.05433', 'logps_train/rejected': '-302.91', 'logps_train/chosen': '-327.28', 'loss/train': '0.68548', 'examples_per_second': '66.595', 'grad_norm': '0', 'counters/examples': 8128, 'counters/updates': 254}
train stats after 8160 examples: {'rewards_train/chosen': '0.10204', 'rewards_train/rejected': '0.063106', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.038931', 'logps_train/rejected': '-310.84', 'logps_train/chosen': '-328.76', 'loss/train': '0.69269', 'examples_per_second': '45.661', 'grad_norm': '0', 'counters/examples': 8160, 'counters/updates': 255}
train stats after 8192 examples: {'rewards_train/chosen': '0.033075', 'rewards_train/rejected': '0.037025', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '-0.0039499', 'logps_train/rejected': '-288.56', 'logps_train/chosen': '-307', 'loss/train': '0.71407', 'examples_per_second': '56.573', 'grad_norm': '0', 'counters/examples': 8192, 'counters/updates': 256}
train stats after 8224 examples: {'rewards_train/chosen': '0.037938', 'rewards_train/rejected': '0.015687', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.022251', 'logps_train/rejected': '-297.46', 'logps_train/chosen': '-315.87', 'loss/train': '0.69949', 'examples_per_second': '70.351', 'grad_norm': '0', 'counters/examples': 8224, 'counters/updates': 257}
train stats after 8256 examples: {'rewards_train/chosen': '0.068551', 'rewards_train/rejected': '0.085458', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.016907', 'logps_train/rejected': '-308.03', 'logps_train/chosen': '-324.04', 'loss/train': '0.71674', 'examples_per_second': '53.151', 'grad_norm': '0', 'counters/examples': 8256, 'counters/updates': 258}
train stats after 8288 examples: {'rewards_train/chosen': '0.043272', 'rewards_train/rejected': '-0.040536', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.083809', 'logps_train/rejected': '-268.44', 'logps_train/chosen': '-299.93', 'loss/train': '0.6726', 'examples_per_second': '44.763', 'grad_norm': '0', 'counters/examples': 8288, 'counters/updates': 259}
train stats after 8320 examples: {'rewards_train/chosen': '0.098694', 'rewards_train/rejected': '0.035988', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.062706', 'logps_train/rejected': '-293.99', 'logps_train/chosen': '-322.22', 'loss/train': '0.68135', 'examples_per_second': '49.247', 'grad_norm': '0', 'counters/examples': 8320, 'counters/updates': 260}
train stats after 8352 examples: {'rewards_train/chosen': '0.080006', 'rewards_train/rejected': '-0.01398', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.093986', 'logps_train/rejected': '-301.69', 'logps_train/chosen': '-294.77', 'loss/train': '0.67015', 'examples_per_second': '56.978', 'grad_norm': '0', 'counters/examples': 8352, 'counters/updates': 261}
train stats after 8384 examples: {'rewards_train/chosen': '0.080602', 'rewards_train/rejected': '0.031149', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.049453', 'logps_train/rejected': '-277.68', 'logps_train/chosen': '-313.46', 'loss/train': '0.68853', 'examples_per_second': '44.805', 'grad_norm': '0', 'counters/examples': 8384, 'counters/updates': 262}
train stats after 8416 examples: {'rewards_train/chosen': '0.07786', 'rewards_train/rejected': '-0.0033039', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.081164', 'logps_train/rejected': '-309.21', 'logps_train/chosen': '-299.22', 'loss/train': '0.67197', 'examples_per_second': '46.782', 'grad_norm': '0', 'counters/examples': 8416, 'counters/updates': 263}
train stats after 8448 examples: {'rewards_train/chosen': '0.075642', 'rewards_train/rejected': '-0.011891', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.087533', 'logps_train/rejected': '-319.13', 'logps_train/chosen': '-347.31', 'loss/train': '0.67011', 'examples_per_second': '49.023', 'grad_norm': '0', 'counters/examples': 8448, 'counters/updates': 264}
train stats after 8480 examples: {'rewards_train/chosen': '0.075813', 'rewards_train/rejected': '0.081233', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.0054197', 'logps_train/rejected': '-307.7', 'logps_train/chosen': '-317.77', 'loss/train': '0.71649', 'examples_per_second': '47.113', 'grad_norm': '0', 'counters/examples': 8480, 'counters/updates': 265}
train stats after 8512 examples: {'rewards_train/chosen': '0.076027', 'rewards_train/rejected': '-0.0056275', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.081655', 'logps_train/rejected': '-316.19', 'logps_train/chosen': '-318.89', 'loss/train': '0.67671', 'examples_per_second': '48.966', 'grad_norm': '0', 'counters/examples': 8512, 'counters/updates': 266}
train stats after 8544 examples: {'rewards_train/chosen': '0.090673', 'rewards_train/rejected': '-0.024189', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.11486', 'logps_train/rejected': '-307.92', 'logps_train/chosen': '-340.92', 'loss/train': '0.65539', 'examples_per_second': '46.22', 'grad_norm': '0', 'counters/examples': 8544, 'counters/updates': 267}
train stats after 8576 examples: {'rewards_train/chosen': '0.068331', 'rewards_train/rejected': '-0.022706', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.091037', 'logps_train/rejected': '-292.01', 'logps_train/chosen': '-313.83', 'loss/train': '0.6737', 'examples_per_second': '52.177', 'grad_norm': '0', 'counters/examples': 8576, 'counters/updates': 268}
train stats after 8608 examples: {'rewards_train/chosen': '0.039534', 'rewards_train/rejected': '-0.054965', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.094499', 'logps_train/rejected': '-329.18', 'logps_train/chosen': '-341.33', 'loss/train': '0.66232', 'examples_per_second': '70.911', 'grad_norm': '0', 'counters/examples': 8608, 'counters/updates': 269}
train stats after 8640 examples: {'rewards_train/chosen': '0.078106', 'rewards_train/rejected': '0.030394', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.047711', 'logps_train/rejected': '-312.62', 'logps_train/chosen': '-329.48', 'loss/train': '0.6917', 'examples_per_second': '48.839', 'grad_norm': '0', 'counters/examples': 8640, 'counters/updates': 270}
train stats after 8672 examples: {'rewards_train/chosen': '0.041107', 'rewards_train/rejected': '0.033641', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.0074658', 'logps_train/rejected': '-277.5', 'logps_train/chosen': '-306.32', 'loss/train': '0.71036', 'examples_per_second': '66.166', 'grad_norm': '0', 'counters/examples': 8672, 'counters/updates': 271}
train stats after 8704 examples: {'rewards_train/chosen': '0.066385', 'rewards_train/rejected': '0.054956', 'rewards_train/accuracies': '0.47656', 'rewards_train/margins': '0.011429', 'logps_train/rejected': '-303.69', 'logps_train/chosen': '-323.75', 'loss/train': '0.70843', 'examples_per_second': '63.508', 'grad_norm': '0', 'counters/examples': 8704, 'counters/updates': 272}
train stats after 8736 examples: {'rewards_train/chosen': '0.050522', 'rewards_train/rejected': '0.020965', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.029558', 'logps_train/rejected': '-296.38', 'logps_train/chosen': '-310.96', 'loss/train': '0.70343', 'examples_per_second': '47.263', 'grad_norm': '0', 'counters/examples': 8736, 'counters/updates': 273}
train stats after 8768 examples: {'rewards_train/chosen': '0.060202', 'rewards_train/rejected': '0.0073296', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.052873', 'logps_train/rejected': '-310.81', 'logps_train/chosen': '-327.95', 'loss/train': '0.68452', 'examples_per_second': '65.981', 'grad_norm': '0', 'counters/examples': 8768, 'counters/updates': 274}
train stats after 8800 examples: {'rewards_train/chosen': '0.077819', 'rewards_train/rejected': '0.0029961', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.074823', 'logps_train/rejected': '-273.69', 'logps_train/chosen': '-281.08', 'loss/train': '0.67732', 'examples_per_second': '59.221', 'grad_norm': '0', 'counters/examples': 8800, 'counters/updates': 275}
train stats after 8832 examples: {'rewards_train/chosen': '0.090898', 'rewards_train/rejected': '0.026099', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.064799', 'logps_train/rejected': '-270.5', 'logps_train/chosen': '-302.23', 'loss/train': '0.67789', 'examples_per_second': '71.02', 'grad_norm': '0', 'counters/examples': 8832, 'counters/updates': 276}
train stats after 8864 examples: {'rewards_train/chosen': '0.046724', 'rewards_train/rejected': '-0.026647', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.073371', 'logps_train/rejected': '-299.25', 'logps_train/chosen': '-333.94', 'loss/train': '0.67434', 'examples_per_second': '53.645', 'grad_norm': '0', 'counters/examples': 8864, 'counters/updates': 277}
train stats after 8896 examples: {'rewards_train/chosen': '0.11077', 'rewards_train/rejected': '0.11444', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.0036737', 'logps_train/rejected': '-306.16', 'logps_train/chosen': '-327.38', 'loss/train': '0.7122', 'examples_per_second': '54.628', 'grad_norm': '0', 'counters/examples': 8896, 'counters/updates': 278}
train stats after 8928 examples: {'rewards_train/chosen': '0.07415', 'rewards_train/rejected': '0.026214', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.047937', 'logps_train/rejected': '-294.84', 'logps_train/chosen': '-317.6', 'loss/train': '0.69323', 'examples_per_second': '49.084', 'grad_norm': '0', 'counters/examples': 8928, 'counters/updates': 279}
train stats after 8960 examples: {'rewards_train/chosen': '0.098599', 'rewards_train/rejected': '0.022843', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.075755', 'logps_train/rejected': '-321.21', 'logps_train/chosen': '-334.77', 'loss/train': '0.67393', 'examples_per_second': '49.253', 'grad_norm': '0', 'counters/examples': 8960, 'counters/updates': 280}
train stats after 8992 examples: {'rewards_train/chosen': '0.080928', 'rewards_train/rejected': '0.033822', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.047106', 'logps_train/rejected': '-299.71', 'logps_train/chosen': '-298.25', 'loss/train': '0.68964', 'examples_per_second': '55.069', 'grad_norm': '0', 'counters/examples': 8992, 'counters/updates': 281}
train stats after 9024 examples: {'rewards_train/chosen': '0.094214', 'rewards_train/rejected': '0.070332', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.023881', 'logps_train/rejected': '-275.99', 'logps_train/chosen': '-296.97', 'loss/train': '0.70431', 'examples_per_second': '70.63', 'grad_norm': '0', 'counters/examples': 9024, 'counters/updates': 282}
train stats after 9056 examples: {'rewards_train/chosen': '0.077673', 'rewards_train/rejected': '-0.0034029', 'rewards_train/accuracies': '0.63281', 'rewards_train/margins': '0.081076', 'logps_train/rejected': '-301.06', 'logps_train/chosen': '-313.04', 'loss/train': '0.67918', 'examples_per_second': '45.718', 'grad_norm': '0', 'counters/examples': 9056, 'counters/updates': 283}
train stats after 9088 examples: {'rewards_train/chosen': '0.10986', 'rewards_train/rejected': '0.01673', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.093128', 'logps_train/rejected': '-255.84', 'logps_train/chosen': '-265.75', 'loss/train': '0.66736', 'examples_per_second': '71.974', 'grad_norm': '0', 'counters/examples': 9088, 'counters/updates': 284}
train stats after 9120 examples: {'rewards_train/chosen': '0.09161', 'rewards_train/rejected': '0.033975', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.057635', 'logps_train/rejected': '-336.06', 'logps_train/chosen': '-341.31', 'loss/train': '0.6865', 'examples_per_second': '46.629', 'grad_norm': '0', 'counters/examples': 9120, 'counters/updates': 285}
train stats after 9152 examples: {'rewards_train/chosen': '0.12283', 'rewards_train/rejected': '0.057777', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.065053', 'logps_train/rejected': '-301.16', 'logps_train/chosen': '-313.83', 'loss/train': '0.68273', 'examples_per_second': '52.482', 'grad_norm': '0', 'counters/examples': 9152, 'counters/updates': 286}
train stats after 9184 examples: {'rewards_train/chosen': '0.099118', 'rewards_train/rejected': '0.01903', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.080088', 'logps_train/rejected': '-305.6', 'logps_train/chosen': '-334.2', 'loss/train': '0.68043', 'examples_per_second': '64.442', 'grad_norm': '0', 'counters/examples': 9184, 'counters/updates': 287}
train stats after 9216 examples: {'rewards_train/chosen': '0.08356', 'rewards_train/rejected': '-0.009186', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.092746', 'logps_train/rejected': '-283.18', 'logps_train/chosen': '-307.26', 'loss/train': '0.68777', 'examples_per_second': '49.077', 'grad_norm': '0', 'counters/examples': 9216, 'counters/updates': 288}
train stats after 9248 examples: {'rewards_train/chosen': '0.061029', 'rewards_train/rejected': '0.024366', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.036663', 'logps_train/rejected': '-296.66', 'logps_train/chosen': '-312.09', 'loss/train': '0.69566', 'examples_per_second': '46.909', 'grad_norm': '0', 'counters/examples': 9248, 'counters/updates': 289}
train stats after 9280 examples: {'rewards_train/chosen': '0.11533', 'rewards_train/rejected': '0.080403', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.034928', 'logps_train/rejected': '-321.28', 'logps_train/chosen': '-321.64', 'loss/train': '0.70269', 'examples_per_second': '66.061', 'grad_norm': '0', 'counters/examples': 9280, 'counters/updates': 290}
train stats after 9312 examples: {'rewards_train/chosen': '0.0041906', 'rewards_train/rejected': '-0.0032773', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.0074679', 'logps_train/rejected': '-278.66', 'logps_train/chosen': '-283.47', 'loss/train': '0.70426', 'examples_per_second': '48.906', 'grad_norm': '0', 'counters/examples': 9312, 'counters/updates': 291}
train stats after 9344 examples: {'rewards_train/chosen': '0.044899', 'rewards_train/rejected': '0.015591', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.029309', 'logps_train/rejected': '-309.58', 'logps_train/chosen': '-302.98', 'loss/train': '0.69599', 'examples_per_second': '69.311', 'grad_norm': '0', 'counters/examples': 9344, 'counters/updates': 292}
train stats after 9376 examples: {'rewards_train/chosen': '0.13241', 'rewards_train/rejected': '0.068663', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.063743', 'logps_train/rejected': '-301.09', 'logps_train/chosen': '-342.15', 'loss/train': '0.68697', 'examples_per_second': '49.146', 'grad_norm': '0', 'counters/examples': 9376, 'counters/updates': 293}
train stats after 9408 examples: {'rewards_train/chosen': '0.041634', 'rewards_train/rejected': '0.00096663', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.040667', 'logps_train/rejected': '-318.73', 'logps_train/chosen': '-329.29', 'loss/train': '0.69948', 'examples_per_second': '52.751', 'grad_norm': '0', 'counters/examples': 9408, 'counters/updates': 294}
train stats after 9440 examples: {'rewards_train/chosen': '0.088903', 'rewards_train/rejected': '0.028834', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.060068', 'logps_train/rejected': '-328.09', 'logps_train/chosen': '-343.63', 'loss/train': '0.68307', 'examples_per_second': '51.659', 'grad_norm': '0', 'counters/examples': 9440, 'counters/updates': 295}
train stats after 9472 examples: {'rewards_train/chosen': '0.093602', 'rewards_train/rejected': '-0.011203', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.10481', 'logps_train/rejected': '-320.9', 'logps_train/chosen': '-349.74', 'loss/train': '0.67248', 'examples_per_second': '60.281', 'grad_norm': '0', 'counters/examples': 9472, 'counters/updates': 296}
train stats after 9504 examples: {'rewards_train/chosen': '0.051354', 'rewards_train/rejected': '0.0092993', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.042054', 'logps_train/rejected': '-277.79', 'logps_train/chosen': '-300.71', 'loss/train': '0.69586', 'examples_per_second': '56.126', 'grad_norm': '0', 'counters/examples': 9504, 'counters/updates': 297}
train stats after 9536 examples: {'rewards_train/chosen': '0.077945', 'rewards_train/rejected': '0.022079', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.055865', 'logps_train/rejected': '-285.7', 'logps_train/chosen': '-309.04', 'loss/train': '0.6925', 'examples_per_second': '62.182', 'grad_norm': '0', 'counters/examples': 9536, 'counters/updates': 298}
train stats after 9568 examples: {'rewards_train/chosen': '0.096684', 'rewards_train/rejected': '0.0096418', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.087043', 'logps_train/rejected': '-270.34', 'logps_train/chosen': '-309.52', 'loss/train': '0.6718', 'examples_per_second': '45.773', 'grad_norm': '0', 'counters/examples': 9568, 'counters/updates': 299}
train stats after 9600 examples: {'rewards_train/chosen': '0.068435', 'rewards_train/rejected': '0.075765', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '-0.0073296', 'logps_train/rejected': '-303.56', 'logps_train/chosen': '-306.37', 'loss/train': '0.71946', 'examples_per_second': '53.138', 'grad_norm': '0', 'counters/examples': 9600, 'counters/updates': 300}
train stats after 9632 examples: {'rewards_train/chosen': '0.10412', 'rewards_train/rejected': '0.058536', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.045589', 'logps_train/rejected': '-297.28', 'logps_train/chosen': '-304.96', 'loss/train': '0.68645', 'examples_per_second': '46.966', 'grad_norm': '0', 'counters/examples': 9632, 'counters/updates': 301}
train stats after 9664 examples: {'rewards_train/chosen': '-0.00074798', 'rewards_train/rejected': '-0.042541', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.041793', 'logps_train/rejected': '-283.54', 'logps_train/chosen': '-307.64', 'loss/train': '0.69059', 'examples_per_second': '58.368', 'grad_norm': '0', 'counters/examples': 9664, 'counters/updates': 302}
train stats after 9696 examples: {'rewards_train/chosen': '0.074469', 'rewards_train/rejected': '0.036239', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.03823', 'logps_train/rejected': '-292.26', 'logps_train/chosen': '-353.26', 'loss/train': '0.69788', 'examples_per_second': '46.762', 'grad_norm': '0', 'counters/examples': 9696, 'counters/updates': 303}
train stats after 9728 examples: {'rewards_train/chosen': '0.096439', 'rewards_train/rejected': '0.035134', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.061305', 'logps_train/rejected': '-309.2', 'logps_train/chosen': '-324.45', 'loss/train': '0.68534', 'examples_per_second': '48.651', 'grad_norm': '0', 'counters/examples': 9728, 'counters/updates': 304}
train stats after 9760 examples: {'rewards_train/chosen': '0.053479', 'rewards_train/rejected': '0.0021867', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.051293', 'logps_train/rejected': '-278.84', 'logps_train/chosen': '-292.39', 'loss/train': '0.68884', 'examples_per_second': '60.272', 'grad_norm': '0', 'counters/examples': 9760, 'counters/updates': 305}
train stats after 9792 examples: {'rewards_train/chosen': '0.10814', 'rewards_train/rejected': '0.038458', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.069679', 'logps_train/rejected': '-287.52', 'logps_train/chosen': '-315.44', 'loss/train': '0.67628', 'examples_per_second': '55.013', 'grad_norm': '0', 'counters/examples': 9792, 'counters/updates': 306}
train stats after 9824 examples: {'rewards_train/chosen': '0.060802', 'rewards_train/rejected': '0.031321', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '0.029481', 'logps_train/rejected': '-284.85', 'logps_train/chosen': '-316.82', 'loss/train': '0.69714', 'examples_per_second': '65.733', 'grad_norm': '0', 'counters/examples': 9824, 'counters/updates': 307}
train stats after 9856 examples: {'rewards_train/chosen': '0.11306', 'rewards_train/rejected': '0.061155', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.051909', 'logps_train/rejected': '-324.49', 'logps_train/chosen': '-341.5', 'loss/train': '0.69115', 'examples_per_second': '56.762', 'grad_norm': '0', 'counters/examples': 9856, 'counters/updates': 308}
train stats after 9888 examples: {'rewards_train/chosen': '0.13362', 'rewards_train/rejected': '0.05088', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.082738', 'logps_train/rejected': '-268.37', 'logps_train/chosen': '-304.33', 'loss/train': '0.67607', 'examples_per_second': '72.698', 'grad_norm': '0', 'counters/examples': 9888, 'counters/updates': 309}
train stats after 9920 examples: {'rewards_train/chosen': '0.091614', 'rewards_train/rejected': '0.037047', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.054567', 'logps_train/rejected': '-287.71', 'logps_train/chosen': '-330.55', 'loss/train': '0.68846', 'examples_per_second': '63.848', 'grad_norm': '0', 'counters/examples': 9920, 'counters/updates': 310}
==================================================
==================================================
Running evaluation after 9920 train examples
==================================================
==================================================
train stats after 9952 examples: {'rewards_train/chosen': '0.067903', 'rewards_train/rejected': '0.025785', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.042117', 'logps_train/rejected': '-336.46', 'logps_train/chosen': '-324.15', 'loss/train': '0.69227', 'examples_per_second': '45.456', 'grad_norm': '0', 'counters/examples': 9952, 'counters/updates': 311}
train stats after 9984 examples: {'rewards_train/chosen': '0.046566', 'rewards_train/rejected': '0.056039', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '-0.0094736', 'logps_train/rejected': '-297.21', 'logps_train/chosen': '-308.43', 'loss/train': '0.71697', 'examples_per_second': '46.38', 'grad_norm': '0', 'counters/examples': 9984, 'counters/updates': 312}
eval after 9984: {'rewards_eval/chosen': '0.11925', 'rewards_eval/rejected': '0.12524', 'rewards_eval/accuracies': '0.46484', 'rewards_eval/margins': '-0.0059862', 'logps_eval/rejected': '-345.5', 'logps_eval/chosen': '-354.93', 'loss/eval': '0.72009'}
creating checkpoint to write to .cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_899398/step-9984...
eval after 9984: {'rewards_eval/chosen': '0.11925', 'rewards_eval/rejected': '0.12524', 'rewards_eval/accuracies': '0.46484', 'rewards_eval/margins': '-0.0059862', 'logps_eval/rejected': '-345.5', 'logps_eval/chosen': '-354.93', 'loss/eval': '0.72009'}
creating checkpoint to write to .cache/emiliano.penaloza/hnet_dpo_2024-07-10_16-17-21_899398/step-9984...
train stats after 10016 examples: {'rewards_train/chosen': '0.080553', 'rewards_train/rejected': '0.059113', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.02144', 'logps_train/rejected': '-292.73', 'logps_train/chosen': '-314', 'loss/train': '0.70107', 'examples_per_second': '56.93', 'grad_norm': '0', 'counters/examples': 10016, 'counters/updates': 313}
train stats after 10048 examples: {'rewards_train/chosen': '0.10596', 'rewards_train/rejected': '-0.039451', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.14541', 'logps_train/rejected': '-304.48', 'logps_train/chosen': '-325.33', 'loss/train': '0.64575', 'examples_per_second': '48.786', 'grad_norm': '0', 'counters/examples': 10048, 'counters/updates': 314}
train stats after 10080 examples: {'rewards_train/chosen': '0.099946', 'rewards_train/rejected': '-0.0021546', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.1021', 'logps_train/rejected': '-277.4', 'logps_train/chosen': '-308.66', 'loss/train': '0.66432', 'examples_per_second': '71.039', 'grad_norm': '0', 'counters/examples': 10080, 'counters/updates': 315}
train stats after 10112 examples: {'rewards_train/chosen': '0.099541', 'rewards_train/rejected': '-0.0029842', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.10252', 'logps_train/rejected': '-300.58', 'logps_train/chosen': '-316.78', 'loss/train': '0.66247', 'examples_per_second': '45.811', 'grad_norm': '0', 'counters/examples': 10112, 'counters/updates': 316}
train stats after 10144 examples: {'rewards_train/chosen': '0.11185', 'rewards_train/rejected': '0.0072223', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.10463', 'logps_train/rejected': '-300.19', 'logps_train/chosen': '-321.83', 'loss/train': '0.66367', 'examples_per_second': '53.541', 'grad_norm': '0', 'counters/examples': 10144, 'counters/updates': 317}
train stats after 10176 examples: {'rewards_train/chosen': '0.092205', 'rewards_train/rejected': '0.029161', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.063044', 'logps_train/rejected': '-281.39', 'logps_train/chosen': '-293.41', 'loss/train': '0.68219', 'examples_per_second': '66.275', 'grad_norm': '0', 'counters/examples': 10176, 'counters/updates': 318}
train stats after 10208 examples: {'rewards_train/chosen': '0.042297', 'rewards_train/rejected': '0.071832', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '-0.029534', 'logps_train/rejected': '-257.39', 'logps_train/chosen': '-290.86', 'loss/train': '0.72795', 'examples_per_second': '50.82', 'grad_norm': '0', 'counters/examples': 10208, 'counters/updates': 319}
train stats after 10240 examples: {'rewards_train/chosen': '0.10561', 'rewards_train/rejected': '0.055697', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.049913', 'logps_train/rejected': '-297.4', 'logps_train/chosen': '-312.74', 'loss/train': '0.6841', 'examples_per_second': '46.595', 'grad_norm': '0', 'counters/examples': 10240, 'counters/updates': 320}
train stats after 10272 examples: {'rewards_train/chosen': '0.15188', 'rewards_train/rejected': '0.10462', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.047259', 'logps_train/rejected': '-311.85', 'logps_train/chosen': '-341.16', 'loss/train': '0.69021', 'examples_per_second': '56.131', 'grad_norm': '0', 'counters/examples': 10272, 'counters/updates': 321}
train stats after 10304 examples: {'rewards_train/chosen': '0.075276', 'rewards_train/rejected': '0.044946', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.03033', 'logps_train/rejected': '-281.41', 'logps_train/chosen': '-318.76', 'loss/train': '0.69622', 'examples_per_second': '50.518', 'grad_norm': '0', 'counters/examples': 10304, 'counters/updates': 322}
train stats after 10336 examples: {'rewards_train/chosen': '0.070118', 'rewards_train/rejected': '0.0058049', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.064313', 'logps_train/rejected': '-301.12', 'logps_train/chosen': '-302.76', 'loss/train': '0.67989', 'examples_per_second': '49.541', 'grad_norm': '0', 'counters/examples': 10336, 'counters/updates': 323}
train stats after 10368 examples: {'rewards_train/chosen': '0.097661', 'rewards_train/rejected': '7.777e-05', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.097583', 'logps_train/rejected': '-261.62', 'logps_train/chosen': '-306.54', 'loss/train': '0.66876', 'examples_per_second': '70.935', 'grad_norm': '0', 'counters/examples': 10368, 'counters/updates': 324}
train stats after 10400 examples: {'rewards_train/chosen': '0.093613', 'rewards_train/rejected': '0.063844', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.029769', 'logps_train/rejected': '-269.88', 'logps_train/chosen': '-299.78', 'loss/train': '0.70139', 'examples_per_second': '61.849', 'grad_norm': '0', 'counters/examples': 10400, 'counters/updates': 325}
train stats after 10432 examples: {'rewards_train/chosen': '0.16264', 'rewards_train/rejected': '0.055735', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.1069', 'logps_train/rejected': '-307.77', 'logps_train/chosen': '-326.1', 'loss/train': '0.66526', 'examples_per_second': '66.162', 'grad_norm': '0', 'counters/examples': 10432, 'counters/updates': 326}
train stats after 10464 examples: {'rewards_train/chosen': '0.15283', 'rewards_train/rejected': '0.030808', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.12203', 'logps_train/rejected': '-291.91', 'logps_train/chosen': '-321.8', 'loss/train': '0.65587', 'examples_per_second': '45.439', 'grad_norm': '0', 'counters/examples': 10464, 'counters/updates': 327}
train stats after 10496 examples: {'rewards_train/chosen': '0.17515', 'rewards_train/rejected': '0.097914', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.077237', 'logps_train/rejected': '-287.98', 'logps_train/chosen': '-307.62', 'loss/train': '0.67719', 'examples_per_second': '47.523', 'grad_norm': '0', 'counters/examples': 10496, 'counters/updates': 328}
train stats after 10528 examples: {'rewards_train/chosen': '0.04962', 'rewards_train/rejected': '0.048903', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.00071641', 'logps_train/rejected': '-288.91', 'logps_train/chosen': '-301.8', 'loss/train': '0.71061', 'examples_per_second': '55.593', 'grad_norm': '0', 'counters/examples': 10528, 'counters/updates': 329}
train stats after 10560 examples: {'rewards_train/chosen': '0.16746', 'rewards_train/rejected': '0.094508', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.072955', 'logps_train/rejected': '-307.11', 'logps_train/chosen': '-322.12', 'loss/train': '0.6737', 'examples_per_second': '63.991', 'grad_norm': '0', 'counters/examples': 10560, 'counters/updates': 330}
train stats after 10592 examples: {'rewards_train/chosen': '0.048722', 'rewards_train/rejected': '0.067942', 'rewards_train/accuracies': '0.49219', 'rewards_train/margins': '-0.01922', 'logps_train/rejected': '-306.8', 'logps_train/chosen': '-333.99', 'loss/train': '0.72265', 'examples_per_second': '66.291', 'grad_norm': '0', 'counters/examples': 10592, 'counters/updates': 331}
train stats after 10624 examples: {'rewards_train/chosen': '0.12854', 'rewards_train/rejected': '0.029504', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.099035', 'logps_train/rejected': '-286.08', 'logps_train/chosen': '-321.05', 'loss/train': '0.6709', 'examples_per_second': '63.065', 'grad_norm': '0', 'counters/examples': 10624, 'counters/updates': 332}
train stats after 10656 examples: {'rewards_train/chosen': '0.095405', 'rewards_train/rejected': '-0.0060504', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.10145', 'logps_train/rejected': '-273.8', 'logps_train/chosen': '-318.63', 'loss/train': '0.66154', 'examples_per_second': '67.862', 'grad_norm': '0', 'counters/examples': 10656, 'counters/updates': 333}
train stats after 10688 examples: {'rewards_train/chosen': '0.12237', 'rewards_train/rejected': '0.039826', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.082543', 'logps_train/rejected': '-292.81', 'logps_train/chosen': '-310.28', 'loss/train': '0.67141', 'examples_per_second': '57.236', 'grad_norm': '0', 'counters/examples': 10688, 'counters/updates': 334}
train stats after 10720 examples: {'rewards_train/chosen': '0.084033', 'rewards_train/rejected': '0.069426', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.014607', 'logps_train/rejected': '-310.26', 'logps_train/chosen': '-336.63', 'loss/train': '0.70597', 'examples_per_second': '73.105', 'grad_norm': '0', 'counters/examples': 10720, 'counters/updates': 335}
train stats after 10752 examples: {'rewards_train/chosen': '0.22935', 'rewards_train/rejected': '0.058484', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.17087', 'logps_train/rejected': '-302.58', 'logps_train/chosen': '-336.53', 'loss/train': '0.632', 'examples_per_second': '64.615', 'grad_norm': '0', 'counters/examples': 10752, 'counters/updates': 336}
train stats after 10784 examples: {'rewards_train/chosen': '0.14893', 'rewards_train/rejected': '0.11069', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.038235', 'logps_train/rejected': '-310.7', 'logps_train/chosen': '-321.95', 'loss/train': '0.69221', 'examples_per_second': '68.65', 'grad_norm': '0', 'counters/examples': 10784, 'counters/updates': 337}
train stats after 10816 examples: {'rewards_train/chosen': '0.098993', 'rewards_train/rejected': '0.038344', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.06065', 'logps_train/rejected': '-276.48', 'logps_train/chosen': '-325.91', 'loss/train': '0.68781', 'examples_per_second': '49.074', 'grad_norm': '0', 'counters/examples': 10816, 'counters/updates': 338}
train stats after 10848 examples: {'rewards_train/chosen': '0.12334', 'rewards_train/rejected': '0.057798', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.065547', 'logps_train/rejected': '-280.27', 'logps_train/chosen': '-307.11', 'loss/train': '0.68865', 'examples_per_second': '72.156', 'grad_norm': '0', 'counters/examples': 10848, 'counters/updates': 339}
train stats after 10880 examples: {'rewards_train/chosen': '0.12023', 'rewards_train/rejected': '0.075217', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.045012', 'logps_train/rejected': '-300.67', 'logps_train/chosen': '-317.02', 'loss/train': '0.69433', 'examples_per_second': '50.776', 'grad_norm': '0', 'counters/examples': 10880, 'counters/updates': 340}
train stats after 10912 examples: {'rewards_train/chosen': '0.10573', 'rewards_train/rejected': '0.095575', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.010154', 'logps_train/rejected': '-288.74', 'logps_train/chosen': '-303.66', 'loss/train': '0.71363', 'examples_per_second': '65.557', 'grad_norm': '0', 'counters/examples': 10912, 'counters/updates': 341}
train stats after 10944 examples: {'rewards_train/chosen': '0.10207', 'rewards_train/rejected': '-0.013656', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.11573', 'logps_train/rejected': '-264.45', 'logps_train/chosen': '-312.26', 'loss/train': '0.66166', 'examples_per_second': '53.811', 'grad_norm': '0', 'counters/examples': 10944, 'counters/updates': 342}
train stats after 10976 examples: {'rewards_train/chosen': '0.12665', 'rewards_train/rejected': '0.12957', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '-0.0029202', 'logps_train/rejected': '-311.92', 'logps_train/chosen': '-332', 'loss/train': '0.71897', 'examples_per_second': '62.095', 'grad_norm': '0', 'counters/examples': 10976, 'counters/updates': 343}
train stats after 11008 examples: {'rewards_train/chosen': '0.17348', 'rewards_train/rejected': '0.096424', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.077054', 'logps_train/rejected': '-312.12', 'logps_train/chosen': '-331.97', 'loss/train': '0.67724', 'examples_per_second': '47.366', 'grad_norm': '0', 'counters/examples': 11008, 'counters/updates': 344}
train stats after 11040 examples: {'rewards_train/chosen': '0.14641', 'rewards_train/rejected': '0.031512', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.11489', 'logps_train/rejected': '-276.25', 'logps_train/chosen': '-295.04', 'loss/train': '0.6597', 'examples_per_second': '61.422', 'grad_norm': '0', 'counters/examples': 11040, 'counters/updates': 345}
train stats after 11072 examples: {'rewards_train/chosen': '0.16858', 'rewards_train/rejected': '0.13644', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.03214', 'logps_train/rejected': '-323.39', 'logps_train/chosen': '-353.78', 'loss/train': '0.706', 'examples_per_second': '50.104', 'grad_norm': '0', 'counters/examples': 11072, 'counters/updates': 346}
train stats after 11104 examples: {'rewards_train/chosen': '0.078429', 'rewards_train/rejected': '0.096359', 'rewards_train/accuracies': '0.45312', 'rewards_train/margins': '-0.01793', 'logps_train/rejected': '-284.73', 'logps_train/chosen': '-325.33', 'loss/train': '0.72437', 'examples_per_second': '53.712', 'grad_norm': '0', 'counters/examples': 11104, 'counters/updates': 347}
train stats after 11136 examples: {'rewards_train/chosen': '0.1778', 'rewards_train/rejected': '0.050898', 'rewards_train/accuracies': '0.63281', 'rewards_train/margins': '0.12691', 'logps_train/rejected': '-294.99', 'logps_train/chosen': '-329.59', 'loss/train': '0.64906', 'examples_per_second': '60.377', 'grad_norm': '0', 'counters/examples': 11136, 'counters/updates': 348}
train stats after 11168 examples: {'rewards_train/chosen': '0.14886', 'rewards_train/rejected': '0.069413', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.079448', 'logps_train/rejected': '-280.49', 'logps_train/chosen': '-303.87', 'loss/train': '0.67471', 'examples_per_second': '64.282', 'grad_norm': '0', 'counters/examples': 11168, 'counters/updates': 349}
train stats after 11200 examples: {'rewards_train/chosen': '0.10902', 'rewards_train/rejected': '0.041876', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.067147', 'logps_train/rejected': '-267.65', 'logps_train/chosen': '-300.58', 'loss/train': '0.67879', 'examples_per_second': '52.651', 'grad_norm': '0', 'counters/examples': 11200, 'counters/updates': 350}
train stats after 11232 examples: {'rewards_train/chosen': '0.13485', 'rewards_train/rejected': '0.034246', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.1006', 'logps_train/rejected': '-314.83', 'logps_train/chosen': '-345.54', 'loss/train': '0.6687', 'examples_per_second': '66.794', 'grad_norm': '0', 'counters/examples': 11232, 'counters/updates': 351}
train stats after 11264 examples: {'rewards_train/chosen': '0.10822', 'rewards_train/rejected': '0.095761', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.012461', 'logps_train/rejected': '-302.65', 'logps_train/chosen': '-329.66', 'loss/train': '0.7071', 'examples_per_second': '48.534', 'grad_norm': '0', 'counters/examples': 11264, 'counters/updates': 352}
train stats after 11296 examples: {'rewards_train/chosen': '0.17477', 'rewards_train/rejected': '0.074823', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.099943', 'logps_train/rejected': '-315.14', 'logps_train/chosen': '-340.84', 'loss/train': '0.67002', 'examples_per_second': '63.167', 'grad_norm': '0', 'counters/examples': 11296, 'counters/updates': 353}
train stats after 11328 examples: {'rewards_train/chosen': '0.11474', 'rewards_train/rejected': '0.041665', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.07308', 'logps_train/rejected': '-295.63', 'logps_train/chosen': '-297.95', 'loss/train': '0.67816', 'examples_per_second': '58.739', 'grad_norm': '0', 'counters/examples': 11328, 'counters/updates': 354}
train stats after 11360 examples: {'rewards_train/chosen': '0.1388', 'rewards_train/rejected': '0.025967', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.11284', 'logps_train/rejected': '-281.25', 'logps_train/chosen': '-318.04', 'loss/train': '0.66623', 'examples_per_second': '47.195', 'grad_norm': '0', 'counters/examples': 11360, 'counters/updates': 355}
train stats after 11392 examples: {'rewards_train/chosen': '0.10601', 'rewards_train/rejected': '0.042341', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.063671', 'logps_train/rejected': '-285.57', 'logps_train/chosen': '-310.49', 'loss/train': '0.68292', 'examples_per_second': '66.895', 'grad_norm': '0', 'counters/examples': 11392, 'counters/updates': 356}
train stats after 11424 examples: {'rewards_train/chosen': '0.10231', 'rewards_train/rejected': '0.0249', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.077406', 'logps_train/rejected': '-314.49', 'logps_train/chosen': '-298.32', 'loss/train': '0.67928', 'examples_per_second': '68.314', 'grad_norm': '0', 'counters/examples': 11424, 'counters/updates': 357}
train stats after 11456 examples: {'rewards_train/chosen': '0.12988', 'rewards_train/rejected': '0.058124', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.071759', 'logps_train/rejected': '-302.15', 'logps_train/chosen': '-330.19', 'loss/train': '0.67891', 'examples_per_second': '53.922', 'grad_norm': '0', 'counters/examples': 11456, 'counters/updates': 358}
train stats after 11488 examples: {'rewards_train/chosen': '0.20786', 'rewards_train/rejected': '0.081481', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.12638', 'logps_train/rejected': '-284.05', 'logps_train/chosen': '-296.86', 'loss/train': '0.65486', 'examples_per_second': '48.064', 'grad_norm': '0', 'counters/examples': 11488, 'counters/updates': 359}
train stats after 11520 examples: {'rewards_train/chosen': '0.093077', 'rewards_train/rejected': '0.031914', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.061163', 'logps_train/rejected': '-309.58', 'logps_train/chosen': '-325.7', 'loss/train': '0.69349', 'examples_per_second': '51.426', 'grad_norm': '0', 'counters/examples': 11520, 'counters/updates': 360}
train stats after 11552 examples: {'rewards_train/chosen': '0.11908', 'rewards_train/rejected': '0.011666', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.10741', 'logps_train/rejected': '-346.08', 'logps_train/chosen': '-341.23', 'loss/train': '0.66763', 'examples_per_second': '59.06', 'grad_norm': '0', 'counters/examples': 11552, 'counters/updates': 361}
train stats after 11584 examples: {'rewards_train/chosen': '0.10015', 'rewards_train/rejected': '0.01683', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.083315', 'logps_train/rejected': '-299.53', 'logps_train/chosen': '-328.03', 'loss/train': '0.67619', 'examples_per_second': '48.629', 'grad_norm': '0', 'counters/examples': 11584, 'counters/updates': 362}
train stats after 11616 examples: {'rewards_train/chosen': '0.090739', 'rewards_train/rejected': '0.044779', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.04596', 'logps_train/rejected': '-261.68', 'logps_train/chosen': '-286.44', 'loss/train': '0.69404', 'examples_per_second': '65.07', 'grad_norm': '0', 'counters/examples': 11616, 'counters/updates': 363}
train stats after 11648 examples: {'rewards_train/chosen': '0.095324', 'rewards_train/rejected': '-0.01319', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.10851', 'logps_train/rejected': '-311.89', 'logps_train/chosen': '-347', 'loss/train': '0.66427', 'examples_per_second': '56.79', 'grad_norm': '0', 'counters/examples': 11648, 'counters/updates': 364}
train stats after 11680 examples: {'rewards_train/chosen': '0.057503', 'rewards_train/rejected': '0.0051802', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.052323', 'logps_train/rejected': '-267.38', 'logps_train/chosen': '-292.73', 'loss/train': '0.69462', 'examples_per_second': '62.05', 'grad_norm': '0', 'counters/examples': 11680, 'counters/updates': 365}
train stats after 11712 examples: {'rewards_train/chosen': '0.12717', 'rewards_train/rejected': '0.039324', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.087842', 'logps_train/rejected': '-306.99', 'logps_train/chosen': '-312.72', 'loss/train': '0.67865', 'examples_per_second': '52.549', 'grad_norm': '0', 'counters/examples': 11712, 'counters/updates': 366}
train stats after 11744 examples: {'rewards_train/chosen': '0.16877', 'rewards_train/rejected': '0.002675', 'rewards_train/accuracies': '0.67969', 'rewards_train/margins': '0.16609', 'logps_train/rejected': '-288.29', 'logps_train/chosen': '-318.95', 'loss/train': '0.63622', 'examples_per_second': '49.075', 'grad_norm': '0', 'counters/examples': 11744, 'counters/updates': 367}
train stats after 11776 examples: {'rewards_train/chosen': '0.073554', 'rewards_train/rejected': '-0.00078928', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.074344', 'logps_train/rejected': '-288.67', 'logps_train/chosen': '-305.41', 'loss/train': '0.68442', 'examples_per_second': '49.818', 'grad_norm': '0', 'counters/examples': 11776, 'counters/updates': 368}
train stats after 11808 examples: {'rewards_train/chosen': '0.056223', 'rewards_train/rejected': '0.013014', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.043209', 'logps_train/rejected': '-294', 'logps_train/chosen': '-323.25', 'loss/train': '0.7038', 'examples_per_second': '49.403', 'grad_norm': '0', 'counters/examples': 11808, 'counters/updates': 369}
train stats after 11840 examples: {'rewards_train/chosen': '0.1394', 'rewards_train/rejected': '0.11108', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.028322', 'logps_train/rejected': '-287.88', 'logps_train/chosen': '-313.43', 'loss/train': '0.70842', 'examples_per_second': '51.576', 'grad_norm': '0', 'counters/examples': 11840, 'counters/updates': 370}
train stats after 11872 examples: {'rewards_train/chosen': '0.07864', 'rewards_train/rejected': '0.016053', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.062587', 'logps_train/rejected': '-289.61', 'logps_train/chosen': '-309.27', 'loss/train': '0.68854', 'examples_per_second': '48.491', 'grad_norm': '0', 'counters/examples': 11872, 'counters/updates': 371}
train stats after 11904 examples: {'rewards_train/chosen': '0.11474', 'rewards_train/rejected': '0.052479', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.062263', 'logps_train/rejected': '-315.28', 'logps_train/chosen': '-333.74', 'loss/train': '0.68678', 'examples_per_second': '54.072', 'grad_norm': '0', 'counters/examples': 11904, 'counters/updates': 372}
train stats after 11936 examples: {'rewards_train/chosen': '0.060167', 'rewards_train/rejected': '-0.014788', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.074955', 'logps_train/rejected': '-277.77', 'logps_train/chosen': '-293.73', 'loss/train': '0.67613', 'examples_per_second': '54.206', 'grad_norm': '0', 'counters/examples': 11936, 'counters/updates': 373}
train stats after 11968 examples: {'rewards_train/chosen': '0.06906', 'rewards_train/rejected': '0.12053', 'rewards_train/accuracies': '0.47656', 'rewards_train/margins': '-0.051474', 'logps_train/rejected': '-290.81', 'logps_train/chosen': '-332.22', 'loss/train': '0.74385', 'examples_per_second': '47.458', 'grad_norm': '0', 'counters/examples': 11968, 'counters/updates': 374}
train stats after 12000 examples: {'rewards_train/chosen': '0.09154', 'rewards_train/rejected': '-0.0081208', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.099661', 'logps_train/rejected': '-301.46', 'logps_train/chosen': '-326.29', 'loss/train': '0.67652', 'examples_per_second': '44.531', 'grad_norm': '0', 'counters/examples': 12000, 'counters/updates': 375}
train stats after 12032 examples: {'rewards_train/chosen': '0.071271', 'rewards_train/rejected': '-0.029807', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.10108', 'logps_train/rejected': '-306.48', 'logps_train/chosen': '-322.28', 'loss/train': '0.66909', 'examples_per_second': '49.166', 'grad_norm': '0', 'counters/examples': 12032, 'counters/updates': 376}
train stats after 12064 examples: {'rewards_train/chosen': '0.13172', 'rewards_train/rejected': '0.035061', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.096656', 'logps_train/rejected': '-316.85', 'logps_train/chosen': '-348.04', 'loss/train': '0.67201', 'examples_per_second': '52.979', 'grad_norm': '0', 'counters/examples': 12064, 'counters/updates': 377}
train stats after 12096 examples: {'rewards_train/chosen': '0.10286', 'rewards_train/rejected': '0.077355', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.025508', 'logps_train/rejected': '-293.38', 'logps_train/chosen': '-329.3', 'loss/train': '0.71481', 'examples_per_second': '51.536', 'grad_norm': '0', 'counters/examples': 12096, 'counters/updates': 378}
train stats after 12128 examples: {'rewards_train/chosen': '0.1342', 'rewards_train/rejected': '-0.01474', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.14894', 'logps_train/rejected': '-319.39', 'logps_train/chosen': '-325.32', 'loss/train': '0.66275', 'examples_per_second': '54.75', 'grad_norm': '0', 'counters/examples': 12128, 'counters/updates': 379}
train stats after 12160 examples: {'rewards_train/chosen': '0.12904', 'rewards_train/rejected': '0.049305', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.079735', 'logps_train/rejected': '-298.83', 'logps_train/chosen': '-326.05', 'loss/train': '0.67962', 'examples_per_second': '60.629', 'grad_norm': '0', 'counters/examples': 12160, 'counters/updates': 380}
train stats after 12192 examples: {'rewards_train/chosen': '0.097085', 'rewards_train/rejected': '0.055188', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.041897', 'logps_train/rejected': '-315.48', 'logps_train/chosen': '-314.52', 'loss/train': '0.69752', 'examples_per_second': '49.876', 'grad_norm': '0', 'counters/examples': 12192, 'counters/updates': 381}
train stats after 12224 examples: {'rewards_train/chosen': '0.091757', 'rewards_train/rejected': '-0.020953', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.11271', 'logps_train/rejected': '-288.14', 'logps_train/chosen': '-276.6', 'loss/train': '0.66243', 'examples_per_second': '74.201', 'grad_norm': '0', 'counters/examples': 12224, 'counters/updates': 382}
train stats after 12256 examples: {'rewards_train/chosen': '0.1153', 'rewards_train/rejected': '-0.020884', 'rewards_train/accuracies': '0.64062', 'rewards_train/margins': '0.13618', 'logps_train/rejected': '-272.79', 'logps_train/chosen': '-301.77', 'loss/train': '0.64445', 'examples_per_second': '66.468', 'grad_norm': '0', 'counters/examples': 12256, 'counters/updates': 383}
train stats after 12288 examples: {'rewards_train/chosen': '0.086274', 'rewards_train/rejected': '0.084443', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.0018308', 'logps_train/rejected': '-268.92', 'logps_train/chosen': '-289.04', 'loss/train': '0.71635', 'examples_per_second': '63.592', 'grad_norm': '0', 'counters/examples': 12288, 'counters/updates': 384}
train stats after 12320 examples: {'rewards_train/chosen': '0.13148', 'rewards_train/rejected': '0.066654', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.064823', 'logps_train/rejected': '-292.76', 'logps_train/chosen': '-308.37', 'loss/train': '0.68605', 'examples_per_second': '72.311', 'grad_norm': '0', 'counters/examples': 12320, 'counters/updates': 385}
train stats after 12352 examples: {'rewards_train/chosen': '0.039133', 'rewards_train/rejected': '0.023858', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.015275', 'logps_train/rejected': '-277.35', 'logps_train/chosen': '-297.83', 'loss/train': '0.70719', 'examples_per_second': '51.738', 'grad_norm': '0', 'counters/examples': 12352, 'counters/updates': 386}
train stats after 12384 examples: {'rewards_train/chosen': '0.10809', 'rewards_train/rejected': '0.075298', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.032788', 'logps_train/rejected': '-304.21', 'logps_train/chosen': '-324.09', 'loss/train': '0.69748', 'examples_per_second': '46.568', 'grad_norm': '0', 'counters/examples': 12384, 'counters/updates': 387}
train stats after 12416 examples: {'rewards_train/chosen': '0.086712', 'rewards_train/rejected': '0.020825', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.065887', 'logps_train/rejected': '-296.03', 'logps_train/chosen': '-331.82', 'loss/train': '0.68751', 'examples_per_second': '46.869', 'grad_norm': '0', 'counters/examples': 12416, 'counters/updates': 388}
train stats after 12448 examples: {'rewards_train/chosen': '0.11953', 'rewards_train/rejected': '0.081913', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.037613', 'logps_train/rejected': '-315.8', 'logps_train/chosen': '-344.42', 'loss/train': '0.69458', 'examples_per_second': '47.31', 'grad_norm': '0', 'counters/examples': 12448, 'counters/updates': 389}
train stats after 12480 examples: {'rewards_train/chosen': '0.091521', 'rewards_train/rejected': '0.0058862', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.085635', 'logps_train/rejected': '-291.13', 'logps_train/chosen': '-314.57', 'loss/train': '0.66952', 'examples_per_second': '65.959', 'grad_norm': '0', 'counters/examples': 12480, 'counters/updates': 390}
train stats after 12512 examples: {'rewards_train/chosen': '0.099604', 'rewards_train/rejected': '0.028144', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.07146', 'logps_train/rejected': '-316.25', 'logps_train/chosen': '-356.53', 'loss/train': '0.68861', 'examples_per_second': '49.446', 'grad_norm': '0', 'counters/examples': 12512, 'counters/updates': 391}
train stats after 12544 examples: {'rewards_train/chosen': '0.10995', 'rewards_train/rejected': '0.087194', 'rewards_train/accuracies': '0.48438', 'rewards_train/margins': '0.022754', 'logps_train/rejected': '-320.17', 'logps_train/chosen': '-318.59', 'loss/train': '0.7111', 'examples_per_second': '68.564', 'grad_norm': '0', 'counters/examples': 12544, 'counters/updates': 392}
train stats after 12576 examples: {'rewards_train/chosen': '0.074093', 'rewards_train/rejected': '0.041442', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.032651', 'logps_train/rejected': '-272.32', 'logps_train/chosen': '-297.3', 'loss/train': '0.69759', 'examples_per_second': '64.593', 'grad_norm': '0', 'counters/examples': 12576, 'counters/updates': 393}
train stats after 12608 examples: {'rewards_train/chosen': '0.047748', 'rewards_train/rejected': '-0.0019779', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.049726', 'logps_train/rejected': '-277.29', 'logps_train/chosen': '-315.68', 'loss/train': '0.6856', 'examples_per_second': '45.821', 'grad_norm': '0', 'counters/examples': 12608, 'counters/updates': 394}
train stats after 12640 examples: {'rewards_train/chosen': '0.10625', 'rewards_train/rejected': '-0.04471', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.15096', 'logps_train/rejected': '-318.55', 'logps_train/chosen': '-325.36', 'loss/train': '0.64745', 'examples_per_second': '45.992', 'grad_norm': '0', 'counters/examples': 12640, 'counters/updates': 395}
train stats after 12672 examples: {'rewards_train/chosen': '0.095548', 'rewards_train/rejected': '0.0037465', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.091801', 'logps_train/rejected': '-298.99', 'logps_train/chosen': '-317.01', 'loss/train': '0.67099', 'examples_per_second': '65.191', 'grad_norm': '0', 'counters/examples': 12672, 'counters/updates': 396}
train stats after 12704 examples: {'rewards_train/chosen': '0.029364', 'rewards_train/rejected': '-0.023276', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.05264', 'logps_train/rejected': '-275.77', 'logps_train/chosen': '-316.72', 'loss/train': '0.69441', 'examples_per_second': '52.776', 'grad_norm': '0', 'counters/examples': 12704, 'counters/updates': 397}
train stats after 12736 examples: {'rewards_train/chosen': '0.062404', 'rewards_train/rejected': '0.0041426', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.058261', 'logps_train/rejected': '-296.58', 'logps_train/chosen': '-302.22', 'loss/train': '0.69155', 'examples_per_second': '66.277', 'grad_norm': '0', 'counters/examples': 12736, 'counters/updates': 398}
train stats after 12768 examples: {'rewards_train/chosen': '0.11144', 'rewards_train/rejected': '0.027873', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.083571', 'logps_train/rejected': '-313.33', 'logps_train/chosen': '-356.55', 'loss/train': '0.67898', 'examples_per_second': '54.75', 'grad_norm': '0', 'counters/examples': 12768, 'counters/updates': 399}
train stats after 12800 examples: {'rewards_train/chosen': '0.072264', 'rewards_train/rejected': '0.0021374', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.070127', 'logps_train/rejected': '-320.18', 'logps_train/chosen': '-343.85', 'loss/train': '0.69164', 'examples_per_second': '60.195', 'grad_norm': '0', 'counters/examples': 12800, 'counters/updates': 400}
train stats after 12832 examples: {'rewards_train/chosen': '0.11218', 'rewards_train/rejected': '0.057078', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.055103', 'logps_train/rejected': '-304.79', 'logps_train/chosen': '-325.07', 'loss/train': '0.69765', 'examples_per_second': '62.827', 'grad_norm': '0', 'counters/examples': 12832, 'counters/updates': 401}
train stats after 12864 examples: {'rewards_train/chosen': '0.069403', 'rewards_train/rejected': '0.0061199', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.063284', 'logps_train/rejected': '-266.7', 'logps_train/chosen': '-298.96', 'loss/train': '0.6805', 'examples_per_second': '57.76', 'grad_norm': '0', 'counters/examples': 12864, 'counters/updates': 402}
train stats after 12896 examples: {'rewards_train/chosen': '0.11233', 'rewards_train/rejected': '0.035401', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.076929', 'logps_train/rejected': '-293.99', 'logps_train/chosen': '-309.77', 'loss/train': '0.68469', 'examples_per_second': '55.719', 'grad_norm': '0', 'counters/examples': 12896, 'counters/updates': 403}
train stats after 12928 examples: {'rewards_train/chosen': '0.12498', 'rewards_train/rejected': '0.017742', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.10724', 'logps_train/rejected': '-315.91', 'logps_train/chosen': '-328.53', 'loss/train': '0.66634', 'examples_per_second': '61.458', 'grad_norm': '0', 'counters/examples': 12928, 'counters/updates': 404}
train stats after 12960 examples: {'rewards_train/chosen': '0.12454', 'rewards_train/rejected': '0.065115', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.059423', 'logps_train/rejected': '-288.84', 'logps_train/chosen': '-311.15', 'loss/train': '0.69206', 'examples_per_second': '60.497', 'grad_norm': '0', 'counters/examples': 12960, 'counters/updates': 405}
train stats after 12992 examples: {'rewards_train/chosen': '0.034032', 'rewards_train/rejected': '0.028147', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.0058845', 'logps_train/rejected': '-270.38', 'logps_train/chosen': '-301.09', 'loss/train': '0.71101', 'examples_per_second': '46.226', 'grad_norm': '0', 'counters/examples': 12992, 'counters/updates': 406}
train stats after 13024 examples: {'rewards_train/chosen': '0.082433', 'rewards_train/rejected': '0.010248', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.072185', 'logps_train/rejected': '-286.24', 'logps_train/chosen': '-311.56', 'loss/train': '0.6829', 'examples_per_second': '45.016', 'grad_norm': '0', 'counters/examples': 13024, 'counters/updates': 407}
train stats after 13056 examples: {'rewards_train/chosen': '0.14565', 'rewards_train/rejected': '0.0035206', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.14213', 'logps_train/rejected': '-290.19', 'logps_train/chosen': '-307.34', 'loss/train': '0.64807', 'examples_per_second': '46.878', 'grad_norm': '0', 'counters/examples': 13056, 'counters/updates': 408}
train stats after 13088 examples: {'rewards_train/chosen': '0.1479', 'rewards_train/rejected': '0.056273', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.091627', 'logps_train/rejected': '-298.73', 'logps_train/chosen': '-329.59', 'loss/train': '0.67921', 'examples_per_second': '46.673', 'grad_norm': '0', 'counters/examples': 13088, 'counters/updates': 409}
train stats after 13120 examples: {'rewards_train/chosen': '0.13783', 'rewards_train/rejected': '0.039139', 'rewards_train/accuracies': '0.625', 'rewards_train/margins': '0.098694', 'logps_train/rejected': '-310.07', 'logps_train/chosen': '-312.15', 'loss/train': '0.66314', 'examples_per_second': '51.398', 'grad_norm': '0', 'counters/examples': 13120, 'counters/updates': 410}
train stats after 13152 examples: {'rewards_train/chosen': '0.12633', 'rewards_train/rejected': '0.047826', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.078505', 'logps_train/rejected': '-281.47', 'logps_train/chosen': '-316.85', 'loss/train': '0.6845', 'examples_per_second': '66.649', 'grad_norm': '0', 'counters/examples': 13152, 'counters/updates': 411}
train stats after 13184 examples: {'rewards_train/chosen': '0.13046', 'rewards_train/rejected': '0.025176', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.10528', 'logps_train/rejected': '-296.01', 'logps_train/chosen': '-314.9', 'loss/train': '0.66645', 'examples_per_second': '47.98', 'grad_norm': '0', 'counters/examples': 13184, 'counters/updates': 412}
train stats after 13216 examples: {'rewards_train/chosen': '0.10381', 'rewards_train/rejected': '0.060029', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.043782', 'logps_train/rejected': '-272.47', 'logps_train/chosen': '-305.52', 'loss/train': '0.68933', 'examples_per_second': '44.471', 'grad_norm': '0', 'counters/examples': 13216, 'counters/updates': 413}
train stats after 13248 examples: {'rewards_train/chosen': '0.12524', 'rewards_train/rejected': '0.087035', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.038201', 'logps_train/rejected': '-298.13', 'logps_train/chosen': '-323', 'loss/train': '0.70276', 'examples_per_second': '59.86', 'grad_norm': '0', 'counters/examples': 13248, 'counters/updates': 414}
train stats after 13280 examples: {'rewards_train/chosen': '0.10152', 'rewards_train/rejected': '0.087269', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.014254', 'logps_train/rejected': '-316.44', 'logps_train/chosen': '-324.66', 'loss/train': '0.7098', 'examples_per_second': '49.028', 'grad_norm': '0', 'counters/examples': 13280, 'counters/updates': 415}
train stats after 13312 examples: {'rewards_train/chosen': '0.17608', 'rewards_train/rejected': '0.086281', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.089794', 'logps_train/rejected': '-289.85', 'logps_train/chosen': '-320.55', 'loss/train': '0.67065', 'examples_per_second': '66.441', 'grad_norm': '0', 'counters/examples': 13312, 'counters/updates': 416}
train stats after 13344 examples: {'rewards_train/chosen': '0.17897', 'rewards_train/rejected': '0.095401', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.083568', 'logps_train/rejected': '-295.76', 'logps_train/chosen': '-304.93', 'loss/train': '0.67278', 'examples_per_second': '48.352', 'grad_norm': '0', 'counters/examples': 13344, 'counters/updates': 417}
train stats after 13376 examples: {'rewards_train/chosen': '0.1083', 'rewards_train/rejected': '0.066283', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.042016', 'logps_train/rejected': '-275.58', 'logps_train/chosen': '-310.64', 'loss/train': '0.69873', 'examples_per_second': '55.867', 'grad_norm': '0', 'counters/examples': 13376, 'counters/updates': 418}
train stats after 13408 examples: {'rewards_train/chosen': '0.13338', 'rewards_train/rejected': '0.045021', 'rewards_train/accuracies': '0.5', 'rewards_train/margins': '0.088358', 'logps_train/rejected': '-296.69', 'logps_train/chosen': '-303.53', 'loss/train': '0.6728', 'examples_per_second': '50.231', 'grad_norm': '0', 'counters/examples': 13408, 'counters/updates': 419}
train stats after 13440 examples: {'rewards_train/chosen': '0.1621', 'rewards_train/rejected': '0.065562', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.096537', 'logps_train/rejected': '-263.76', 'logps_train/chosen': '-288.68', 'loss/train': '0.66842', 'examples_per_second': '49.851', 'grad_norm': '0', 'counters/examples': 13440, 'counters/updates': 420}
train stats after 13472 examples: {'rewards_train/chosen': '0.14422', 'rewards_train/rejected': '0.046306', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.097918', 'logps_train/rejected': '-321.96', 'logps_train/chosen': '-322.34', 'loss/train': '0.67637', 'examples_per_second': '54.425', 'grad_norm': '0', 'counters/examples': 13472, 'counters/updates': 421}
train stats after 13504 examples: {'rewards_train/chosen': '0.072495', 'rewards_train/rejected': '0.0077725', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.064723', 'logps_train/rejected': '-285.05', 'logps_train/chosen': '-301.56', 'loss/train': '0.68685', 'examples_per_second': '53.706', 'grad_norm': '0', 'counters/examples': 13504, 'counters/updates': 422}
train stats after 13536 examples: {'rewards_train/chosen': '0.16303', 'rewards_train/rejected': '0.07069', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.092339', 'logps_train/rejected': '-282.09', 'logps_train/chosen': '-292.24', 'loss/train': '0.67179', 'examples_per_second': '69.96', 'grad_norm': '0', 'counters/examples': 13536, 'counters/updates': 423}
train stats after 13568 examples: {'rewards_train/chosen': '0.14847', 'rewards_train/rejected': '0.054356', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.094109', 'logps_train/rejected': '-293.08', 'logps_train/chosen': '-330.26', 'loss/train': '0.67092', 'examples_per_second': '51.468', 'grad_norm': '0', 'counters/examples': 13568, 'counters/updates': 424}
train stats after 13600 examples: {'rewards_train/chosen': '0.13639', 'rewards_train/rejected': '0.073628', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.062765', 'logps_train/rejected': '-315.44', 'logps_train/chosen': '-316.88', 'loss/train': '0.68558', 'examples_per_second': '44.785', 'grad_norm': '0', 'counters/examples': 13600, 'counters/updates': 425}
train stats after 13632 examples: {'rewards_train/chosen': '0.093673', 'rewards_train/rejected': '0.039496', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.054177', 'logps_train/rejected': '-314.8', 'logps_train/chosen': '-341.48', 'loss/train': '0.69969', 'examples_per_second': '48.273', 'grad_norm': '0', 'counters/examples': 13632, 'counters/updates': 426}
train stats after 13664 examples: {'rewards_train/chosen': '0.18902', 'rewards_train/rejected': '0.10377', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.085246', 'logps_train/rejected': '-275.35', 'logps_train/chosen': '-310.39', 'loss/train': '0.67893', 'examples_per_second': '73.838', 'grad_norm': '0', 'counters/examples': 13664, 'counters/updates': 427}
train stats after 13696 examples: {'rewards_train/chosen': '0.15848', 'rewards_train/rejected': '0.087187', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.071293', 'logps_train/rejected': '-322.07', 'logps_train/chosen': '-335.39', 'loss/train': '0.68804', 'examples_per_second': '46.797', 'grad_norm': '0', 'counters/examples': 13696, 'counters/updates': 428}
train stats after 13728 examples: {'rewards_train/chosen': '0.20431', 'rewards_train/rejected': '0.06406', 'rewards_train/accuracies': '0.64844', 'rewards_train/margins': '0.14025', 'logps_train/rejected': '-291.85', 'logps_train/chosen': '-295.94', 'loss/train': '0.64644', 'examples_per_second': '63.142', 'grad_norm': '0', 'counters/examples': 13728, 'counters/updates': 429}
train stats after 13760 examples: {'rewards_train/chosen': '0.19013', 'rewards_train/rejected': '0.093581', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.096548', 'logps_train/rejected': '-328.6', 'logps_train/chosen': '-330.49', 'loss/train': '0.67669', 'examples_per_second': '47.889', 'grad_norm': '0', 'counters/examples': 13760, 'counters/updates': 430}
train stats after 13792 examples: {'rewards_train/chosen': '0.15784', 'rewards_train/rejected': '0.083601', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.074242', 'logps_train/rejected': '-302.93', 'logps_train/chosen': '-335.73', 'loss/train': '0.68416', 'examples_per_second': '48.593', 'grad_norm': '0', 'counters/examples': 13792, 'counters/updates': 431}
train stats after 13824 examples: {'rewards_train/chosen': '0.20997', 'rewards_train/rejected': '0.13723', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.072748', 'logps_train/rejected': '-298.72', 'logps_train/chosen': '-318.31', 'loss/train': '0.68285', 'examples_per_second': '71.43', 'grad_norm': '0', 'counters/examples': 13824, 'counters/updates': 432}
train stats after 13856 examples: {'rewards_train/chosen': '0.1223', 'rewards_train/rejected': '0.062211', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.060093', 'logps_train/rejected': '-283.1', 'logps_train/chosen': '-308.34', 'loss/train': '0.68895', 'examples_per_second': '51.703', 'grad_norm': '0', 'counters/examples': 13856, 'counters/updates': 433}
train stats after 13888 examples: {'rewards_train/chosen': '0.12171', 'rewards_train/rejected': '0.028794', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.092919', 'logps_train/rejected': '-284.52', 'logps_train/chosen': '-313.82', 'loss/train': '0.67093', 'examples_per_second': '49.584', 'grad_norm': '0', 'counters/examples': 13888, 'counters/updates': 434}
train stats after 13920 examples: {'rewards_train/chosen': '0.16367', 'rewards_train/rejected': '0.079967', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.083707', 'logps_train/rejected': '-291.23', 'logps_train/chosen': '-316.23', 'loss/train': '0.67709', 'examples_per_second': '59.761', 'grad_norm': '0', 'counters/examples': 13920, 'counters/updates': 435}
train stats after 13952 examples: {'rewards_train/chosen': '0.15685', 'rewards_train/rejected': '0.077224', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.079629', 'logps_train/rejected': '-287.48', 'logps_train/chosen': '-318.4', 'loss/train': '0.68086', 'examples_per_second': '46.049', 'grad_norm': '0', 'counters/examples': 13952, 'counters/updates': 436}
train stats after 13984 examples: {'rewards_train/chosen': '0.22795', 'rewards_train/rejected': '0.060496', 'rewards_train/accuracies': '0.65625', 'rewards_train/margins': '0.16745', 'logps_train/rejected': '-283.72', 'logps_train/chosen': '-313.84', 'loss/train': '0.63961', 'examples_per_second': '54.131', 'grad_norm': '0', 'counters/examples': 13984, 'counters/updates': 437}
train stats after 14016 examples: {'rewards_train/chosen': '0.17924', 'rewards_train/rejected': '0.022361', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.15688', 'logps_train/rejected': '-311.02', 'logps_train/chosen': '-334.91', 'loss/train': '0.65512', 'examples_per_second': '53.273', 'grad_norm': '0', 'counters/examples': 14016, 'counters/updates': 438}
train stats after 14048 examples: {'rewards_train/chosen': '0.16804', 'rewards_train/rejected': '0.1008', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.067248', 'logps_train/rejected': '-303.55', 'logps_train/chosen': '-309.75', 'loss/train': '0.69349', 'examples_per_second': '50.585', 'grad_norm': '0', 'counters/examples': 14048, 'counters/updates': 439}
train stats after 14080 examples: {'rewards_train/chosen': '0.18303', 'rewards_train/rejected': '0.11334', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.069697', 'logps_train/rejected': '-285.17', 'logps_train/chosen': '-293.12', 'loss/train': '0.68344', 'examples_per_second': '51.42', 'grad_norm': '0', 'counters/examples': 14080, 'counters/updates': 440}
train stats after 14112 examples: {'rewards_train/chosen': '0.18139', 'rewards_train/rejected': '0.16041', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.020977', 'logps_train/rejected': '-315.89', 'logps_train/chosen': '-329.23', 'loss/train': '0.70906', 'examples_per_second': '51.832', 'grad_norm': '0', 'counters/examples': 14112, 'counters/updates': 441}
train stats after 14144 examples: {'rewards_train/chosen': '0.11507', 'rewards_train/rejected': '0.080191', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.034882', 'logps_train/rejected': '-316.19', 'logps_train/chosen': '-309.63', 'loss/train': '0.70755', 'examples_per_second': '46.455', 'grad_norm': '0', 'counters/examples': 14144, 'counters/updates': 442}
train stats after 14176 examples: {'rewards_train/chosen': '0.16045', 'rewards_train/rejected': '0.12583', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.034619', 'logps_train/rejected': '-289.1', 'logps_train/chosen': '-335.63', 'loss/train': '0.70997', 'examples_per_second': '62.578', 'grad_norm': '0', 'counters/examples': 14176, 'counters/updates': 443}
train stats after 14208 examples: {'rewards_train/chosen': '0.073996', 'rewards_train/rejected': '0.066714', 'rewards_train/accuracies': '0.50781', 'rewards_train/margins': '0.0072822', 'logps_train/rejected': '-316.09', 'logps_train/chosen': '-333.7', 'loss/train': '0.71529', 'examples_per_second': '50.151', 'grad_norm': '0', 'counters/examples': 14208, 'counters/updates': 444}
train stats after 14240 examples: {'rewards_train/chosen': '0.17549', 'rewards_train/rejected': '0.093345', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.082147', 'logps_train/rejected': '-308.34', 'logps_train/chosen': '-357.75', 'loss/train': '0.68049', 'examples_per_second': '69.108', 'grad_norm': '0', 'counters/examples': 14240, 'counters/updates': 445}
train stats after 14272 examples: {'rewards_train/chosen': '0.11633', 'rewards_train/rejected': '0.077972', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.038359', 'logps_train/rejected': '-277.65', 'logps_train/chosen': '-297.44', 'loss/train': '0.69986', 'examples_per_second': '57.253', 'grad_norm': '0', 'counters/examples': 14272, 'counters/updates': 446}
train stats after 14304 examples: {'rewards_train/chosen': '0.16889', 'rewards_train/rejected': '0.052844', 'rewards_train/accuracies': '0.63281', 'rewards_train/margins': '0.11605', 'logps_train/rejected': '-312.59', 'logps_train/chosen': '-312.05', 'loss/train': '0.66242', 'examples_per_second': '63.447', 'grad_norm': '0', 'counters/examples': 14304, 'counters/updates': 447}
train stats after 14336 examples: {'rewards_train/chosen': '0.13096', 'rewards_train/rejected': '0.080956', 'rewards_train/accuracies': '0.53125', 'rewards_train/margins': '0.050006', 'logps_train/rejected': '-292.81', 'logps_train/chosen': '-313.27', 'loss/train': '0.69699', 'examples_per_second': '51.845', 'grad_norm': '0', 'counters/examples': 14336, 'counters/updates': 448}
train stats after 14368 examples: {'rewards_train/chosen': '0.08872', 'rewards_train/rejected': '0.091965', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '-0.0032449', 'logps_train/rejected': '-311.83', 'logps_train/chosen': '-329.95', 'loss/train': '0.72345', 'examples_per_second': '46.923', 'grad_norm': '0', 'counters/examples': 14368, 'counters/updates': 449}
train stats after 14400 examples: {'rewards_train/chosen': '0.16237', 'rewards_train/rejected': '0.076027', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.086345', 'logps_train/rejected': '-336.56', 'logps_train/chosen': '-337.6', 'loss/train': '0.67665', 'examples_per_second': '55.729', 'grad_norm': '0', 'counters/examples': 14400, 'counters/updates': 450}
train stats after 14432 examples: {'rewards_train/chosen': '0.16429', 'rewards_train/rejected': '0.075148', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.08914', 'logps_train/rejected': '-310.4', 'logps_train/chosen': '-336', 'loss/train': '0.676', 'examples_per_second': '61.755', 'grad_norm': '0', 'counters/examples': 14432, 'counters/updates': 451}
train stats after 14464 examples: {'rewards_train/chosen': '0.11001', 'rewards_train/rejected': '0.035597', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.07441', 'logps_train/rejected': '-286.78', 'logps_train/chosen': '-305.97', 'loss/train': '0.68049', 'examples_per_second': '46.122', 'grad_norm': '0', 'counters/examples': 14464, 'counters/updates': 452}
train stats after 14496 examples: {'rewards_train/chosen': '0.1245', 'rewards_train/rejected': '0.054951', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.069552', 'logps_train/rejected': '-301.11', 'logps_train/chosen': '-329.61', 'loss/train': '0.68344', 'examples_per_second': '46.996', 'grad_norm': '0', 'counters/examples': 14496, 'counters/updates': 453}
train stats after 14528 examples: {'rewards_train/chosen': '0.15359', 'rewards_train/rejected': '0.01907', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.13452', 'logps_train/rejected': '-271.54', 'logps_train/chosen': '-314.91', 'loss/train': '0.6503', 'examples_per_second': '45.014', 'grad_norm': '0', 'counters/examples': 14528, 'counters/updates': 454}
train stats after 14560 examples: {'rewards_train/chosen': '0.19887', 'rewards_train/rejected': '0.099816', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.099056', 'logps_train/rejected': '-303.95', 'logps_train/chosen': '-322.7', 'loss/train': '0.66863', 'examples_per_second': '66.85', 'grad_norm': '0', 'counters/examples': 14560, 'counters/updates': 455}
train stats after 14592 examples: {'rewards_train/chosen': '0.15192', 'rewards_train/rejected': '0.11187', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.040054', 'logps_train/rejected': '-299.76', 'logps_train/chosen': '-310.71', 'loss/train': '0.6962', 'examples_per_second': '45.654', 'grad_norm': '0', 'counters/examples': 14592, 'counters/updates': 456}
train stats after 14624 examples: {'rewards_train/chosen': '0.1394', 'rewards_train/rejected': '0.045382', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.094015', 'logps_train/rejected': '-282.07', 'logps_train/chosen': '-303.97', 'loss/train': '0.68569', 'examples_per_second': '73.783', 'grad_norm': '0', 'counters/examples': 14624, 'counters/updates': 457}
train stats after 14656 examples: {'rewards_train/chosen': '0.11648', 'rewards_train/rejected': '0.063566', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.052912', 'logps_train/rejected': '-327.62', 'logps_train/chosen': '-313.23', 'loss/train': '0.69245', 'examples_per_second': '51.506', 'grad_norm': '0', 'counters/examples': 14656, 'counters/updates': 458}
train stats after 14688 examples: {'rewards_train/chosen': '0.086926', 'rewards_train/rejected': '0.039932', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.046994', 'logps_train/rejected': '-271.99', 'logps_train/chosen': '-281.91', 'loss/train': '0.68867', 'examples_per_second': '64.936', 'grad_norm': '0', 'counters/examples': 14688, 'counters/updates': 459}
train stats after 14720 examples: {'rewards_train/chosen': '0.10121', 'rewards_train/rejected': '0.087548', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.01366', 'logps_train/rejected': '-284.56', 'logps_train/chosen': '-310.05', 'loss/train': '0.71094', 'examples_per_second': '53.702', 'grad_norm': '0', 'counters/examples': 14720, 'counters/updates': 460}
train stats after 14752 examples: {'rewards_train/chosen': '0.17642', 'rewards_train/rejected': '0.13503', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.041384', 'logps_train/rejected': '-310.53', 'logps_train/chosen': '-326.52', 'loss/train': '0.69622', 'examples_per_second': '48.255', 'grad_norm': '0', 'counters/examples': 14752, 'counters/updates': 461}
train stats after 14784 examples: {'rewards_train/chosen': '0.20099', 'rewards_train/rejected': '0.10604', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.094945', 'logps_train/rejected': '-305.16', 'logps_train/chosen': '-334.37', 'loss/train': '0.6759', 'examples_per_second': '47.678', 'grad_norm': '0', 'counters/examples': 14784, 'counters/updates': 462}
train stats after 14816 examples: {'rewards_train/chosen': '0.1498', 'rewards_train/rejected': '0.070625', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.079173', 'logps_train/rejected': '-320.68', 'logps_train/chosen': '-338.53', 'loss/train': '0.67395', 'examples_per_second': '55.619', 'grad_norm': '0', 'counters/examples': 14816, 'counters/updates': 463}
train stats after 14848 examples: {'rewards_train/chosen': '0.12311', 'rewards_train/rejected': '0.018202', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.10491', 'logps_train/rejected': '-306.32', 'logps_train/chosen': '-334.74', 'loss/train': '0.67466', 'examples_per_second': '48.152', 'grad_norm': '0', 'counters/examples': 14848, 'counters/updates': 464}
train stats after 14880 examples: {'rewards_train/chosen': '0.13606', 'rewards_train/rejected': '0.025631', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.11043', 'logps_train/rejected': '-324.04', 'logps_train/chosen': '-349.91', 'loss/train': '0.66697', 'examples_per_second': '55.628', 'grad_norm': '0', 'counters/examples': 14880, 'counters/updates': 465}
==================================================
==================================================
Running evaluation after 14880 train examples
==================================================
==================================================
train stats after 14912 examples: {'rewards_train/chosen': '0.17755', 'rewards_train/rejected': '0.11388', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.063669', 'logps_train/rejected': '-324.98', 'logps_train/chosen': '-338.82', 'loss/train': '0.6849', 'examples_per_second': '62.759', 'grad_norm': '0', 'counters/examples': 14912, 'counters/updates': 466}
train stats after 14944 examples: {'rewards_train/chosen': '0.11877', 'rewards_train/rejected': '0.043645', 'rewards_train/accuracies': '0.57812', 'rewards_train/margins': '0.075122', 'logps_train/rejected': '-314', 'logps_train/chosen': '-319.02', 'loss/train': '0.68116', 'examples_per_second': '52.807', 'grad_norm': '0', 'counters/examples': 14944, 'counters/updates': 467}
train stats after 14976 examples: {'rewards_train/chosen': '0.18682', 'rewards_train/rejected': '0.042569', 'rewards_train/accuracies': '0.63281', 'rewards_train/margins': '0.14426', 'logps_train/rejected': '-280.47', 'logps_train/chosen': '-291.52', 'loss/train': '0.64635', 'examples_per_second': '49.035', 'grad_norm': '0', 'counters/examples': 14976, 'counters/updates': 468}
train stats after 15008 examples: {'rewards_train/chosen': '0.19354', 'rewards_train/rejected': '0.052638', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.1409', 'logps_train/rejected': '-296.38', 'logps_train/chosen': '-329.49', 'loss/train': '0.66304', 'examples_per_second': '63.14', 'grad_norm': '0', 'counters/examples': 15008, 'counters/updates': 469}
train stats after 15040 examples: {'rewards_train/chosen': '0.15782', 'rewards_train/rejected': '0.034165', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.12366', 'logps_train/rejected': '-292.41', 'logps_train/chosen': '-332.25', 'loss/train': '0.65948', 'examples_per_second': '46.728', 'grad_norm': '0', 'counters/examples': 15040, 'counters/updates': 470}
train stats after 15072 examples: {'rewards_train/chosen': '0.080907', 'rewards_train/rejected': '-0.066098', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.14701', 'logps_train/rejected': '-288.12', 'logps_train/chosen': '-297.39', 'loss/train': '0.65394', 'examples_per_second': '70.591', 'grad_norm': '0', 'counters/examples': 15072, 'counters/updates': 471}
train stats after 15104 examples: {'rewards_train/chosen': '0.14019', 'rewards_train/rejected': '0.039373', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.10082', 'logps_train/rejected': '-268.69', 'logps_train/chosen': '-318.87', 'loss/train': '0.67715', 'examples_per_second': '72.689', 'grad_norm': '0', 'counters/examples': 15104, 'counters/updates': 472}
train stats after 15136 examples: {'rewards_train/chosen': '0.09632', 'rewards_train/rejected': '0.015957', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.080363', 'logps_train/rejected': '-285.01', 'logps_train/chosen': '-315.9', 'loss/train': '0.67266', 'examples_per_second': '51.643', 'grad_norm': '0', 'counters/examples': 15136, 'counters/updates': 473}
train stats after 15168 examples: {'rewards_train/chosen': '0.15443', 'rewards_train/rejected': '0.014426', 'rewards_train/accuracies': '0.60156', 'rewards_train/margins': '0.14', 'logps_train/rejected': '-299.77', 'logps_train/chosen': '-315.95', 'loss/train': '0.65362', 'examples_per_second': '68.861', 'grad_norm': '0', 'counters/examples': 15168, 'counters/updates': 474}
train stats after 15200 examples: {'rewards_train/chosen': '0.10335', 'rewards_train/rejected': '0.00037562', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.10297', 'logps_train/rejected': '-323.78', 'logps_train/chosen': '-342.31', 'loss/train': '0.6721', 'examples_per_second': '51.354', 'grad_norm': '0', 'counters/examples': 15200, 'counters/updates': 475}
train stats after 15232 examples: {'rewards_train/chosen': '0.11041', 'rewards_train/rejected': '0.02392', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.086494', 'logps_train/rejected': '-312.36', 'logps_train/chosen': '-339.34', 'loss/train': '0.67882', 'examples_per_second': '64.773', 'grad_norm': '0', 'counters/examples': 15232, 'counters/updates': 476}
train stats after 15264 examples: {'rewards_train/chosen': '0.13295', 'rewards_train/rejected': '0.014271', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.11868', 'logps_train/rejected': '-285.96', 'logps_train/chosen': '-301.24', 'loss/train': '0.66506', 'examples_per_second': '45.832', 'grad_norm': '0', 'counters/examples': 15264, 'counters/updates': 477}
train stats after 15296 examples: {'rewards_train/chosen': '0.082389', 'rewards_train/rejected': '0.020643', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.061746', 'logps_train/rejected': '-283.62', 'logps_train/chosen': '-301.75', 'loss/train': '0.69147', 'examples_per_second': '51.792', 'grad_norm': '0', 'counters/examples': 15296, 'counters/updates': 478}
train stats after 15328 examples: {'rewards_train/chosen': '0.0078948', 'rewards_train/rejected': '-0.073502', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.081396', 'logps_train/rejected': '-301.21', 'logps_train/chosen': '-295.56', 'loss/train': '0.67959', 'examples_per_second': '72.85', 'grad_norm': '0', 'counters/examples': 15328, 'counters/updates': 479}
train stats after 15360 examples: {'rewards_train/chosen': '0.12801', 'rewards_train/rejected': '-0.017997', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.14601', 'logps_train/rejected': '-282.04', 'logps_train/chosen': '-303.43', 'loss/train': '0.64808', 'examples_per_second': '45.013', 'grad_norm': '0', 'counters/examples': 15360, 'counters/updates': 480}
train stats after 15392 examples: {'rewards_train/chosen': '0.13114', 'rewards_train/rejected': '0.018981', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.11216', 'logps_train/rejected': '-302.26', 'logps_train/chosen': '-309.24', 'loss/train': '0.66745', 'examples_per_second': '65.239', 'grad_norm': '0', 'counters/examples': 15392, 'counters/updates': 481}
train stats after 15424 examples: {'rewards_train/chosen': '0.12485', 'rewards_train/rejected': '-0.012706', 'rewards_train/accuracies': '0.625', 'rewards_train/margins': '0.13756', 'logps_train/rejected': '-279.78', 'logps_train/chosen': '-313.29', 'loss/train': '0.65919', 'examples_per_second': '47.697', 'grad_norm': '0', 'counters/examples': 15424, 'counters/updates': 482}
train stats after 15456 examples: {'rewards_train/chosen': '0.087672', 'rewards_train/rejected': '-0.021339', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.10901', 'logps_train/rejected': '-296.17', 'logps_train/chosen': '-321.05', 'loss/train': '0.66948', 'examples_per_second': '56.025', 'grad_norm': '0', 'counters/examples': 15456, 'counters/updates': 483}
train stats after 15488 examples: {'rewards_train/chosen': '0.10836', 'rewards_train/rejected': '0.10178', 'rewards_train/accuracies': '0.46875', 'rewards_train/margins': '0.0065769', 'logps_train/rejected': '-309.8', 'logps_train/chosen': '-312.92', 'loss/train': '0.72318', 'examples_per_second': '48.188', 'grad_norm': '0', 'counters/examples': 15488, 'counters/updates': 484}
train stats after 15520 examples: {'rewards_train/chosen': '0.12258', 'rewards_train/rejected': '0.018472', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.10411', 'logps_train/rejected': '-301.74', 'logps_train/chosen': '-317.61', 'loss/train': '0.6741', 'examples_per_second': '51.073', 'grad_norm': '0', 'counters/examples': 15520, 'counters/updates': 485}
train stats after 15552 examples: {'rewards_train/chosen': '0.087909', 'rewards_train/rejected': '0.052221', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.035688', 'logps_train/rejected': '-300.68', 'logps_train/chosen': '-312.75', 'loss/train': '0.70318', 'examples_per_second': '51.908', 'grad_norm': '0', 'counters/examples': 15552, 'counters/updates': 486}
train stats after 15584 examples: {'rewards_train/chosen': '0.10916', 'rewards_train/rejected': '0.047998', 'rewards_train/accuracies': '0.51562', 'rewards_train/margins': '0.06116', 'logps_train/rejected': '-306.49', 'logps_train/chosen': '-316.9', 'loss/train': '0.69612', 'examples_per_second': '56.476', 'grad_norm': '0', 'counters/examples': 15584, 'counters/updates': 487}
train stats after 15616 examples: {'rewards_train/chosen': '0.11558', 'rewards_train/rejected': '0.021965', 'rewards_train/accuracies': '0.59375', 'rewards_train/margins': '0.09362', 'logps_train/rejected': '-278.47', 'logps_train/chosen': '-286.42', 'loss/train': '0.68087', 'examples_per_second': '57.198', 'grad_norm': '0', 'counters/examples': 15616, 'counters/updates': 488}
train stats after 15648 examples: {'rewards_train/chosen': '0.10419', 'rewards_train/rejected': '0.024595', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.079597', 'logps_train/rejected': '-289.28', 'logps_train/chosen': '-305.37', 'loss/train': '0.68082', 'examples_per_second': '51.835', 'grad_norm': '0', 'counters/examples': 15648, 'counters/updates': 489}
train stats after 15680 examples: {'rewards_train/chosen': '0.14741', 'rewards_train/rejected': '0.10422', 'rewards_train/accuracies': '0.54688', 'rewards_train/margins': '0.043185', 'logps_train/rejected': '-308.58', 'logps_train/chosen': '-329.91', 'loss/train': '0.71447', 'examples_per_second': '48.208', 'grad_norm': '0', 'counters/examples': 15680, 'counters/updates': 490}
train stats after 15712 examples: {'rewards_train/chosen': '0.083974', 'rewards_train/rejected': '-0.00010656', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.084081', 'logps_train/rejected': '-293.9', 'logps_train/chosen': '-298.01', 'loss/train': '0.68259', 'examples_per_second': '51.732', 'grad_norm': '0', 'counters/examples': 15712, 'counters/updates': 491}
train stats after 15744 examples: {'rewards_train/chosen': '0.17575', 'rewards_train/rejected': '-0.0025964', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.17835', 'logps_train/rejected': '-283.34', 'logps_train/chosen': '-318.02', 'loss/train': '0.64118', 'examples_per_second': '69.424', 'grad_norm': '0', 'counters/examples': 15744, 'counters/updates': 492}
train stats after 15776 examples: {'rewards_train/chosen': '0.14651', 'rewards_train/rejected': '0.11477', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.031734', 'logps_train/rejected': '-316.15', 'logps_train/chosen': '-307.33', 'loss/train': '0.7071', 'examples_per_second': '69.813', 'grad_norm': '0', 'counters/examples': 15776, 'counters/updates': 493}
train stats after 15808 examples: {'rewards_train/chosen': '0.046113', 'rewards_train/rejected': '0.028236', 'rewards_train/accuracies': '0.58594', 'rewards_train/margins': '0.017877', 'logps_train/rejected': '-296.26', 'logps_train/chosen': '-327.39', 'loss/train': '0.71522', 'examples_per_second': '46.585', 'grad_norm': '0', 'counters/examples': 15808, 'counters/updates': 494}
train stats after 15840 examples: {'rewards_train/chosen': '0.10437', 'rewards_train/rejected': '0.007201', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.097166', 'logps_train/rejected': '-282.61', 'logps_train/chosen': '-313.76', 'loss/train': '0.66751', 'examples_per_second': '50.942', 'grad_norm': '0', 'counters/examples': 15840, 'counters/updates': 495}
train stats after 15872 examples: {'rewards_train/chosen': '0.11996', 'rewards_train/rejected': '0.023652', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.096309', 'logps_train/rejected': '-306.76', 'logps_train/chosen': '-325.32', 'loss/train': '0.67508', 'examples_per_second': '47.878', 'grad_norm': '0', 'counters/examples': 15872, 'counters/updates': 496}
train stats after 15904 examples: {'rewards_train/chosen': '0.088703', 'rewards_train/rejected': '-0.018902', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.1076', 'logps_train/rejected': '-303.68', 'logps_train/chosen': '-333.99', 'loss/train': '0.66429', 'examples_per_second': '45.015', 'grad_norm': '0', 'counters/examples': 15904, 'counters/updates': 497}
train stats after 15936 examples: {'rewards_train/chosen': '0.12832', 'rewards_train/rejected': '0.03211', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.096206', 'logps_train/rejected': '-305.02', 'logps_train/chosen': '-329.91', 'loss/train': '0.67803', 'examples_per_second': '47.438', 'grad_norm': '0', 'counters/examples': 15936, 'counters/updates': 498}
train stats after 15968 examples: {'rewards_train/chosen': '0.17355', 'rewards_train/rejected': '0.02049', 'rewards_train/accuracies': '0.61719', 'rewards_train/margins': '0.15306', 'logps_train/rejected': '-284.76', 'logps_train/chosen': '-321.59', 'loss/train': '0.64984', 'examples_per_second': '49.44', 'grad_norm': '0', 'counters/examples': 15968, 'counters/updates': 499}
train stats after 16000 examples: {'rewards_train/chosen': '0.14524', 'rewards_train/rejected': '0.029906', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.11534', 'logps_train/rejected': '-332.94', 'logps_train/chosen': '-324.04', 'loss/train': '0.66619', 'examples_per_second': '48.953', 'grad_norm': '0', 'counters/examples': 16000, 'counters/updates': 500}
train stats after 16032 examples: {'rewards_train/chosen': '0.11436', 'rewards_train/rejected': '-0.0060679', 'rewards_train/accuracies': '0.5625', 'rewards_train/margins': '0.12043', 'logps_train/rejected': '-291.03', 'logps_train/chosen': '-329.84', 'loss/train': '0.67176', 'examples_per_second': '58.133', 'grad_norm': '0', 'counters/examples': 16032, 'counters/updates': 501}
train stats after 16064 examples: {'rewards_train/chosen': '0.020877', 'rewards_train/rejected': '-0.00030683', 'rewards_train/accuracies': '0.52344', 'rewards_train/margins': '0.021184', 'logps_train/rejected': '-304.01', 'logps_train/chosen': '-326.01', 'loss/train': '0.71271', 'examples_per_second': '70.267', 'grad_norm': '0', 'counters/examples': 16064, 'counters/updates': 502}
train stats after 16096 examples: {'rewards_train/chosen': '0.14409', 'rewards_train/rejected': '-0.0025817', 'rewards_train/accuracies': '0.57031', 'rewards_train/margins': '0.14667', 'logps_train/rejected': '-269.6', 'logps_train/chosen': '-300.73', 'loss/train': '0.64925', 'examples_per_second': '46.152', 'grad_norm': '0', 'counters/examples': 16096, 'counters/updates': 503}
train stats after 16128 examples: {'rewards_train/chosen': '0.1222', 'rewards_train/rejected': '-0.0055814', 'rewards_train/accuracies': '0.60938', 'rewards_train/margins': '0.12778', 'logps_train/rejected': '-314.09', 'logps_train/chosen': '-325.56', 'loss/train': '0.66315', 'examples_per_second': '45.748', 'grad_norm': '0', 'counters/examples': 16128, 'counters/updates': 504}
train stats after 16160 examples: {'rewards_train/chosen': '-0.014512', 'rewards_train/rejected': '-0.021199', 'rewards_train/accuracies': '0.53906', 'rewards_train/margins': '0.0066872', 'logps_train/rejected': '-269.96', 'logps_train/chosen': '-272.25', 'loss/train': '0.72314', 'examples_per_second': '48.512', 'grad_norm': '0', 'counters/examples': 16160, 'counters/updates': 505}
train stats after 16192 examples: {'rewards_train/chosen': '0.044415', 'rewards_train/rejected': '-0.01742', 'rewards_train/accuracies': '0.55469', 'rewards_train/margins': '0.061835', 'logps_train/rejected': '-279.96', 'logps_train/chosen': '-331.17', 'loss/train': '0.68814', 'examples_per_second': '53.009', 'grad_norm': '0', 'counters/examples': 16192, 'counters/updates': 506}
