[2024-07-10 14:14:19,678] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 14:14:19,970] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 14:14:19,977] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 14:14:19,980] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-10 14:14:22,591] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 14:14:22,591] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[2024-07-10 14:14:22,593][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2024-07-10 14:14:22,902] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 14:14:22,903][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 2
[2024-07-10 14:14:22,926] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 14:14:22,928][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
[2024-07-10 14:14:22,929] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-10 14:14:22,930][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 3
[2024-07-10 14:14:22,930][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2024-07-10 14:14:22,933][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2024-07-10 14:14:22,937][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2024-07-10 14:14:22,939][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
seed: 0
exp_name: dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: false
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
local_run_dir: .cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_015877
lr: 5.0e-06
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 1
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
  n_transformer_layers: 2
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

seed: 0
exp_name: dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: false
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
local_run_dir: .cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_015956
lr: 5.0e-06
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 1
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
  n_transformer_layers: 2
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_015877
================================================================================
building policy
================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_015956
================================================================================
building policy
WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
WARNING: eval_every must be divisible by batch_size
Setting eval_every to 4960
seed: 0
exp_name: dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: false
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
local_run_dir: .cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_029142
lr: 5.0e-06
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 1
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
  n_transformer_layers: 2
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

seed: 0
exp_name: dpo
batch_size: 32
eval_batch_size: 64
debug: false
fsdp_port: null
datasets:
- prism
use_lora: false
use_hnet: false
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
local_run_dir: .cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_029413
lr: 5.0e-06
gradient_accumulation_steps: 1
max_grad_norm: 10.0
max_length: 512
max_prompt_length: 256
n_epochs: 1
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
  n_transformer_layers: 2
  d_emb: 188
  d_hnet: 128
  dropout: 0.0
  use_dummies: true
lora:
  r: 16

================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_029142
================================================================================
building policy
================================================================================
Writing to cn-g023.server.mila.quebec:.cache/emiliano.penaloza/dpo_2024-07-10_14-14-23_029413
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
Number of trainable parameters: 124.44M
starting single-process worker
Creating trainer on process 1 with world size 1
Number of trainable parameters: 124.44M
Creating trainer on process 2 with world size 1
starting single-process worker
starting single-process worker
Number of trainable parameters: 124.44M
Creating trainer on process 3 with world size 1
Number of trainable parameters: 124.44M
name='prism'
name='prism'
name='prism'
Creating trainer on process 0 with world size 1
Loading tokenizer vicgalle/gpt2-open-instruct-v1
Loaded train data iterator
name='prism'
Finished generating 1 epochs on train split
Finished generating 1 epochs on train split
name='prism'
name='prism'
Finished generating 1 epochs on train split
name='prism'
Finished generating 1 epochs on train split
name='prism'
Finished generating 1 epochs on test split
Finished generating 1 epochs on test split
Finished generating 1 epochs on test split
Finished generating 1 epochs on test split
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
ninja: no work to do.
Time to load cpu_adam op: 2.589517116546631 seconds
Time to load cpu_adam op: 2.6207010746002197 seconds
Time to load cpu_adam op: 2.612321138381958 seconds
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000005, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000005, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000005, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
[2024-07-10 14:14:47,555][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 1
[2024-07-10 14:14:48,349][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 2
[2024-07-10 14:14:48,349][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 3
Installed CUDA version 11.7 does not match the version torch was compiled with 11.8 but since the APIs are compatible, accepting this combination
ninja: no work to do.
Time to load cpu_adam op: 2.616515636444092 seconds
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000005, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
[2024-07-10 14:14:49,403] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed info: version=0.14.0, git-hash=unknown, git-branch=unknown
[2024-07-10 14:14:49,636][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:2 to store for rank: 0
[2024-07-10 14:14:49,636][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 14:14:49,644][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 14:14:49,644][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 14:14:49,646][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:2 with 4 nodes.
[2024-07-10 14:14:50,377] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2024-07-10 14:14:50,378] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2024-07-10 14:14:50,378] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2024-07-10 14:14:50,382] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = DeepSpeedCPUAdam
[2024-07-10 14:14:50,382] [INFO] [utils.py:56:is_zero_supported_optimizer] Checking ZeRO support for optimizer=DeepSpeedCPUAdam type=<class 'deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam'>
[2024-07-10 14:14:50,382] [INFO] [logging.py:96:log_dist] [Rank 0] Creating torch.bfloat16 ZeRO stage 2 optimizer
[2024-07-10 14:14:50,382] [INFO] [stage_1_and_2.py:149:__init__] Reduce bucket size 500,000,000
[2024-07-10 14:14:50,382] [INFO] [stage_1_and_2.py:150:__init__] Allgather bucket size 500,000,000
[2024-07-10 14:14:50,382] [INFO] [stage_1_and_2.py:151:__init__] CPU Offload: True
[2024-07-10 14:14:50,382] [INFO] [stage_1_and_2.py:152:__init__] Round robin gradient partitioning: False
[2024-07-10 14:14:51,013] [INFO] [utils.py:800:see_memory_usage] Before initializing optimizer states
[2024-07-10 14:14:51,014] [INFO] [utils.py:801:see_memory_usage] MA 0.32 GB         Max_MA 0.32 GB         CA 0.32 GB         Max_CA 0 GB 
[2024-07-10 14:14:51,015] [INFO] [utils.py:808:see_memory_usage] CPU Virtual Memory:  used = 36.79 GB, percent = 3.7%
[2024-07-10 14:14:51,418] [INFO] [utils.py:800:see_memory_usage] After initializing optimizer states
[2024-07-10 14:14:51,419] [INFO] [utils.py:801:see_memory_usage] MA 0.32 GB         Max_MA 0.32 GB         CA 0.32 GB         Max_CA 0 GB 
[2024-07-10 14:14:51,419] [INFO] [utils.py:808:see_memory_usage] CPU Virtual Memory:  used = 37.11 GB, percent = 3.7%
[2024-07-10 14:14:51,419] [INFO] [stage_1_and_2.py:542:__init__] optimizer state initialized
[2024-07-10 14:14:51,797] [INFO] [utils.py:800:see_memory_usage] After initializing ZeRO optimizer
[2024-07-10 14:14:51,798] [INFO] [utils.py:801:see_memory_usage] MA 0.32 GB         Max_MA 0.32 GB         CA 0.32 GB         Max_CA 0 GB 
[2024-07-10 14:14:51,798] [INFO] [utils.py:808:see_memory_usage] CPU Virtual Memory:  used = 37.11 GB, percent = 3.7%
[2024-07-10 14:14:51,799] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = DeepSpeedCPUAdam
[2024-07-10 14:14:51,799] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2024-07-10 14:14:51,799] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2024-07-10 14:14:51,799] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[3.311258278145696e-08], mom=[(0.9, 0.999)]
[2024-07-10 14:14:51,799] [INFO] [config.py:996:print] DeepSpeedEngine configuration:
[2024-07-10 14:14:51,800] [INFO] [config.py:1000:print]   activation_checkpointing_config  {
    "partition_activations": false, 
    "contiguous_memory_optimization": false, 
    "cpu_checkpointing": false, 
    "number_checkpoints": null, 
    "synchronize_checkpoint_boundary": false, 
    "profile": false
}
[2024-07-10 14:14:51,800] [INFO] [config.py:1000:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2024-07-10 14:14:51,800] [INFO] [config.py:1000:print]   amp_enabled .................. False
[2024-07-10 14:14:51,800] [INFO] [config.py:1000:print]   amp_params ................... False
[2024-07-10 14:14:51,800] [INFO] [config.py:1000:print]   autotuning_config ............ {
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
[2024-07-10 14:14:51,801] [INFO] [config.py:1000:print]   bfloat16_enabled ............. True
[2024-07-10 14:14:51,801] [INFO] [config.py:1000:print]   bfloat16_immediate_grad_update  False
[2024-07-10 14:14:51,801] [INFO] [config.py:1000:print]   checkpoint_parallel_write_pipeline  False
[2024-07-10 14:14:51,801] [INFO] [config.py:1000:print]   checkpoint_tag_validation_enabled  True
[2024-07-10 14:14:51,801] [INFO] [config.py:1000:print]   checkpoint_tag_validation_fail  False
[2024-07-10 14:14:51,801] [INFO] [config.py:1000:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7f2c61773640>
[2024-07-10 14:14:51,801] [INFO] [config.py:1000:print]   communication_data_type ...... None
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   compile_config ............... enabled=False backend='inductor' kwargs={}
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   curriculum_enabled_legacy .... False
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   curriculum_params_legacy ..... False
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   data_efficiency_enabled ...... False
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   dataloader_drop_last ......... False
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   disable_allgather ............ False
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   dump_state ................... False
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   dynamic_loss_scale_args ...... None
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   eigenvalue_enabled ........... False
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   eigenvalue_gas_boundary_resolution  1
[2024-07-10 14:14:51,802] [INFO] [config.py:1000:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   eigenvalue_layer_num ......... 0
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   eigenvalue_max_iter .......... 100
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   eigenvalue_stability ......... 1e-06
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   eigenvalue_tol ............... 0.01
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   eigenvalue_verbose ........... False
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   elasticity_enabled ........... False
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   flops_profiler_config ........ {
    "enabled": false, 
    "recompute_fwd_factor": 0.0, 
    "profile_step": 1, 
    "module_depth": -1, 
    "top_modules": 1, 
    "detailed": true, 
    "output_file": null
}
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   fp16_auto_cast ............... None
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   fp16_enabled ................. False
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   fp16_master_weights_and_gradients  False
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   global_rank .................. 0
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   grad_accum_dtype ............. None
[2024-07-10 14:14:51,803] [INFO] [config.py:1000:print]   gradient_accumulation_steps .. 1
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   gradient_clipping ............ 10.0
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   gradient_predivide_factor .... 1.0
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   graph_harvesting ............. False
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   initial_dynamic_scale ........ 1
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   load_universal_checkpoint .... False
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   loss_scale ................... 1.0
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   memory_breakdown ............. False
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   mics_hierarchial_params_gather  False
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   mics_shard_size .............. -1
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') enabled=False
[2024-07-10 14:14:51,804] [INFO] [config.py:1000:print]   nebula_config ................ {
    "enabled": false, 
    "persistent_storage_path": null, 
    "persistent_time_interval": 100, 
    "num_of_version_in_retention": 2, 
    "enable_nebula_load": true, 
    "load_path": null
}
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   optimizer_legacy_fusion ...... False
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   optimizer_name ............... None
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   optimizer_params ............. None
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   pld_enabled .................. False
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   pld_params ................... False
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   prescale_gradients ........... False
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   scheduler_name ............... None
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   scheduler_params ............. None
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   seq_parallel_communication_data_type  torch.float32
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   sparse_attention ............. None
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   sparse_gradients_enabled ..... False
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   steps_per_print .............. inf
[2024-07-10 14:14:51,805] [INFO] [config.py:1000:print]   train_batch_size ............. 128
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   train_micro_batch_size_per_gpu  32
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   use_data_before_expert_parallel_  False
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   use_node_local_storage ....... False
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   wall_clock_breakdown ......... False
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   weight_quantization_config ... None
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   world_size ................... 4
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   zero_allow_untested_optimizer  True
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   zero_config .................. stage=2 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500,000,000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500,000,000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=DeepSpeedZeroOffloadParamConfig(device='cpu', nvme_path=None, buffer_count=5, buffer_size=100,000,000, max_in_cpu=1,000,000,000, pin_memory=False) offload_optimizer=DeepSpeedZeroOffloadOptimizerConfig(device='cpu', nvme_path=None, buffer_count=4, pin_memory=False, pipeline=False, pipeline_read=False, pipeline_write=False, fast_init=False, ratio=1.0) sub_group_size=1,000,000,000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50,000,000 param_persistence_threshold=100,000 model_persistence_threshold=sys.maxsize max_live_parameters=1,000,000,000 max_reuse_distance=1,000,000,000 gather_16bit_weights_on_model_save=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   zero_enabled ................. True
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   zero_force_ds_cpu_optimizer .. True
[2024-07-10 14:14:51,806] [INFO] [config.py:1000:print]   zero_optimization_stage ...... 2
[2024-07-10 14:14:51,806] [INFO] [config.py:986:print_user_config]   json = {
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
==================================================
policy_chosen_logps=tensor([-353.4438, -353.9847, -403.8945, -586.7198, -794.3762, -442.9021,
        -381.4707, -327.1400, -303.5878, -444.5127, -142.8492, -209.4908,
        -671.8138, -243.2903, -284.5875, -163.3138, -387.5571, -623.2220,
        -635.6361, -150.4126, -347.6905, -240.8307, -236.6509, -200.1641,
        -476.6263, -384.4733, -343.1382, -194.0466, -225.6235, -558.7418,
        -691.1704, -258.6511, -162.7530, -216.7386, -183.4482, -565.8156,
        -320.1089, -333.1843, -163.2983, -311.9850, -274.3492, -740.6007,
        -495.7584, -337.7808, -605.1988, -187.9356, -192.7101,  -45.0100,
        -151.3162, -625.1614, -269.5983, -526.2825, -807.6501, -751.0339,
        -295.9523, -378.1992, -344.7043, -719.8946, -294.2471, -637.5215,
        -154.0328, -182.6703, -282.8351, -284.5453], device='cuda:1')
==================================================
==================================================
policy_chosen_logps=tensor([-349.5011, -461.1997, -179.4229, -511.8547, -238.2988, -190.9185,
        -237.9461, -325.5363, -373.7138, -391.5602, -387.8332, -334.0201,
        -262.8650, -661.7019, -164.5441, -198.2607, -190.1854, -456.4599,
        -619.8917, -499.7127, -495.6325, -257.4736, -130.0356, -363.4862,
        -156.6761, -650.6331, -540.3646, -541.3998, -361.3112, -634.3711,
        -198.2416, -285.4804, -500.5507, -276.7548, -410.7408, -668.8566,
        -155.7357, -351.6784, -377.5518,  -88.4357, -197.1076, -130.1238,
        -539.1643, -450.4019,  -62.2116, -654.2671, -350.8903, -149.3905,
        -305.9323, -192.6785, -738.1078, -618.4640, -282.1917, -242.4765,
        -715.0076, -241.9378, -294.6611, -460.6273, -115.8855, -622.3152,
        -302.3416, -158.0328,  -36.3685, -664.5181], device='cuda:3')
==================================================
==================================================
policy_chosen_logps=tensor([-478.8079, -277.3048, -340.0723, -130.0618, -151.0795, -765.4478,
        -546.0079, -184.7119, -351.4344, -302.6402, -650.2316, -278.5900,
        -134.7650, -381.6561, -368.8956, -330.6810, -494.5209, -166.1696,
        -368.0650, -246.7997, -113.2997, -664.9802, -314.3577, -222.2381,
        -197.6728, -233.8745, -293.4621,  -74.2336, -171.6956, -297.6754,
        -315.1131,  -90.8974, -461.1844, -227.4813, -286.5475, -478.0888,
        -445.6951, -247.7635, -381.7415, -147.3932, -160.4300, -348.0693,
        -417.0550, -568.4487, -178.0456, -320.5659, -607.7352, -147.1096,
        -330.3785, -614.9235, -229.1317, -180.7630, -178.1554, -322.1158,
        -437.6347,  -88.2290, -382.1200, -275.1857, -673.1816, -461.6810,
        -747.5604, -301.3508, -282.8802, -173.9111], device='cuda:0')
==================================================
==================================================
policy_chosen_logps=tensor([-215.0832, -606.7388, -601.3875, -399.6035, -378.1168, -540.8073,
        -163.6498,  -58.8844, -205.2244, -187.8398, -369.4939, -272.1057,
        -680.1929, -336.1908, -124.3267, -178.7757, -492.6132, -214.5767,
         -86.5205, -328.1690, -111.2397, -309.7108,  -30.7430, -566.0378,
        -157.4795, -531.5207, -334.5767, -326.6253, -298.1713, -660.1744,
        -221.4224, -104.1412, -250.4739, -210.9373, -165.0508, -668.6172,
        -368.8184, -788.8470, -569.3666, -832.6170, -217.3992, -427.5106,
        -230.0893, -568.2428, -239.8789, -666.7268, -278.0306, -200.9087,
        -376.8568, -216.6897, -723.4661, -544.2168, -362.9224, -394.4196,
        -240.1073, -174.4366, -341.5350, -717.7162, -529.0885, -566.6774,
        -779.3796, -159.1875, -101.4708,  -57.6738], device='cuda:2')
==================================================
