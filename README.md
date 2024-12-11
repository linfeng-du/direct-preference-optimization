## Project Structure
The core packages and modules in this project are:

- `preference_datasets`:
    A package of preference datasets and necessary utilities for data loading.
    - `dataset.py`: PyTorch dataset, sampler, and collate function.
    - `persona.py`: Function for loading and processing the PERSONA dataset.

- `adapters`:
    A package of adapters.
    Each adapter module has a controller and an adapter layer.
    The controller swaps in the adapter layers and manages layer attributes.

- `trainer.py`: Trainer class that supports distrubuted training via Hugging Face Accelerate.

- `train.py`: Entrence of model training.

This project relies on `hydra` and `OmegaConf` to manage all configurations.
The structure of the configuration files and the most related arguments for each module are as follows:

```bash
config/
├── dataset
│   └── persona.yaml
│       ├── dataset             # Name of the dataset
│       ├── prepend_persona     # Whether to prepend the persona before each prompt
│       └── n_clusters          # Number of user clusters, should be equal to adapter.n_loras
├── model
│   ├── gpt2-instruct.yaml
│   │   ├── model               # Model identifier on Hugging Face
│   │   ├── target_modules      # Target modules to be replaced by adapters
│   │   └── ...
│   └── llama1B-instruct.yaml
├── adapter
│   ├── lora.yaml
│   │   ├── adaptor             # Name of the adapter
│   │   ├── r                   # Rank of the adatper matrices
│   │   └── ...
│   └── mixture_of_loras.yaml
│       ├── ...
│       └── n_loras             # Number of LoRA experts. Should be equal to dataset.n_clusters
├── loss
│   ├── dpo.yaml
│   └── ipo.yaml
└── config.yaml
```

## Training
`scripts` contains the all the bash scripts that we use to train our models.
Arguments with the value of "???" in the configuration files must be specified in commend line.
