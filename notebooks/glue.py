from datasets import Dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# %%
from transformers import BertTokenizer, BertModel
import torch
from torch.nn import MultiheadAttention
from transformers import LlamaModel, LlamaConfig,LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch.nn as nn 
#if the directory is not the root of the project, change it
os.chdir('/home/mila/e/emiliano.penaloza/RLPHF')
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ['TRANSFORMERS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['HF_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['HF_DATASETS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['TORCH_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
cache_dir = '/home/mila/e/emiliano.penaloza/scratch/models'



# %%
import copy 
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class HyperNetController():
    def __init__(self,hyper_net, r ,  A, a , target_modules = ['q_proj','k_proj','v_proj']):
        self.hyper_net = hyper_net
        self.target_modules = target_modules
        self.hypernet_layers = []
        self.d_emb = hyper_net.d_emb
        self.r = r 
        self.A = A
        self.a = a 
        # self.num_layers = num_layers
        
    def setHyperNetLayer(self,w0):
        self.hypernet_layers.append( HyperNetLinear(self,w0,self.d_emb,self.d_emb,r = self.r,A = self.A, a = self.a))
        return self.hypernet_layers[-1]
    
    def replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():


            if "lora_" in name:
                
                module.to(old_module.weight.device)    
    def get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name
    
    def augmentLLM(self, model,verbose = False):
        
        self.model = model
        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            if isinstance(self.target_modules,str):
                target_module_found = re.fullmatch(self.target_modules, key)
            else: 
                target_module_found = any(key.endswith(target_key) for target_key in self.target_modules)


            if target_module_found:
                parent,target,targe_name = self.get_submodules(key)
                new_module = self.setHyperNetLayer( target.weight)
                self.replace_module(parent, targe_name, new_module, target)
                if verbose:
                    print('replaced',key)

    def setLayerWeights(self,hypernet_outputs, layer):
        assert hypernet_outputs.shape[1] == self.num_layers 
        for l,hyper_out in zip(self.hypernet_layers,hypernet_outputs):
            for k,_ in enumerate(self.target_modules):
                
                l = self.hypernet_layers[l]
    def updateLayers(self,new_layer_tensor):

        #new_layer_tensor is of shape b X (l * kvq) X r X a*2 
        for i,l in enumerate(self.hypernet_layers,):
            #l_new is of shape b X r X a*2 
            l_new = new_layer_tensor[:,i]

            a = l_new[:,:,:self.a ]
            b = l_new[:,:,self.a: ]
            l.set_adapter(a ,b )
    def freezeParams(self,model,verbose = True):
        #print trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'hyper' not in name: 
                    param.requires_grad = False
                elif verbose:
                    print (name, param.data.shape)
            
            
                
        

class HyperNet(nn.Module):
    def __init__(self, r, alpha, dropout, output_emb,d_emb, d_model,n_transformer_layers,hypernet_heads = 2,target_modules = ['q_proj','k_proj','v_proj'],a=16):
        super(HyperNet, self).__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.n_transformer_layers = n_transformer_layers
        self.d_emb = d_emb
        self.a = a 

        
        #each attention head is 3*(d_emb * d_emb) as each k,v,h matrix is d_emb * d_emb
        #We want to use an r-ranked represtation of the d_emb * d_emb matrix so we decompose the total output parameters by r 
        #We replace d_emb by a singular r rankned vector

        
        # Initialize 
        # encoder layers with user_emb_dim
        self.target_modules = target_modules
        encoder_layer = TransformerEncoderLayer(d_model=d_emb, nhead=hypernet_heads, dim_feedforward=d_emb * 4, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        #make an mlp to transform the output of the encoder into d_model 
        self.mlp_encoder = nn.Sequential(
            nn.Linear(d_emb, d_emb * 2),
            nn.ReLU(),
            nn.Linear(d_emb * 2, d_model)
        )
        # Initialize transformer decoder layers with d_emb

        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=hypernet_heads, dim_feedforward=d_emb * 4, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=n_transformer_layers)
        
        # MLP to map the decoder output to self.net_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model* 2, self.a  * 2 *3  )
        )
        
    def forward(self, task_embedding, identity_matrix):
        
        # Encode the task embedding
        encoded_task = self.transformer_encoder(task_embedding)
        encoded_task = self.mlp_encoder(encoded_task)
        
        # Decode to generate the network parameters
        decoded_output = self.transformer_decoder(identity_matrix, encoded_task)
        
        # Map the decoder output to the network parameters using MLP
        params = self.mlp(decoded_output)
        # params = self.organize_outputs(params)
        return params
    
    def organize_outputs(self, outputs):
        param_l = []
        for out in outputs:
            sub_list = []
            s = 0 
            for target in self.target_modules:
                
                sub_list.append(out[s:2 *(s+self.r) ])
                s = 2* (s + self.r)
            param_l.append(sub_list)
        return param_l
            

            
class HyperNetLinear(nn.Linear):
    def __init__(self, hypernet, w0, in_features, out_features, r=1, scaling=1, A=512, a=16):
        nn.Linear.__init__(self, in_features, out_features)
        self.scaling = scaling
        self.hypernet = hypernet
        self.register_buffer('w0', w0.detach())  # Register w0 as a buffer

        self.hyperAdapterA = nn.Parameter(nn.init.xavier_normal_(torch.empty(a, r)).to(w0.device))
        self.hyperAdapterB = nn.Parameter(nn.init.xavier_normal_(torch.empty(a, r)).to(w0.device))

        # Register orthA and orthB as buffers instead of Parameters
        self.register_buffer('orthA', self.make_orth(A, a).to(w0.device))
        self.register_buffer('orthB', self.make_orth(A, a).to(w0.device))

    def make_orth(self, A, a):
        gaus = torch.randn(A, A)
        svd = torch.linalg.svd(gaus)        
        return (svd[0] @ svd[2])[:a]

    def set_adapter(self, adapterA, adapterB, transposeA=True):
        self.hyperAdapterA = nn.Parameter(adapterA if not transposeA else torch.transpose(adapterA, 1, 2))
        self.hyperAdapterB = nn.Parameter(adapterB)
        
    def forward(self, x):
        bsize = x.shape[0]
        orthA = self.orthA.T.repeat(bsize, 1, 1)
        orthB = self.orthB.repeat(bsize, 1, 1)
        _A = torch.bmm(orthA, self.hyperAdapterA)
        _B = torch.bmm(self.hyperAdapterB, orthB)
        out = x @ (self.scaling * (self.w0 + _A @ _B))
        return out 

    def train(self, mode=True):
        super().train(mode)
        # Only adapterA and adapterB are Parameters and will be affected by mode
        # orthA, orthB, and w0 are buffers and won't be affected
        return self


r = 1
alpha = 0.1
dropout = 0.1
n_transformer_layers = 1
n_transformer_heads = 2
d_emb = 64
d_model = 16
a_b = 2 
kvq =3
#this is to produce a layer at a time
output_dim = (r *  d_model ) * a_b  * kvq   
a = 16 


# Load the datasets
sst2 = load_dataset("nyu-mll/glue", "sst2")
mrpc = load_dataset("nyu-mll/glue", "mrpc")

# Function to process MRPC dataset
def process_mrpc(example):
    return {
        "text": example["sentence1"] + " " + example["sentence2"],
        "label": example["label"],
        "dataset": "mrpc"
    }

# Process MRPC dataset (only the 'train' split)
mrpc_processed = mrpc['train'].map(process_mrpc)

# Function to process SST2 dataset
def process_sst2(example):
    return {
        "text": example["sentence"],
        "label": example["label"],
        "dataset": "sst2"
    }

# Process SST2 dataset (only the 'train' split)
sst2_processed = sst2['train'].map(process_sst2)

# Convert both datasets to pandas DataFrames
df_mrpc = mrpc_processed.to_pandas()
df_sst2 = sst2_processed.to_pandas()

# Calculate the number of samples to keep for SST2
n_mrpc = len(df_mrpc)
n_sst2_to_keep = int(n_mrpc * 1.5)  # 60% SST2, 40% MRPC

# Downsample SST2
df_sst2_downsampled = df_sst2.sample(n=n_sst2_to_keep, random_state=42)

# Combine the DataFrames
df_combined = pd.concat([df_mrpc, df_sst2_downsampled], ignore_index=True)

# Ensure all columns have the same dtype
df_combined["text"] = df_combined["text"].astype(str)
df_combined["label"] = df_combined["label"].astype(int)
df_combined["dataset"] = df_combined["dataset"].astype(str)

# Create a stratified train-test split
train_df, test_df = train_test_split(df_combined, test_size=0.2, stratify=df_combined['dataset'], random_state=42)

# Shuffle the datasets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert back to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Print info about the final datasets
print("Train dataset:")
print(train_dataset)
print("\nTest dataset:")
print(test_dataset)

# Verify the distribution of tasks in each split
print("\nTask distribution in train set:")
print(train_df['dataset'].value_counts(normalize=True))
print("\nTask distribution in test set:")
print(test_df['dataset'].value_counts(normalize=True))


#subset the train_dataset to only have text label and dataset 
train_dataset = train_dataset.remove_columns(['idx'])
test_dataset = test_dataset.remove_columns(['idx'])

train_dataset = train_dataset.remove_columns(['sentence1','sentence2','sentence'])
test_dataset = test_dataset.remove_columns(['sentence1','sentence2','sentence'])

# %%
#laod train_df into a dataloader 


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# %%

# Define the model name
model_name = "bert-base-uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded: {model_name}")
print(f"Model is on device: {device}")

# %%
class_encodings = nn.Parameter(torch.empty(2,64))

nn.init.xavier_normal_(class_encodings)
print('initialized')

# %%

r = 1
alpha = 0.1
dropout = 0.1
n_transformer_layers = 1
n_transformer_heads = 2
d_emb = 64
d_model = 16
a_b = 2 
kvq =3
#this is to produce a layer at a time
output_dim = (r *  d_model ) * a_b  * kvq   
a = 16 

#load bert as the model 
# model = BertModel.from_pretrained('bert-base-uncased',cache_dir  = cache_dir).to(0)
A = model.config.hidden_size

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# Initialize wandb
wandb.init(project="your_project_name", name="your_run_name")

# Define hyperparameters
hyperparameters = {
    "learning_rate": 1e-2,
    "epochs": 1000,
    "batch_size": 32,
    "r": r,
    "alpha": alpha,
    "dropout": dropout,
    "output_dim": output_dim,
    "d_emb": d_emb,
    "d_model": d_model,
    "n_transformer_layers": n_transformer_layers,
    "n_transformer_heads": n_transformer_heads,
    "a": a,
    "A": A
}

wandb.config.update(hyperparameters)

# Set up model, hypernet, and controller
hypernet = HyperNet(r, alpha, dropout, output_dim, d_emb, d_model, n_transformer_layers, n_transformer_heads, a=a).to(0)
controller = HyperNetController(hypernet, target_modules=['query', 'key', 'value'], r=r, A=A, a=a)
controller.augmentLLM(model)
controller.freezeParams(model, False)

# Set up optimizers
model_trainable_params = [param for param in model.parameters() if param.requires_grad]
modelOptimizer = torch.optim.Adam(model_trainable_params, lr=hyperparameters['learning_rate'])
hypernetOptimizer = torch.optim.Adam(hypernet.parameters(), lr=hyperparameters['learning_rate'])

# Evaluation function
def evaluate(model, hypernet, controller, eval_loader, task_mapper, device):
    model.eval()
    hypernet.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for b in eval_loader:
            encodings = tokenizer(b['text'], truncation=True, padding=True, return_tensors='pt')
            labels = b['label'].to(device)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            tasks = [task_mapper[task] for task in b['dataset']]

            task_embedding = torch.stack([class_encodings[task] for task in tasks]).unsqueeze(1).repeat(1,12,1).to(device)
            identity_matrix = torch.eye(d_model).unsqueeze(0).repeat(input_ids.shape[0], 1, 1).to(device)[:,:12,:]
            hypernet_layers = hypernet(task_embedding, identity_matrix).view(input_ids.shape[0], n_layers*kvq, r, -1)
            controller.updateLayers(hypernet_layers)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.shape[0]

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Training loop
task_mapper = {'mrpc': 0, 'sst2': 1}
device = 'cuda'
n_layers = 12
for epoch in range(hyperparameters['epochs']):
    model.train()
    hypernet.train()
    total_train_loss = 0

    for b in tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparameters['epochs']}"):
        modelOptimizer.zero_grad()
        hypernetOptimizer.zero_grad()

        encodings = tokenizer(b['text'], truncation=True, padding=True, return_tensors='pt')
        labels = b['label'].to(device)
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        tasks = [task_mapper[task] for task in b['dataset']]

        task_embedding = torch.stack([class_encodings[task] for task in tasks]).unsqueeze(1).repeat(1,12,1).to(device)
        identity_matrix = torch.eye(d_model).unsqueeze(0).repeat(input_ids.shape[0], 1, 1).to(device)[:,:12,:]
        hypernet_layers = hypernet(task_embedding, identity_matrix).view(input_ids.shape[0], n_layers*kvq, r, -1)
        controller.updateLayers(hypernet_layers)

        loss = model(input_ids, attention_mask=attention_mask, labels=labels).loss
        total_train_loss += loss.item()

        loss.backward()
        modelOptimizer.step()
        hypernetOptimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    
    # Evaluation
    eval_loss, eval_accuracy = evaluate(model, hypernet, controller, test_loader, task_mapper, device)

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy
    })

    print(f"Epoch {epoch+1}/{hyperparameters['epochs']}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Eval Accuracy: {eval_accuracy:.4f}")



