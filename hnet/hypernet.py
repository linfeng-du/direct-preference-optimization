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


class PolicyWrapper(nn.Module):
    def __init__(self, hnet, policy):
        super(PolicyWrapper, self).__init__()
        self.hnet = hnet
        self.policy = policy

class HyperNetController():
    def __init__(self,config,hyper_net):
        self.hyper_net = hyper_net
        self.target_modules = config.model.target_modules
        self.hypernet_layers = []
        self.d_emb = hyper_net.d_emb
        self.a = config.hnet.d_a
        self.d_model = config.hnet.d_model
        self.A = config.hnet.d_A 
        self.config= config
        self.first_pass = True
        # self.num_layers = num_layers
        
    def setHyperNetLayer(self,w0):
        self.hypernet_layers.append( HyperNetLinear(self,w0,self.d_emb,self.d_emb,d_model = self.d_model,A = self.A,transpose_AB=self.config.model.transpose_AB if self.config is not None else  False))
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
    def change_forward(self,forward_f = 'adaptor'):
        for l in self.hypernet_layers:
            l.change_forward(forward_f)  
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

                print(f"{key=}")
                parent,target,targe_name = self.get_submodules(key)
                new_module = self.setHyperNetLayer( target)
                self.replace_module(parent, targe_name, new_module, target)
                if verbose:
                    print('replaced',key)

    def setLayerWeights(self,hypernet_outputs, layer):
        assert hypernet_outputs.shape[1] == self.num_layers 
        for l,hyper_out in zip(self.hypernet_layers,hypernet_outputs):
            for k,_ in enumerate(self.target_modules):
                
                l = self.hypernet_layers[l]

    def updateLayers(self,new_layer_tensor,first_pass):
        # params.view(-1,self.n_layers * len(self.target_modules), self.A, self.a * 2)
        #new_layer_tensor is of shape b X (l * kvq)=i X A X a*2 
        for i,l in enumerate(self.hypernet_layers,):
            #l_new is of shape b X A X a*2 
            l_new = new_layer_tensor[:,i]


            a = l_new[:,:,:self.a ]
            if first_pass:
                b = torch.zeros_like(l_new[:,:,self.a: ])

            else:
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
            
            
                
        

# class HyperNet(nn.Module):
#     def __init__(self,  alpha, dropout, d_output,d_emb, d_model,n_transformer_layers,hypernet_heads = 2,target_modules = ['q_proj','k_proj','v_proj'],a=16):
#         super(HyperNet, self).__init__()
#         self.alpha = alpha
#         self.dropout = dropout
#         self.n_transformer_layers = n_transformer_layers
#         self.d_emb = d_emb
#         self.a = a 

        
#         #each attention head is 3*(d_emb * d_emb) as each k,v,h matrix is d_emb * d_emb
#         #We want to use an r-ranked represtation of the d_emb * d_emb matrix so we decompose the total output parameters by r 
#         #We replace d_emb by a singular r rankned vector
#         # Initialize 
#         # encoder layers with user_emb_dim
#         self.target_modules = target_modules
#         encoder_layer = TransformerEncoderLayer(d_model=d_emb, nhead=hypernet_heads, dim_feedforward=d_emb * 4, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
#         #make an mlp to transform the output of the encoder into d_model 
#         self.mlp_encoder = nn.Sequential(
#             nn.Linear(d_emb, d_emb * 2),
#             nn.ReLU(),
#             nn.Linear(d_emb * 2, d_model)
#         )
#         # Initialize transformer decoder layers with d_emb

#         decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=hypernet_heads, dim_feedforward=d_emb * 4, dropout=dropout)
#         self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=n_transformer_layers)
        
#         # MLP to map the decoder output to self.net_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, d_model * 2),
#             nn.ReLU(),
#             nn.Linear(d_model* 2, self.a  * 2 *3 *self.a )
#         )
#         self.init_weights()
#     #xavier normal init 
#     def init_weights(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_normal_(p)
        
#     def forward(self, task_embedding, identity_matrix):
        
#         # Encode the task embedding
#         encoded_task = self.transformer_encoder(task_embedding)
#         encoded_task = self.mlp_encoder(encoded_task)
        
#         # Decode to generate the network parameters
#         decoded_output = self.transformer_decoder(identity_matrix, encoded_task)
        
#         # Map the decoder output to the network parameters using MLP
#         params = self.mlp(decoded_output)
#         # params = self.organize_outputs(params)
#         return params
    
#     def organize_outputs(self, outputs):
#         param_l = []
#         for out in outputs:
#             sub_list = []
#             s = 0 
#             for target in self.target_modules:
                
#                 sub_list.append(out[s:2 *(s+self.r) ])
#                 s = 2* (s + self.a)
#             param_l.append(sub_list)
#         return param_l
            


class HyperNetOneShot(nn.Module):
    # def __init__(self,  config,alpha, dropout,A,a,d_hnet, d_output,d_emb,n_transformer_layers,n_layers,hypernet_heads = 2,target_modules = ['q_proj','k_proj','v_proj'],device = 0):
    def __init__(self,  config):
        super(HyperNet, self).__init__()
        self.alpha = config.hnet.alpha
        self.n_layers = config.hnet.n_layers
        self.dropout = config.hnet.dropout
        self.n_transformer_layers = config.hnet.n_transformer_layers
        self.d_emb = config.hnet.d_emb
        self.a = config.hnet.a 
        self.A = config.hnet.A
        d_model = config.hnet.d_hnet
        self.hypernet_heads = config.hnet.hypernet_heads
        A_B = 2
        self.d_output = config.num_hidden_layers * (config.hnet.d_a *  config.hnet.d_A ) * A_B   *  config.model.adaptors_per_layer

        
        #each attention head is 3*(d_emb * d_emb) as each k,v,h matrix is d_emb * d_emb
        #We want to use an r-ranked represtation of the d_emb * d_emb matrix so we decompose the total output parameters by r 
        #We replace d_emb by a singular r rankned vector
        # Initialize 
        # encoder layers with user_emb_dim
        self.target_modules = config.hnet.target_modules
        self.downsize_mlp = nn.Sequential(
            nn.Linear(self.d_emb, self.d_emb ),
            nn.ReLU(),
            nn.Linear(self.d_emb, d_model))
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=self.hypernet_heads, dim_feedforward=d_model * 4, dropout=config.hnet.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.n_transformer_layers)
        
        #make an mlp to transform the output of the encoder into d_model 
        self.mlp_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2,self.d_output)
        )

        self.init_weights()
    #xavier normal init 
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                #initialize to zeros
                # nn.init.zeros_(p)
        
    def forward(self, task_embedding):
        # if not isinstance(task_embedding,torch.Tensor):

        #     task_embedding = torch.tensor(task_embedding).to(self.device)

        #print the weights of self.transformerencoder 
        # Encode the task embedding
        
        task_embedding = self.downsize_mlp(task_embedding)
        encoded_task = self.transformer_encoder(task_embedding)

        params = self.mlp_encoder(encoded_task)

        
    
        
            
        # params = self.organize_outputs(params)
        #dimsize is Batch X (n_layers * number of target modules) X a X 2)

        return params.view(-1,self.n_layers * len(self.target_modules), self.A, self.a * 2)
    
    def organize_outputs(self, outputs):
        param_l = []
        for out in outputs:
            sub_list = []
            s = 0 
            for target in self.target_modules:
                
                sub_list.append(out[s:2 *(s+self.r) ])
                s = 2* (s + self.a)
            param_l.append(sub_list)
        return param_l
            
            

class HyperNet(nn.Module):
    def __init__(self,  config):
        super(HyperNet, self).__init__()
        self.alpha = config.hnet.alpha
        self.n_layers = config.hnet.n_layers
        self.dropout = config.hnet.dropout
        self.n_transformer_layers = config.hnet.n_transformer_layers
        A_B = 2
        self.d_emb = config.hnet.d_emb
        self.a = config.hnet.d_a 
        self.A = config.hnet.d_A
        d_model = config.hnet.d_hnet
        #positional encodings 
        self.positional_encodings = nn.Parameter(torch.randn(1, config.hnet.n_layers,d_model))

        self.d_output = self.n_layers * (config.hnet.d_a *  config.hnet.d_A ) * A_B   *  config.model.adaptors_per_layer
        
        #each attention head is 3*(d_emb * d_emb) as each k,v,h matrix is d_emb * d_emb
        #We want to use an r-ranked represtation of the d_emb * d_emb matrix so we decompose the total output parameters by r 
        #We replace d_emb by a singular r rankned vector
        # Initialize 
        # encoder layers with user_emb_dim
        self.target_modules = config.model.target_modules
        self.downsize_mlp = nn.Sequential(
            nn.Linear(config.hnet.d_emb, config.hnet.d_emb ),
            nn.ReLU(),
            nn.Linear(config.hnet.d_emb, d_model))
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=config.hnet.n_transformer_heads, dim_feedforward=d_model * 4, dropout=config.hnet.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config.hnet.n_transformer_layers)
        
        #make an mlp to transform the output of the encoder into d_model 

        self.mlp_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2,(self.d_output//self.n_layers))
        )

        self.init_weights()
    #xavier normal init 
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                #initialize to zeros
                # nn.init.zeros_(p)
        
    def forward(self, task_embedding):
        layers = [ ]
        task_embedding = self.downsize_mlp(task_embedding)

        for i in range(self.n_layers):
            
            task_embedding = self.transformer_encoder(task_embedding + self.positional_encodings[:,i])


            layer_weights = self.mlp_encoder(task_embedding)

        
    
            layers.append(layer_weights)
        params = torch.concat(layers)

        # params = self.organize_outputs(params)
        #dimsize is Batch X (n_layers * number of target modules) X a X 2)

        return params.view(-1,self.n_layers * len(self.target_modules), self.A, self.a * 2)
    
    def organize_outputs(self, outputs):
        param_l = []
        for out in outputs:
            sub_list = []
            s = 0 
            for target in self.target_modules:
                
                sub_list.append(out[s:2 *(s+self.r) ])
                s = 2* (s + self.a)
            param_l.append(sub_list)
        return param_l
                        
            
class HyperNetLinear(nn.Linear):
    def __init__(self, hypernet, w0, in_features, out_features, scaling=1, d_model=512, A=16,transpose_AB= False):
        nn.Linear.__init__(self, in_features, out_features)
        self.scaling = scaling
        self.hypernet = hypernet
        self.transpose_AB = transpose_AB
        self.nf = out_features

        #check if w0 is a tensor or a parameter

        if isinstance(w0, nn.Module):
            
            
            self.register_parameter('w0', w0.weight)
            #check if it has a bias 

            if w0.bias is not None:
                self.register_parameter('b0', w0.bias)

        else:
            self.register_buffer('w0', w0.detach())
            self.register_buffer('b0', torch.zeros(out_features))
            
        # self.hyperAdapterA = nn.Parameter(nn.init.xavier_normal_(torch.empty(a, r)).to(w0.device))
        # self.hyperAdapterB = nn.Parameter(nn.init.xavier_normal_(torch.empty(a, r)).to(w0.device))

        # Register orthA and orthB as buffers instead of Parameters
        self.A = A
        #this is needed when number number of query heads is larger than kv heads
        d_model_A =self.w0.shape[0]
        d_model_B = self.w0.shape[1]

        self.register_buffer('orthA', self.make_orth(d_model_A, A).to(self.w0.device))
        self.register_buffer('orthB', self.make_orth(d_model_B, A).to(self.w0.device))

    def make_orth(self, model_dim, A):
        gaus = torch.randn(model_dim, model_dim)
        q, r =  torch.linalg.qr(gaus.to(0).to(torch.float32))
        #check q for otrhonormality 

        return q[:, :A]

    def set_adapter(self, adapterA, adapterB, transposeA=True):
        self.hyperAdapterA = adapterA if not transposeA else torch.transpose(adapterA, 1, 2)
        self.hyperAdapterB = adapterB
        
    def change_forward(self,forward_f = 'adaptor'):
        if forward_f == 'adaptor':
            self.forward = self.forward_adaptor
        elif forward_f == 'base': 
            self.forward = self.forward_no_adaptor
            
    def forward_no_adaptor(self, x):
        size_out = x.size()[:-1] + (self.b0.shape[0],)

        x = x.view(-1, x.size(-1))
        w_prime = torch.addmm(self.b0, x, self.w0)


        return w_prime.view(size_out)
    def forward_adaptor(self, x):
        size_out = x.size()[:-1] + (self.bias.shape[0],)
        bsize = x.shape[0]


        #bsize = 4 you have 4 xs but for each x you have 2ys so your true batch size is bsize *2
        #rejected [[x1,y1],[x2,y2] , [x1,y1'],[x2,y2']] \pi(y1|x1) \pi(y1'|x1)...
        
        #[[prefered],[rejected]]
        
        #prefered
        #bsize*2 , -1 
        #each x will u 
        #transformer encoder -> bsize = 4 * -1
        if bsize //2 == self.hyperAdapterA.shape[0]:
            #dpo code stacks chosen and rejected tensors into one so true batch size is * 2
            self.hyperAdapterA = torch.concat([self.hyperAdapterA,self.hyperAdapterA])
            self.hyperAdapterB = torch.concat([self.hyperAdapterB,self.hyperAdapterB])
        
        self.hyperAdapterA = self.hyperAdapterA.view(bsize,self.A,-1)
        self.hyperAdapterB = self.hyperAdapterB.view(bsize,-1,self.A)

        
        
        orthA = self.orthA.repeat(bsize, 1, 1).view(bsize,-1,self.A)
        orthB = self.orthB.repeat(bsize, 1, 1).view(bsize,self.A,-1)    
        _w0 = self.w0.repeat(bsize, 1, 1)

        
        _A = torch.bmm(orthA, self.hyperAdapterA)

        _B = torch.bmm(self.hyperAdapterB, orthB)
        
        #x shape bsize x seq_len x w0.shape[0] 
        #W' = bsize x w0.shape[0] x w0.shape[1]

        if self.transpose_AB:
            w_prime = (_w0 + (self.scaling/self.A) * (_A @ _B)).transpose(1,2)

            #calculate A@B norm
        else: 
            w_prime = (_w0 + (self.scaling/self.A) * (_A @ _B))
            # print(f"{ (_A @ _B).sum()=}")
        
        out = torch.bmm(x, w_prime)
        out = self.b0 + out

        return out 

    def train(self, mode=True):
        super().train(mode)
        # Only adapterA and adapterB are Parameters and will be affected by mode
        # orthA, orthB, and w0 are buffers and won't be affected
        return self




