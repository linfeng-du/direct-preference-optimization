
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
import regex as re
os.environ['TRANSFORMERS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['HF_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['HF_DATASETS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['TORCH_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
cache_dir = '/home/mila/e/emiliano.penaloza/scratch/models'



class LoRAHnet_controller():
    def __init__(self,config):
        self.hyper_net = None
        self.target_modules = config.model.target_modules
        self.hypernet_layers = []
        self.r = config.hnet.r
        self.config= config
        # self.num_layers = num_layers
        
    def setLoRA(self,w0):
        fan_in = w0.weight.shape[0]
        fan_out = w0.weight.shape[1]
        self.hypernet_layers.append( LoRAPer(w0,fan_in,fan_out,self.r,self.config))
        return self.hypernet_layers[-1]
    
    def replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

        new_module.weight = nn.Parameter(old_module.weight)
        if old_module.bias is not None:
            new_module.bias = nn.Parameter(old_module.bias)
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
                new_module = self.setLoRA( target)
                self.replace_module(parent, targe_name, new_module, target)
                if verbose:
                    print('replaced',key)


    def setLayerWeights(self,hypernet_outputs, layer):
        assert hypernet_outputs.shape[1] == self.num_layers 
        for l,hyper_out in zip(self.hypernet_layers,hypernet_outputs):
            for k,_ in enumerate(self.target_modules):
                
                l = self.hypernet_layers[l]

            
    def freezeParams(self,model,verbose = True):
        #print trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'LoRA' not in name and 'hnet' not in name: 
                    param.requires_grad = False
                elif verbose:
                    print (name, param.data.shape)
                    
    def updateLayers(self,context: torch.Tensor):
        # params.view(-1,self.n_layers * len(self.target_modules), self.A, self.a * 2)
        #new_layer_tensor is of shape b X (l * kvq)=i X A X a*2 
        for i,l in enumerate(self.hypernet_layers,):
            #l_new is of shape b X A X a*2 
            l.set_context(context)

            
class LoRAHnet(nn.Module):
    def __init__(self,emb_dim,r):
        super(LoRAHnet, self).__init__()
        #make a linear layer that takes emb_dim to r*r dim 
        self.hnet = nn.Linear(emb_dim,r*r)
        self.r = r
        self.__init__params()
    def __init__params(self):
        nn.init.xavier_normal_(self.hnet.weight)
        
    
    def forward(self,x):
        h = self.hnet(x)
        return h.view(-1,self.r,self.r)

class LoRAPer(nn.Linear):
    def __init__(self, w0, in_features, out_features, r, config,scaling=32,transpose_AB= False):

        nn.Linear.__init__(self, in_features, out_features)
        self.scaling = scaling
        self.transpose_AB = transpose_AB
        self.r = r

        self.register_parameter('w0', w0.weight)
        if w0.bias is not None:
                self.register_parameter('b0', w0.bias)



        # self.LoRA_A = nn.Parameter(nn.init.xavier_normal_(torch.empty(a, r)).to(w0.device))
        # self.LoRA_B = nn.Parameter(nn.init.xavier_normal_(torch.empty(a, r)).to(w0.device))

        #this is needed when number number of query heads is larger than kv heads
        fan_in  =self.w0.shape[0]
        fan_out = self.w0.shape[1]

        self.LoRA_A = nn.Parameter(torch.empty(fan_in, r,requires_grad=True)).to(self.w0.device).to(self.w0.dtype)
        self.LoRA_B = nn.Parameter(torch.zeros(r, fan_out,requires_grad=True)).to(self.w0.device).to(self.w0.dtype)
        self.context = torch.ones((r,r))
        self.hnet = LoRAHnet(config.hnet.d_emb,r).to(self.w0.device).to(self.w0.dtype)
        self.__init__params()
    def __init__params(self):

        nn.init.xavier_normal_(self.LoRA_A)

    def set_context(self,context:torch.Tensor):

            self.context = context.to(self.w0.device).to(self.w0.dtype)
        
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
        bsize = x.shape[0]
        
        size_out = x.size()[:-1] + (self.b0.shape[0],)
        
        #b x r x r 
        adaptor = self.hnet(self.context)
        
        #size b x d x r 
        A_ = torch.matmul(self.LoRA_A, adaptor.transpose(1, 2))

        w_delta = self.scaling/self.r * torch.matmul(A_, self.LoRA_B)

        if bsize //2 == w_delta.shape[0]:

            w_delta = w_delta.repeat(2,1,1)

            # w0 = self.w0.repeat(bsize*2,1,1)
            # print(f"{w0.shape=}")
        # w_prime = self.scaling/self.r*(self.LoRA_A @ self.LoRA_B)
        # if  self.transpose_AB:
        #     w_prime = w_prime.transpose(1,2)
        


        # x = x.view(-1, x.size(-1))
        w_prime = self.w0 + w_delta
        out = torch.baddbmm(self.b0, x, w_prime).view(size_out)


        # out = torch.addmm(self.b0, x, w_prime).view(size_out)
        # out.sum().backward()




        return out 
