
#import model out put from transformers 
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn

class RewardModel(nn.Module):
    def __init__(self, model_instance, num_classes,hnet_dim = None):
        super().__init__()  # Call the parent class's constructor
        
        # Hold a reference to the model instance
        self.model = model_instance
        
        # Assuming the model's configuration attribute is accessible via model.config.hidden_size
        hidden_size = self.model.config.hidden_size
        # Ensure that output_hidden_states is set to True
        self.model.config.output_hidden_states = True

        
        # Define a linear classification head
        self.classification_head = nn.Linear(hidden_size if hnet_dim is None else hidden_size + hnet_dim , num_classes)
        self.__init__weights()
    def __init__weights(self):
        nn.init.xavier_normal_(self.classification_head.weight)
        nn.init.constant_(self.classification_head.bias, 0)
    def forward(self, input_ids, b_len,attention_mask=None, token_type_ids=None,user_emb = None, **kwargs):
        # Call the forward method of the base model
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        
        # Assuming the output we need is the last hidden state

        chosen_logits = outputs['hidden_states'][-1][ b_len:, :]
        rejected_logits = outputs['hidden_states'][-1][:b_len, :]
        cls_chosen_logits = chosen_logits[:, -1, :]
        cls_rejected_logits = rejected_logits[:, -1, :]
        if user_emb is not None:
            cls_chosen_logits = torch.cat([cls_chosen_logits,user_emb],dim = 1)
            cls_rejected_logits = torch.cat([cls_rejected_logits,user_emb],dim = 1)


        chosen_reward = self.classification_head(cls_chosen_logits)       

        rejected_reward = self.classification_head(cls_rejected_logits)
        #make a model output dictionary that can be interacted with by model_output.logits 

        # model_output = SequenceClassifierOutput(logits=final_logits)

        # model_output = {"logits": final_logits}
        return chosen_reward,rejected_reward

