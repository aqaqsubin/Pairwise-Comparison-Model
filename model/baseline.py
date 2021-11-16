import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class BertEmbeddings(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.pad_token_id = kwargs['pad_token_id']

    def forward(self, tok_ids):
        attn_mask = torch.zeros_like(tok_ids).type_as(tok_ids)
        attn_mask[tok_ids!=self.pad_token_id] = 1
        output = self.model(tok_ids, attn_mask, torch.zeros(tok_ids.size(), dtype=torch.int).type_as(tok_ids))

        return output['last_hidden_state']

# Pairwise Comparison Model
class PCModel(nn.Module):
    def __init__(self, args, model_name, pooling_method):
        super(PCModel, self).__init__()
        self.args = args
        self.context_encoder = BertEmbeddings(model_name=model_name, pad_token_id=1)
        self.cand_encoder = BertEmbeddings(model_name=model_name, pad_token_id=1)
        self.pooling_method = pooling_method

    def aggregate(self, outputs):
        r"""
            pooling: first or avg
            outputs: (batch_size, N, embed_size)
        """
        if self.pooling_method == 'first':
            output =  outputs[:, 0, :]
        elif self.pooling_method =='avg':
            output = torch.mean(outputs, dim=1)
        else:
            raise NotImplementedError(f'Not Implemented Operation : {self.method}')

        return output


