import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import PCModel

class PolyEncoder(PCModel):
    def __init__(self, args, model_name, pooling_method):
        super(PolyEncoder, self).__init__(args, model_name, pooling_method)
        

    def dot_product_attention(self, query, key, value, mask=None):
        r'''
            query: (batch_size, 1, Q)
            key: (batch_size, seq_len, K)
            value: (batch_size, seq_len, V)
        '''
        
        # (batch_size, 1, seq_len)
        matmul_qk = torch.bmm(query, key.permute(0, 2, 1))  
        if mask is not None:
            matmul_qk += (mask * -1e9)

        attn_weights = F.softmax(matmul_qk, dim=-1)  

        # (batch_size, 1, V)
        output = torch.bmm(attn_weights, value)

        # (batch_size, V)
        return output.squeeze(1)

    def forward(self, query, cand):
        # Embedding
        # (batch_size, seq_len, embd_size)
        y_ctxts = self.context_encoder(query)
        y_cands = self.cand_encoder(cand)

        # Reducing
        # (batch_size, embd_size)
        y_cand = self.aggregate(y_cands)
        
        # Dot Attention
        # (batch_size, embd_size)
        y_ctxt = self.dot_product_attention(y_cand.unsqueeze(1), y_ctxts, y_ctxts, mask=None)

        # (batch_size, 1)
        score = torch.bmm(y_ctxt, y_cand.permute(1, 0))
        return score
