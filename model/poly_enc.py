import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import PCModel

class PolyEncoder(PCModel):
    def __init__(self, args, model_name, pooling_method, n_codes=64):
        super(PolyEncoder, self).__init__(args, model_name, pooling_method)
        self.n_codes = n_codes
        
        codes = torch.empty(self.n_codes, args.embed_size)
        codes = torch.nn.init.uniform_(codes)
        self.codes = torch.nn.Parameter(codes)

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
        batch_size = query.size(0)

        # Embedding
        # (batch_size, seq_len, embd_size)
        h = self.context_encoder(query)
        y_cands = self.cand_encoder(cand)

        # Reducing
        # (batch_size, embd_size)
        y_cand = self.aggregate(y_cands)

        # Extracting m Global Features 
        # (batch_size, n_codes, embd_size)
        y_ctxt_i = self.dot_product_attention(self.codes.repeat(batch_size, 1, 1), h, h, mask=None)

        # Context-Candidate Attention
        # (batch_size, embd_size)
        y_ctxt = self.dot_product_attention(y_cand.unsqueeze(1), y_ctxt_i, y_ctxt_i, mask=None)

        # Scoring
        # (batch_size, 1, 1)
        score = torch.bmm(y_ctxt.unsqueeze(1), y_cand.unsqueeze(1).permute(0, 2, 1))
        
        return score.squeeze(1)
