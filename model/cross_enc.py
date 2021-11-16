import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import PCModel

class CrossEncoder(PCModel):
    def __init__(self, args, model_name, pooling_method):
        super(CrossEncoder, self).__init__(args, model_name, pooling_method)
        self.W = nn.Linear(args.embed_size, 1, bias=False)

    def forward(self, pair_seq):
        # Embedding
        # (batch_size, seq_len, embd_size)
        y_cands = self.context_encoder(pair_seq)

        # Reducing
        # (batch_size, embd_size)
        y_cand = self.aggregate(y_cands)

        # Dim Reduction
        # (batch_size, 1)
        score = self.W(y_cand)
        
        return score
