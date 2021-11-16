import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import PCModel

class BiEncoder(PCModel):
    def __init__(self, args, model_name, pooling_method):
        super(BiEncoder, self).__init__(args, model_name, pooling_method)

    def forward(self, query, cand):
        # Embedding
        # (batch_size, seq_len, embd_size)
        y_ctxts = self.context_encoder(query)
        y_cands = self.cand_encoder(cand)

        # Reducing
        # (batch_size, embd_size)
        y_ctxt = self.aggregate(y_ctxts)
        y_cand = self.aggregate(y_cands)

        # (batch_size, 1)
        score = torch.bmm(y_ctxt, y_cand.permute(1, 0))
        return score
