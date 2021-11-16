import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning.core.lightning import LightningModule
from transformers.optimization import AdamW

from dataloader import PCData
from model.poly_enc import PolyEncoder
from model.bi_enc import BiEncoder
from model.cross_enc import CrossEncoder

class LightningPCModel(LightningModule):
    def __init__(self, hparams, tokenizer, **kwargs):
        super(LightningPCModel, self).__init__()
        self.hparams = hparams
        self.tok = tokenizer

        if self.hparams.model_type == 'poly':
            self.pc_model = PolyEncoder(self.hparams, model_name=self.hparams.pretrained_model, pooling=self.hparams.pooling_method)
        elif self.hparams.model_type == 'bi':
            self.pc_model = BiEncoder(self.hparams, model_name=self.hparams.pretrained_model, pooling=self.hparams.pooling_method)
        elif self.hparams.model_type == 'cross':
            self.pc_model = CrossEncoder(self.hparams, model_name=self.hparams.pretrained_model, pooling=self.hparams.pooling_method)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_len',
                            type=int,
                            default=128)
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        return parser

    def forward(self, query, cand):
        score = self.pc_model(query, cand)
        return score

    def training_step(self, batch, batch_idx):
        query, cands, label = batch    

        scores = []
        for i in range(self.hparams.cand_size):
            cand_i = cands[:, i, :]
            # (batch_size, 1)
            score = self(query, cand_i)
            scores.append(score.squeeze(1))

        scores = torch.stack(scores, dim=1)
        train_loss = self.criterion(scores, label)

        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss


    def validation_step(self, batch, batch_idx):
        query, cands, label = batch    

        scores = []
        for i in range(self.hparams.cand_size):
            cand_i = cands[:, i, :]
            # (batch_size, 1)
            score = self(query, cand_i)
            scores.append(score.squeeze(1))

        scores = torch.stack(scores, dim=1)
        val_loss = self.criterion(scores, label)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_losses = []
        for loss_avg in outputs:
            avg_losses.append(loss_avg)
        self.log('val_loss', torch.stack(avg_losses).mean(), prog_bar=True)
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)

        # Cosine Annealing Learning Rate
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=0
        )
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_annealing_learning_rate',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        query = [item[0] for item in batch]
        cands = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(query), torch.LongTensor(cands), torch.LongTensor(label)

    def train_dataloader(self):
        self.train_set = PCData(data_path=f"{self.hparams.data_dir}/train.csv", tokenizer=self.tok, cand_size=self.hparams.cand_size, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def val_dataloader(self):
        self.valid_set = PCData(data_path=f"{self.hparams.data_dir}/val.csv", tokenizer=self.tok, cand_size=self.hparams.cand_size, max_len=self.hparams.max_len)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return val_dataloader
    

