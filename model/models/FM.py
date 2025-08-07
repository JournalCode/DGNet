import torch
import torch.nn as nn
from model.BasiclLayer import BasicCTR, FactorizationMachine, FeaturesLinear

class FM(BasicCTR):
    def __init__(self, field_dims, embed_dim):
        super(FM, self).__init__(field_dims, embed_dim)
        self.lr = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        emb_x = self.embedding(x)
        x = self.lr(x) + self.fm(emb_x)
        return x