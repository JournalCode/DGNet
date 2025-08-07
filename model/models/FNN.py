import torch
import torch.nn as nn
from model.BasiclLayer  import FeaturesEmbedding, MultiLayerPerceptron


class FNN(nn.Module):
    def __init__(self, field_dims, embed_dim, num_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(FNN, self).__init__()
        mlp_layers = [400] * num_layers
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)


    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.mlp(x_emb.view(x.size(0), -1))
        return pred_y