import torch
import torch.nn as nn

from model.BasiclLayer import FeaturesEmbedding, MultiLayerPerceptron
class DCNV2P(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCNV2P, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetV2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x).view(-1, self.embed_output_dim)  # B,F*E
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)

        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y

class CrossNetV2(nn.Module):
    def __init__(self, input_dim, cn_layers):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x)
            x = x0 * (xw + self.b[i]) + x
        return x