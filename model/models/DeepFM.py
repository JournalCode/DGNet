from model.BasiclLayer import BasicCTR, FactorizationMachine, MultiLayerPerceptron, FeaturesLinear
import torch.nn as nn

class DeepFM(BasicCTR):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DeepFM, self).__init__(field_dims, embed_dim)
        self.lr = FeaturesLinear(field_dims=field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0), -1))
        return x_out

