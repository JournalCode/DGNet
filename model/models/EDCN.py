import torch
import torch.nn as nn
from model.BasiclLayer import BasicCTR, MLP_Block

class EDCN(BasicCTR):
    def __init__(self, 
                 field_dims, 
                 embed_dim,
                 num_cross_layers=3,
                 hidden_activations="Relu",
                 bridge_type="pointwise_addition",
                 temperature=1,
                 net_dropout=0.5,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 ):
        super().__init__(field_dims, embed_dim)
        self.hidden_dim = len(field_dims) * embed_dim 
        self.dense_layers = nn.ModuleList([MLP_Block(input_dim=self.hidden_dim, 
                                                     output_dim=None, 
                                                     hidden_units=[self.hidden_dim],
                                                     hidden_activations=hidden_activations,
                                                     output_activation=None,
                                                     dropout_rates=net_dropout, 
                                                     batch_norm=False) \
                                           for _ in range(num_cross_layers)])
        self.cross_layers = nn.ModuleList([CrossInteraction(self.hidden_dim) 
                                           for _ in range(num_cross_layers)])
        
        self.bridge_modules = nn.ModuleList([BridgeModule(self.hidden_dim, bridge_type) 
                                             for _ in range(num_cross_layers)])
        
        self.regulation_modules = nn.ModuleList([RegulationModule(len(field_dims), 
                                                                  embed_dim,
                                                                  tau=temperature,
                                                                  use_bn=batch_norm) \
                                                 for _ in range(num_cross_layers)])
        self.fc = nn.Linear(self.hidden_dim * 3, 1) 
        self.reset_parameters()

    def forward(self, input_ids):
        x_emb = self.embedding(input_ids)
        cross_i, deep_i = self.regulation_modules[0](x_emb.flatten(start_dim=1))
        cross_0 = cross_i
        for i in range(len(self.cross_layers)):
            cross_i = self.cross_layers[i](cross_0, cross_i)
            deep_i = self.dense_layers[i](deep_i)
            bridge_i = self.bridge_modules[i](cross_i, deep_i)
            if i + 1 < len(self.cross_layers):
                cross_i, deep_i = self.regulation_modules[i + 1](bridge_i) 
        y_pred = self.fc(torch.cat([cross_i, deep_i, bridge_i], dim=-1))
        return y_pred
    

class BridgeModule(nn.Module):
    def __init__(self, hidden_dim, bridge_type="hadamard_product"):
        super(BridgeModule, self).__init__()
        assert bridge_type in ["hadamard_product", "pointwise_addition", "concatenation", "attention_pooling"],\
               "bridge_type={} is not supported.".format(bridge_type)
        self.bridge_type = bridge_type
        if bridge_type == "concatenation":
            self.concat_pooling = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), 
                                                nn.ReLU())
        elif bridge_type == "attention_pooling":
            self.attention1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.Softmax(dim=-1))
            self.attention2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.Softmax(dim=-1))
    
    def forward(self, X1, X2):
        out = None
        if self.bridge_type == "hadamard_product":
            out = X1 * X2
        elif self.bridge_type == "pointwise_addition":
            out = X1 + X2
        elif self.bridge_type == "concatenation":
            out = self.concat_pooling(torch.cat([X1, X2], dim=-1))
        elif self.bridge_type == "attention_pooling":
            out = self.attention1(X1) * X1 + self.attention1(X2) * X2
        return out
            

class RegulationModule(nn.Module):
    def __init__(self, num_fields, embedding_dim, tau=1, use_bn=False):
        super(RegulationModule, self).__init__()
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.use_bn = use_bn
        self.g1 = nn.Parameter(torch.ones(num_fields))
        self.g2 = nn.Parameter(torch.ones(num_fields))
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(num_fields * embedding_dim)
            self.bn2 = nn.BatchNorm1d(num_fields * embedding_dim)
    
    def forward(self, X):
        g1 = (self.g1 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).view(1, -1)
        g2 = (self.g2 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).view(1, -1)
        out1, out2 = g1 * X, g2 * X
        if self.use_bn:
            out1, out2 = self.bn1(out1), self.bn2(out2)
        return out1, out2


class CrossInteraction(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteraction, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interact_out = self.weight(X_i) * X_0 + self.bias + X_i
        return interact_out

class MultiLayerPerceptron_(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims: 
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        # 使用 *，
        self.mlp = torch.nn.Sequential(*layers) 
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)
        
        

        
