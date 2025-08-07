import torch
import torch.nn as nn
import torch.nn.functional as F

class DEG(nn.Module):
    def __init__(self, field_num, emb_dim, bit_layers=3, mlp_layer=256, att_size=16):
        super(DEG, self).__init__()
        self.emb_frn = MLP2emb(field_num, emb_dim, bit_layers=bit_layers, mlp_layer=mlp_layer)
        self.my_attention_dnn = SelfAttention(emb_dim, att_size=att_size)
        self.my_attention_cn = SelfAttention(emb_dim, att_size=att_size)
        self.acti = nn.Sigmoid()
        
    def forward(self, x_emb):
        dnn_emb, cross_emb = self.emb_frn(x_emb)   # B,F,E 
        dnn_att_emb = self.my_attention_dnn(x_emb) 
        cn_att_emb = self.my_attention_cn(x_emb)  
        dnn_out = self.acti(dnn_emb * dnn_att_emb) * x_emb 
        cn_out = self.acti(cross_emb * cn_att_emb) * x_emb
        return dnn_out, cn_out

class MLP2emb(nn.Module):
    def __init__(self, field_num, emb_dim, bit_layers=1,
                 mlp_layer=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_num = field_num
        self.emb_dim = emb_dim
        input_len = field_num*emb_dim 
        self.input_dim= input_len
        
        mlp_layers = [mlp_layer for _ in range(bit_layers)]
        self.mlps1 = MultiLayerPerceptron_(input_len, embed_dims=mlp_layers)
        self.projection_dnn = nn.Sequential(self.mlps1,
                                            nn.Linear(mlp_layer, emb_dim))
        
        self.mlps2 = MultiLayerPerceptron_(input_len, embed_dims=mlp_layers)
        self.projection_cross = nn.Sequential(self.mlps2, 
                                              nn.Linear(mlp_layer, emb_dim))
        
    def forward(self, x_emb):
        x_emb = x_emb.view(-1, self.input_dim)
        dnn_emb = self.projection_dnn(x_emb).unsqueeze(1) 
        cross_emb = self.projection_cross(x_emb).unsqueeze(1)
        return dnn_emb, cross_emb


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, att_size=16):
        """
        :param embed_dim:
        :param att_size: hyper-parameter, generally we set att_size equal to embed_dim
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.trans_Q = nn.Linear(embed_dim, att_size)
        self.trans_K = nn.Linear(embed_dim, att_size)
        self.trans_V = nn.Linear(embed_dim, att_size)
        self.projection = nn.Linear(att_size, embed_dim)
        self.scale = embed_dim  ** (0.5)

    def forward(self, x_emb, scale=None):
        """
        :param x: B,F,E
        :return: B,F,E
        """
        Q = self.trans_Q(x_emb)
        K = self.trans_K(x_emb)
        V = self.trans_V(x_emb)

        attention = torch.matmul(Q,  K.permute(0, 2, 1))/self.scale # B,F,F
        attention_score = F.softmax(attention, dim=-1)
        context = torch.matmul(attention_score, V)
        context = self.projection(context)
        return context
    

class MultiLayerPerceptron_(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, if_batch=False):
        super().__init__() 
        layers = list()
        for embed_dim in embed_dims: 
            layers.append(torch.nn.Linear(input_dim, embed_dim,bias=False))
            # if if_batch:
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.PReLU()) 
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        self.mlp = torch.nn.Sequential(*layers) 
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)