import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasiclLayer import FeaturesEmbedding, MultiLayerPerceptron, FactorizationMachine, MultiLayerPerceptron, FeaturesLinear
from model.DEG_ import DEG
from model.models.AFN import LNN
from model.models.DCN import  CrossNetwork
from model.models.DCNv2 import CrossNetV2

class DGNet_DCNv2(nn.Module):
    def __init__(self, 
                field_dims,
                embed_dim,
                bit_layers= 2, 
                mlp_layer= 64, 
                att_size = 32,
                cn_layers=9,
                if_ln = 2,
                c_gate = False, 
                MLP_layer = 3,
                mlp_layers = [400,400,400],
                dropouts = [0.5, 0.5]):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.field_nums = len(field_dims)
        self.num_cross_layers = cn_layers
        self.embed_output_dim =  self.field_nums* embed_dim
        self.c_gate = c_gate
        
        self.cn_dims = [self.embed_output_dim] * self.num_cross_layers
        self.mlp_layers = [self.embed_output_dim] * (MLP_layer+1)
        
        self.gen_two_frn_emb = DEG(self.field_nums, embed_dim, bit_layers=bit_layers, 
                                       mlp_layer=mlp_layer, att_size=att_size)        

        self.cross_nets = nn.ModuleList([CrossNetV2(self.embed_output_dim, cn_layers=1) 
                                         for _ in range(self.num_cross_layers)])
        
        self.cross_dnns = nn.ModuleList([MultiLayerPerceptron(self.mlp_layers[i], [self.mlp_layers[i+1]], 
                                        output_layer=False, dropout= dropouts[0]) 
                                         for i in range(3)])
        
        self.weight_cn = nn.ModuleList([nn.Linear(self.embed_output_dim,self.embed_output_dim, bias=False) 
                                        for i in range(self.num_cross_layers)])
        self.weight_dnn = nn.ModuleList([nn.Linear(self.embed_output_dim,self.embed_output_dim, bias=False) 
                                         for i in range(3)])
       
        self.fc = nn.Linear(2*self.embed_output_dim, 1)
        
        self.if_ln = if_ln 
        self.activation = nn.Sigmoid()
        self.ln_dnn = nn.LayerNorm(self.embed_output_dim)
        self.ln_cn = nn.LayerNorm(self.embed_output_dim)

    def forward(self, input_ids):
        x_emb = self.embedding(input_ids)  # B,F*E
        # 1、DEG
        dnn_emb, cross_emb = self.gen_two_frn_emb(x_emb) # 
        cross_dnn_0, cross_cn_0 = dnn_emb.flatten(start_dim=1), cross_emb.flatten(start_dim=1)
        # 2、 Interaction + GateF
        cross_dnn, cross_cn = cross_dnn_0, cross_cn_0
        for i in range(self.num_cross_layers):
            cross_cn_ = self.cross_nets[i](cross_cn) 
            gate_cn = self.activation(self.weight_cn[i](cross_cn))
            
            if self.if_ln == 0:
                cross_cn = cross_cn_+ cross_cn_0 * gate_cn
            elif self.if_ln == 1:
                cross_cn = self.ln_cn(cross_cn_+ cross_cn_0 * gate_cn)
            elif self.if_ln == 2:
                cross_cn = self.ln_cn(cross_cn_)+ self.ln_cn(cross_cn_0 * gate_cn)   
                        
        for i in range(3):
            cross_dnn_ = self.cross_dnns[i](cross_dnn)
            gate_dnn = self.activation(self.weight_dnn[i](cross_dnn))
            if self.if_ln==0:
                cross_dnn = cross_dnn_ + cross_dnn_0*gate_dnn
            elif self.if_ln==1:
                cross_dnn = self.ln_dnn(cross_dnn_ + cross_dnn_0*gate_dnn)
            elif self.if_ln==2:
                cross_dnn = self.ln_dnn(cross_dnn_) + self.ln_dnn(cross_dnn_0*gate_dnn)            

        y_pred = self.fc(torch.cat([cross_cn, cross_dnn], dim=1))

        return y_pred



class DGNet_DCN(nn.Module):
    def __init__(self, 
                field_dims,
                embed_dim,
                bit_layers= 2, 
                mlp_layer= 64, 
                att_size = 32,
                cn_layers=9,
                if_ln = 2,
                c_gate = False, 
                MLP_layer = 3,
                mlp_layers = [400,400,400],
                dropouts = [0.5, 0.5]):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.field_nums = len(field_dims)
        self.num_cross_layers = cn_layers
        self.embed_output_dim =  self.field_nums* embed_dim
        self.c_gate = c_gate
        
        self.cn_dims = [self.embed_output_dim] * self.num_cross_layers
        self.mlp_layers = [self.embed_output_dim] * (MLP_layer+1)
        
        self.gen_two_frn_emb = DEG(self.field_nums, embed_dim, bit_layers=bit_layers, 
                                       mlp_layer=mlp_layer, att_size=att_size)        

        self.cross_nets = nn.ModuleList([CrossNetwork(self.embed_output_dim, cn_layers=1) 
                                         for _ in range(self.num_cross_layers)])
        
        self.cross_dnns = nn.ModuleList([MultiLayerPerceptron(self.mlp_layers[i], [self.mlp_layers[i+1]], 
                                        output_layer=False, dropout= dropouts[0]) 
                                         for i in range(3)])
        
        self.weight_cn = nn.ModuleList([nn.Linear(self.embed_output_dim,self.embed_output_dim, bias=False) 
                                        for i in range(self.num_cross_layers)])
        self.weight_dnn = nn.ModuleList([nn.Linear(self.embed_output_dim,self.embed_output_dim, bias=False) 
                                         for i in range(3)])
       
        self.fc = nn.Linear(2*self.embed_output_dim, 1)
        
        self.if_ln = if_ln 
        self.activation = nn.Sigmoid()
        self.ln_dnn = nn.LayerNorm(self.embed_output_dim)
        self.ln_cn = nn.LayerNorm(self.embed_output_dim)

    def forward(self, input_ids):
        x_emb = self.embedding(input_ids)  # B,F*E
        # 1、DEG
        dnn_emb, cross_emb = self.gen_two_frn_emb(x_emb) # 
        cross_dnn_0, cross_cn_0 = dnn_emb.flatten(start_dim=1), cross_emb.flatten(start_dim=1)
        # 2、 Interaction + GateF
        cross_dnn, cross_cn = cross_dnn_0, cross_cn_0
        for i in range(self.num_cross_layers):
            cross_cn_ = self.cross_nets[i](cross_cn) 
            gate_cn = self.activation(self.weight_cn[i](cross_cn))
            
            if self.if_ln == 0:
                cross_cn = cross_cn_+ cross_cn_0 * gate_cn
            elif self.if_ln == 1:
                cross_cn = self.ln_cn(cross_cn_+ cross_cn_0 * gate_cn)
            elif self.if_ln == 2:
                cross_cn = self.ln_cn(cross_cn_)+ self.ln_cn(cross_cn_0 * gate_cn)   
                        
        for i in range(3):
            cross_dnn_ = self.cross_dnns[i](cross_dnn)
            gate_dnn = self.activation(self.weight_dnn[i](cross_dnn))
            if self.if_ln==0:
                cross_dnn = cross_dnn_ + cross_dnn_0*gate_dnn
            elif self.if_ln==1:
                cross_dnn = self.ln_dnn(cross_dnn_ + cross_dnn_0*gate_dnn)
            elif self.if_ln==2:
                cross_dnn = self.ln_dnn(cross_dnn_) + self.ln_dnn(cross_dnn_0*gate_dnn)            

        y_pred = self.fc(torch.cat([cross_cn, cross_dnn], dim=1))

        return y_pred
    


class DEG_AFNPlus(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, LNN_dim=100, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400), dropouts=(0.5, 0.5), bit_layers=2, mlp_layer=32,att_size=64):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Embedding

        self.gen_two_frn_emb = DEG(self.num_fields, embed_dim, bit_layers=bit_layers, 
                                       mlp_layer=mlp_layer, att_size=att_size)      

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1], output_layer=True)


    def forward(self, x):
        x_emb = self.embedding(x)
        dnn_emb, cross_emb = self.gen_two_frn_emb(x_emb)

        lnn_out = self.LNN(cross_emb)
        x_lnn = self.mlp(lnn_out)

        x_dnn = self.mlp2(dnn_emb.view(-1, self.embed_output_dim))

        pred_y = x_dnn + x_lnn

        return pred_y
    

class DEG_DFM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5,
                 bit_layers=2, mlp_layer=32,att_size=64):
        super(DEG_DFM, self).__init__(field_dims, embed_dim)
        
        self.lr = FeaturesLinear(field_dims=field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.gen_two_frn_emb = DEG(self.num_fields, embed_dim, bit_layers=bit_layers, 
                                       mlp_layer=mlp_layer, att_size=att_size)      
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        dnn_emb, cross_emb = self.gen_two_frn_emb(x_embed)

        x_out = self.lr(x) + self.fm(dnn_emb) + self.mlp(cross_emb.view(x.size(0), -1))
        return x_out
