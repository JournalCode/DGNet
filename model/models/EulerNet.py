import torch
import torch.nn as nn
import torch.nn.functional as F 
from model.BasiclLayer import BasicCTR

class EulerNet(BasicCTR):
    def __init__(self, 
                 field_dims, 
                 embed_dim=10,
                 shape= [3],
                 net_ex_dropout=0.2,
                 net_im_dropout=0.2,
                 layer_norm=True, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 **kwargs):
        super(EulerNet, self).__init__(field_dims, embed_dim)
        # self.embedding_layer = FeaturesEmbedding(field_dims, embed_dim)
        # input_dim = feature_map.sum_emb_out_dim() # 这是什么 
        field_num = len(field_dims)
        
        shape_list = [embed_dim * field_num] + [num_neurons * embed_dim for num_neurons in shape]
        self.reset_parameters()

        interaction_shapes = []
        for inshape, outshape in zip(shape_list[:-1], shape_list[1:]):
            interaction_shapes.append(EulerInteractionLayer(inshape, outshape, embed_dim, layer_norm, net_ex_dropout, net_im_dropout))
        self.Euler_interaction_layers = nn.Sequential(*interaction_shapes)
        self.mu = nn.Parameter(torch.ones(1, field_num, 1))

        self.reg = nn.Linear(shape_list[-1], 1)
        nn.init.xavier_normal_(self.reg.weight)

    def forward(self, input_ids):
        # X = self.get_inputs(inputs)
        feature_emb = self.embedding(input_ids)
        r, p =  self.mu * torch.cos(feature_emb), self.mu *  torch.sin(feature_emb)
        # r, p = r / 4, p / 4     🫱 for large embedding size, you should add a scaling factor.
        o_r, o_p = self.Euler_interaction_layers((r, p))
        o_r, o_p = o_r.reshape(o_r.shape[0], -1), o_p.reshape(o_p.shape[0], -1)
        re, im = self.reg(o_r), self.reg(o_p)
        y_pred = im + re
        return y_pred
        # y_pred = self.output_activation(y_pred)
        # return_dict = {"y_pred": y_pred}
        # return return_dict

class EulerInteractionLayer(nn.Module):
    def __init__(self, inshape, outshape, embedding_dim, apply_norm, net_ex_dropout, net_im_dropout):
        super().__init__()
        self.inshape, self.outshape = int(inshape), int(outshape)
        self.feature_dim = embedding_dim
        self.apply_norm = apply_norm

        # Initial assignment of the order vectors, which significantly affects the training effectiveness of the model.
        # We empirically provide two effective initialization methods here.
        # How to better initialize is still a topic to be further explored.
        # Note: 👆 👆 👆
        if inshape == outshape:
            init_orders = torch.eye(inshape // self.feature_dim, outshape // self.feature_dim)
        else:
            init_orders = torch.softmax(torch.randn(inshape // self.feature_dim, (outshape) // self.feature_dim) / 0.01, dim = 0)
        
        self.inter_orders = nn.Parameter(init_orders)
        self.im = nn.Linear(inshape, outshape)
        #nn.init.normal_(self.im.weight , mean = 0 , std = 0.1) 🫱 Use for large embedding size, e.g. 128 in KKBox
        nn.init.xavier_uniform_(self.im.weight)

        self.bias_lam = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)
        self.bias_theta = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)

        self.drop_ex = nn.Dropout(p = net_ex_dropout)
        self.drop_im = nn.Dropout(p = net_im_dropout)
        self.norm_r = nn.LayerNorm([self.feature_dim])
        self.norm_p = nn.LayerNorm([self.feature_dim])

    def forward(self, complex_features):
        r, p = complex_features
        lam = r ** 2 + p ** 2 + 1e-8
        theta = torch.atan2(p, r)
        lam, theta = lam.reshape(lam.shape[0], -1, self.feature_dim), theta.reshape(theta.shape[0], -1, self.feature_dim)
        lam = 0.5 * torch.log(lam)
        lam, theta = self.drop_ex(lam), self.drop_ex(theta)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)
        lam, theta =  lam @ (self.inter_orders) + self.bias_lam,  theta @ (self.inter_orders) + self.bias_theta
        lam = torch.exp(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)

        r, p = r.reshape(r.shape[0], -1), p.reshape(p.shape[0], -1)
        r, p = self.drop_im(r), self.drop_im(p)
        r, p = self.im(r), self.im(p)
        r, p = torch.relu(r), torch.relu(p)
        r, p = r.reshape(r.shape[0], -1, self.feature_dim), p.reshape(p.shape[0], -1, self.feature_dim)
        
        o_r, o_p =  r + lam * torch.cos(theta), p + lam * torch.sin(theta)
        o_r, o_p = o_r.reshape(o_r.shape[0], -1, self.feature_dim), o_p.reshape(o_p.shape[0], -1, self.feature_dim)
        if self.apply_norm:
            o_r, o_p = self.norm_r(o_r), self.norm_p(o_p)

        return o_r, o_p