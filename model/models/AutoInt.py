import torch
import torch.nn.functional as F
import torch.nn as nn

from model.BasiclLayer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron,BasicCTR

class AutoIntPlus(BasicCTR):
    def __init__(self, field_dims, embed_dim, atten_embed_dim=32, num_heads=2,
                 num_layers=3, mlp_dims=(400, 400, 400), dropouts=(0.5,0.5), has_residual=True):
        super().__init__(field_dims, embed_dim)
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        x_emb = self.embedding(x)
        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        
        pred_y = self.linear(x) + self.attn_fc(cross_term) + self.mlp(x_emb.view(-1, self.embed_output_dim))
        return pred_y



class AutoInt(nn.Module):
    def __init__(self, field_dims, embed_dim, atten_embed_dim=16, num_heads=3,
                 attention_layers=3, mlp_dims=(400, 400, 400), dropouts=(0.5,0.5), has_residual=True):
        super(AutoInt, self).__init__() 
        self.embedding= FeaturesEmbedding(field_dims, embed_dim)
        self.lr_layer = FeaturesLinear(field_dims)
        # self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[0])
        
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embed_dim if i == 0 else atten_embed_dim,
                                     attention_dim=atten_embed_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=dropouts[1], 
                                     use_residual=True, 
                                     use_scale=False,
                                     layer_norm=False) \
             for i in range(attention_layers)])
        self.fc = nn.Linear(len(field_dims) * embed_dim, 1)
        # self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        # self.reset_parameters()
        # self.model_to_device()

    def forward(self, input_ids):
        x_emb = self.embedding(input_ids)
        attention_out = self.self_attention(x_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_att = self.fc(attention_out)
        # if self.lr_layer is not None:
        y_pred = self.lr_layer(input_ids) + y_att
        return y_pred 
    


class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        residual = X
        
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention