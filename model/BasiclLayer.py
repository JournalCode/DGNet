import torch.nn as nn
import torch 
import numpy as np
import torch.nn.functional as F


class BasicCTR(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(BasicCTR, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        raise NotImplementedError


class BasicCL4CTR(nn.Module):
    def __init__(self, field_dims, embed_dim, batch_size=4096, pratio=0.3, fi_type="att"):
        super(BasicCL4CTR, self).__init__()
        # 1、embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.field_dims = field_dims
        self.num_field = len(field_dims)
        self.input_dim = self.num_field * embed_dim
        self.batch_size = batch_size

        self.row, self.col = list(), list()
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                self.row.append(i), self.col.append(j)

        # 2.1 Random mask.
        self.pratio = pratio
        self.dp1 = nn.Dropout(p=pratio)
        self.dp2 = nn.Dropout(p=pratio)

        # 2.2 FI_encoder. In most cases, we utilize three layer transformer layers.
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=1, dim_feedforward=128,
                                                        dropout=0.2)
        
        self.fi_cl = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        # 2.3 Projection
        self.projector1 = nn.Linear(self.input_dim, embed_dim)
        self.projector2 = nn.Linear(self.input_dim, embed_dim)

    def forward(self, x):
        raise NotImplementedError

    def compute_cl_loss(self, x, alpha=1.0, beta=0.01):
        x_emb = self.embedding(x)

        # 1. Compute feature alignment loss (L_ali) and feature uniformity loss (L_uni).
        cl_align_loss = self.compute_alignment_loss(x_emb)
        cl_uniform_loss = self.compute_uniformity_loss(x_emb)
        if alpha == 0.0:
            return (cl_align_loss + cl_uniform_loss) * beta

        # 2. Compute contrastive loss.
        #  equal to random 
        x_emb1, x_emb2 = self.dp1(x_emb), self.dp2(x_emb)
        
        x_h1 = self.fi_cl(x_emb1.transpose(0,1)).view(-1, self.input_dim)  # B,E
        x_h2 = self.fi_cl(x_emb2.transpose(0,1)).view(-1, self.input_dim)  # B,E

        x_h1 = self.projector1(x_h1)
        x_h2 = self.projector2(x_h2)

        cl_loss = torch.norm(x_h1.sub(x_h2), dim=1).pow(2).mean()
        # return cl_loss
        # 3. Combine L_cl and (L_ali + L_uni) with two loss weights (alpha and beta) 
        loss = cl_loss * alpha + (cl_align_loss + cl_uniform_loss) * beta
        return loss

    def compute_cl_loss_(self, x, alpha=1.0, beta=0.01):
        """
        :param x: embedding
        :param alpha:
        :param beta: beta = gamma
        :return: L_cl * alpha + (L_ali+L_uni) * beta

        # This is a simplified computation based only on the embedding of each batch,
        # which can accelerate the training process.f
        """
        x_emb = self.embedding(x)

        # 1. Compute feature alignment loss (L_ali) and feature uniformity loss (L_uni).
        # cl_align_loss = self.compute_alignment_loss(x_emb)
        # cl_uniform_loss = self.compute_uniformity_loss(x_emb)
        # if alpha == 0.0:
        #     return (cl_align_loss + cl_uniform_loss) * beta

        # 2. Compute contrastive loss.
        #  equal to random 
        x_emb1, x_emb2 = self.dp1(x_emb), self.dp2(x_emb)
        
        x_h1 = self.fi_cl(x_emb1.transpose(0,1)).view(-1, self.input_dim)  # B,E
        x_h2 = self.fi_cl(x_emb2.transpose(0,1)).view(-1, self.input_dim)  # B,E

        x_h1 = self.projector1(x_h1)
        x_h2 = self.projector2(x_h2)

        cl_loss = torch.norm(x_h1.sub(x_h2), dim=1).pow(2).mean()
        return cl_loss*alpha
        # 3. Combine L_cl and (L_ali + L_uni) with two loss weights (alpha and beta) 
        # loss = cl_loss * alpha + (cl_align_loss + cl_uniform_loss) * beta
        # return loss

    def compute_cl_loss_all(self, x, alpha=1.0, beta=0.01):
        """
        :param x: embedding
        :param alpha:
        :param beta: beta
        :return: L_cl * alpha + (L_ali+L_uni) * beta

        This is the full version of Cl4CTR, which computes L_ali and L_uni with full feature representations.
        """
        x_emb = self.embedding(x)

        # 1. Compute feature alignment loss (L_ali) and feature uniformity loss (L_uni). 
        cl_align_loss = self.compute_all_alignment_loss() 
        cl_uniform_loss = self.compute_all_uniformity_loss()
        if alpha == 0.0:
            return (cl_align_loss + cl_uniform_loss) * beta

        # 2. Compute contrastive loss (L_cl).
        x_emb1, x_emb2 = self.dp1(x_emb), self.dp2(x_emb)
        
        x_h1 = self.fi_cl(x_emb1.transpose(0,1)).view(-1, self.input_dim)  # B,E
        x_h2 = self.fi_cl(x_emb2.transpose(0,1)).view(-1, self.input_dim)  # B,E

        x_h1 = self.projector1(x_h1)
        x_h2 = self.projector2(x_h2)

        cl_loss = torch.norm(x_h1.sub(x_h2), dim=1).pow(2).mean()
        # return cl_loss * alpha

        # 3. Combine L_cl and (L_ali + L_uni) with two loss weights (alpha and beta)
        loss = cl_loss * alpha + (cl_align_loss + cl_uniform_loss) * beta
        return loss

    def compute_alignment_loss(self, x_emb):
        alignment_loss = torch.norm(x_emb[self.row].sub(x_emb[self.col]), dim=2).pow(2).mean()
        return alignment_loss

    def compute_uniformity_loss(self, x_emb):
        frac = torch.matmul(x_emb, x_emb.transpose(2, 1))  # B,F,F
        denom = torch.matmul(torch.norm(x_emb, dim=2).unsqueeze(2), torch.norm(x_emb, dim=2).unsqueeze(1))  # 64，30,30
        res = torch.div(frac, denom + 1e-4)
        uniformity_loss = res.mean()
        return uniformity_loss

    def compute_all_uniformity_loss(self):
        """
            Calculate field uniformity loss based on all feature representation.
        """
        embedds = self.embedding.embedding.weight
        field_dims = self.field_dims
        field_dims_cum = np.array(object=(0, *np.cumsum(field_dims)))
        field_len = embedds.size()[0]
        field_index = np.array(range(field_len))
        uniformity_loss = 0.0
        #     for i in
        pairs = 0
        for i, (start, end) in enumerate(zip(field_dims_cum[:-1], field_dims_cum[1:])):
            index_f = np.logical_and(field_index >= start, field_index < end)  # 前闭后开
            embed_f = embedds[index_f, :]
            embed_not_f = embedds[~index_f, :]
            frac = torch.matmul(embed_f, embed_not_f.transpose(1, 0))  # f1,f2
            denom = torch.matmul(torch.norm(embed_f, dim=1).unsqueeze(1),
                                 torch.norm(embed_not_f, dim=1).unsqueeze(0))  # f1,f2
            res = torch.div(frac, denom + 1e-4)
            uniformity_loss += res.sum()
            pairs += (field_len - field_dims[i]) * field_dims[i]
        uniformity_loss /= pairs
        return uniformity_loss

    def compute_all_alignment_loss(self):
        """
        Calculate feature alignment loss based on all feature representation.
        """
        embedds = self.embedding.embedding.weight
        field_dims = self.field_dims
        field_dims_cum = np.array((0, *np.cumsum(field_dims)))
        alignment_loss = 0.0
        pairs = 0
        for i, (start, end) in enumerate(zip(field_dims_cum[:-1], field_dims_cum[1:])):
            embed_f = embedds[int(start):int(end), :]
            loss_f = 0.0
            for j in range(field_dims[i]):
                loss_f += torch.norm(embed_f[j, :].sub(embed_f), dim=1).pow(2).sum()
            pairs += field_dims[i] * field_dims[i]
            alignment_loss += loss_f

        alignment_loss /= pairs
        return alignment_loss


class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.float64)

    def forward(self, x):
        """
        :param x: B,F
        :return: B,1
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: B,F,E
        """
        square_of_sum = torch.sum(x, dim=1) ** 2  # B，embed_dim
        sum_of_square = torch.sum(x ** 2, dim=1)  # B，embed_dim
        ix = square_of_sum - sum_of_square  # B,embed_dim
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True) # B，1
        return 0.5 * ix


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        """
        :param field_dims: list
        :param embed_dim
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.float64)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)


class AFM(nn.Module):
    def __init__(self, num_fields, embed_dim, attn_size=16, dropouts=(0.5, 0.5), reduce=True):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts
        self.reduce = reduce
        self.row, self.col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        p, q = x[:, self.row], x[:, self.col]
        inner_product = p * q

        attn_scores = F.relu(self.attention(inner_product))  

        attn_scores = F.softmax(self.projection(attn_scores), dim=1)  # B,nf，1
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0])

        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1])
        if not self.reduce:
            return attn_output
        return self.fc(attn_output) 

class FeaturesLinearWeight(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x, weight=None):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(torch.mul(self.fc(x), weight),dim=1) + self.bias
    
class MLP_Block(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None, 
                 dropout_rates=0.0,
                 batch_norm=False, 
                 bn_only_once=False, # Set True for inference speed up
                 use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        if batch_norm and bn_only_once:
            dense_layers.append(nn.BatchNorm1d(input_dim))
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm and not bn_only_once:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.mlp(inputs)
    

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation