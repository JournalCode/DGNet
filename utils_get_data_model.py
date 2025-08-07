from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import math 
sys.path.append("..")
from model import *
from data import get_mltag_loader721, getdataloader_frappe, getdataloader_ml1m

def get_model(model_name, field_dims, config):
    if model_name == "FM":
        return FM(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "DGNet_DCN":
        return DGNet_DCN(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "DGNet_DCNv2":
        return DGNet_DCNv2(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "DEG_AFNPlus":
        return DEG_AFNPlus(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "DEG_DFM":
        return DEG_DFM(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "AFNPlus":
        return AFNPlus(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "AFN":
        return AFN(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "AutoIntPlus":
        return AutoIntPlus(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "AutoInt":
        return AutoInt(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "IPNN":
        return IPNN(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "OPNN":
        return OPNN(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "CNv2":
        return CrossNetV2(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "EDCN":
        return EDCN(field_dims=field_dims, embed_dim=config.embed_dim)
    elif model_name == "FinalMLP":
        return FinalMLP(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "FiBiNet":
        return FiBiNet(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "MaskNet":
        return MaskNet(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "DCNv2":
        return DCNV2P(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "DeepFM":
        return DeepFM(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "NAME":
        return AFN(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "CIN":
        return CIN(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "xDeepFM":
        return xDeepFM(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "FINT":
        return FINT(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "DIFM":
        return DIFM(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "DCN":
        return DCN(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "CN":
        return CN(field_dims=field_dims,embed_dim=config.embed_dim)
    elif model_name == "FMFM":
        return FMFM(field_dims=field_dims,embed_dim=config.embed_dim)


def get_dataset(dataset_name, batch_size):
    dataset_name = dataset_name.lower()
    if dataset_name=="frappe":
        field_dims, trainLoader, validLoader, testLoader = getdataloader_frappe(batch_size=batch_size)
        return field_dims, trainLoader, validLoader, testLoader    
    elif dataset_name == "ml1m":
        field_dims, trainLoader, validLoader, testLoader = getdataloader_ml1m(batch_size=batch_size)
        return field_dims, trainLoader, validLoader, testLoader
    elif dataset_name == "mltag":
        field_dims, trainLoader, validLoader, testLoader = get_mltag_loader721(batch_size=batch_size)
        return field_dims, trainLoader, validLoader, testLoader
    else:
        raise ValueError('unknown dataset name: ' + dataset_name)


@dataclass
class CTRModelArguments:
    bit_layers: int = field(default=1)
    mlp_layer: int=field(default=256)
    att_size: int = field(default=32)
    fusion: str= field(default="add")
    if_ln: int = field(default=0)
    cn_layers :int =field(default=3)
    ctr_model_name: str = field(default="fm")
    embed_size: int = field(default=32)
    type_name: str = field(default="deemb")
    
    embed_dropout_rate: float = field(default=0.0)
    gcn_layers :int = field(default=4)
    cn_layers :int =field(default=3)
    hidden_size: int = field(default=128)
    num_hidden_layers: int = field(default=1)
    bridge_type: str = field(default="gate")
    hidden_act: str = field(default='relu')
    hidden_dropout_rate: float = field(default=0.0)
    num_attn_heads: int = field(default=1)
    attn_probs_dropout_rate: float = field(default=0.1)
    intermediate_size: int = field(default=128)
    norm_first: bool = field(default=False)
    layer_norm_eps: float = field(default=1e-12)
    res_conn: bool = field(default=False)
    output_dim: int = field(default=1)
    num_cross_layers: int = field(default=1)
    share_embedding: bool = field(default=False)
    channels: str = field(default='14,16,18,20')
    kernel_heights: str = field(default='7,7,7,7')
    pooling_sizes: str = field(default='2,2,2,2')
    recombined_channels: str = field(default='3,3,3,3')
    conv_act: str = field(default='tanh')
    reduction_ratio: int = field(default=3)
    bilinear_type: str = field(default='field_interaction')
    reuse_graph_layer: bool = field(default=False)
    attn_scale: bool = field(default=False)
    use_lr: bool = field(default=False)
    attn_size: int = field(default=40)
    num_attn_layers: int = field(default=2)
    cin_layer_units: str = field(default='50,50')
    product_type: str = field(default='inner')
    outer_product_kernel_type: str = field(default='mat')
    dnn_size: int = field(default=1000, metadata={'help': "The size of each dnn layer"})
    num_dnn_layers: int = field(default=0, metadata={"help": "The number of dnn layers"})
    dnn_act: str = field(default='relu', metadata={'help': "The activation function for dnn layers"})
    dnn_drop: float = field(default=0.0, metadata={'help': "The dropout for dnn layers"})

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output
