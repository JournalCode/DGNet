import torch
import torch.nn as nn
import torch.nn.functional as F
def load_trained_embedding(from_model,to_model):
    """
    :param from_model:
    :param to_model:
    :return: model with trained params
    """
    model_dict = to_model.state_dict()
    state_dict_trained = {name: param for name, param in from_model.named_parameters() if name in model_dict.keys()}
    model_dict.update(state_dict_trained)
    to_model.load_state_dict(model_dict)
    return to_model


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params
