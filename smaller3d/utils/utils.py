from collections import MutableMapping

import torch
from loguru import logger


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_baseline_model(cfg, model):
    # if it is Minkoski weights
    cfg.model_teacher.in_channels = 3
    cfg.model_teacher.config.conv1_kernel_size = 5

    cfg.model_student.in_channels = 3
    cfg.model_student.config.conv1_kernel_size = 5

    cfg.data.add_normals = False
    cfg.data.train_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.validation_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.test_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.voxel_size = 0.02
    model = model(cfg)
    state_dict = torch.load(cfg.general.checkpoint_teacher)["state_dict"]
    model.model_teacher.load_state_dict(state_dict)

    state_dict = torch.load(cfg.general.checkpoint_student)["state_dict"]
    model.model_student.load_state_dict(state_dict)
    return cfg, model

import os
def load_checkpoint_with_missing_or_exsessive_keys(cfg, model):

    state_dict = torch.load(cfg.general.checkpoint_teacher)["state_dict"]
    correct_dict = dict(model.teacher_model.state_dict())

    # if parametrs not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(key[len("model."):], None) is None:
            logger.warning(f"Key not found, it will be initialized randomly: {key}")

    # if parametrs have different shape, it will randomly initialize
    state_dict = torch.load(cfg.general.checkpoint_teacher)["state_dict"]
    correct_dict = dict(model.teacher_model.state_dict())

    for key in correct_dict.keys():

        if state_dict["model."+key].shape != correct_dict[key].shape:
            logger.warning(
                f"incorrect shape {key}:{state_dict['model.' + key].shape} vs {correct_dict[key].shape}"
            )
            state_dict.update({"model."+key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.teacher_model.state_dict())
    new_state_dict = dict()
    for key in state_dict.keys():
        key_st = key[len("model."):]
        if key_st in correct_dict.keys():
            new_state_dict.update({key_st: state_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    model.teacher_model.load_state_dict(new_state_dict)
    return cfg, model


def freeze_until(net, param_name: str = None):
    """
    Freeze net until param_name
    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD
    Args:
        net:
        param_name:
    Returns:
    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
