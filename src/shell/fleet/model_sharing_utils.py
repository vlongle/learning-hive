'''
File: /model_sharing_utils.py
Project: fleet
Created Date: Monday March 27th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import torch


def is_in(string, exclude_set):
    """
    return true if the string partially matches any keyword in exclude_set
    e.g. string = "decoder.0.weight", exclude_set = ["decoder"] => True
    """
    for exclude in exclude_set:
        if exclude in string:
            return True
    return False


def exclude_model(model_state_dict, excluded_params):
    # remove the task-specific parameters
    to_excludes = [name for name in model_state_dict.keys(
    ) if is_in(name, excluded_params)]

    # remove to_excludes from model
    for name in to_excludes:
        # print("Popping", name)
        model_state_dict.pop(name)
    return model_state_dict


def diff_models(modelA_statedict, modelB_statedict, keys=None):
    diffs = {}
    # compute the average difference between two models
    for key in modelA_statedict.keys():
        if keys is not None and not is_in(key, keys):
            continue
        if key in modelB_statedict.keys():
            diffs[key] = torch.mean(
                torch.abs(modelA_statedict[key] - modelB_statedict[key])).item()
    return diffs
