import logging
# TODO: maybe implement as a loss
from shell.fleet.utils.model_sharing_utils import is_in


def model_L2(model1, model2, verbose=False, excluded_params=None):
    if excluded_params is None:
        excluded_params = set()
    proximal_term = 0.0
    # for w, w_t in zip(model1.parameters(), model2.parameters()):
    #     proximal_term += (w - w_t).norm(2)
    for n, w in model1.named_parameters():
        if n in model2.state_dict():
            if is_in(n, excluded_params):
                continue
            w_t = model2.state_dict()[n]
            if w.shape == w_t.shape:
                diff = (w - w_t).norm(2)
                proximal_term += diff
                if verbose:
                    # logging.info(f"model_L2: {n} {diff}")
                    print(f"model_L2: {n} {diff}")

    return proximal_term


def compute_fedprox_aux_loss(local_model, global_model, mu, excluded_params=None):
    aux_loss = (mu / 2.0) * model_L2(local_model,
                                     global_model, excluded_params=excluded_params)
    return aux_loss
