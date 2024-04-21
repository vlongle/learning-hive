import logging
# TODO: maybe implement as a loss


def model_L2(model1, model2):
    proximal_term = 0.0
    # for w, w_t in zip(model1.parameters(), model2.parameters()):
    #     proximal_term += (w - w_t).norm(2)
    for n, w in model1.named_parameters():
        if n in model2.state_dict():
            w_t = model2.state_dict()[n]
            if w.shape == w_t.shape:
                proximal_term += (w - w_t).norm(2)
    return proximal_term


def compute_fedprox_aux_loss(local_model, global_model, mu):
    aux_loss = (mu / 2.0) * model_L2(local_model, global_model)
    return aux_loss
