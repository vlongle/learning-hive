import logging
def model_L2(model1, model2):
    proximal_term = 0.0
    for w, w_t in zip(model1.parameters(), model2.parameters()):
        proximal_term += (w - w_t).norm(2)
    return proximal_term

def compute_fedprox_aux_loss(local_model, global_model, mu):
    # aux_loss = (mu / 2.0) * model_L2(local_model, global_model)
    aux_loss = mu * model_L2(local_model, global_model)
    # aux_loss = 0.0
    # logging.info(f"aux_loss: {aux_loss}")
    return aux_loss