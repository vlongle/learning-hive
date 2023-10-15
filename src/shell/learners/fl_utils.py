def model_L2(model1, model2):
    proximal_term = 0.0
    for w, w_t in zip(model1.parameters(), model2.parameters()):
        proximal_term += (w - w_t).norm(2)
    return proximal_term

def compute_fedprox_aux_loss(local_model, global_model, mu):
    return (mu / 2.0) * model_L2(local_model, global_model)