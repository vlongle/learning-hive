def compute_data_pt_size(w=1, h=1, c=1, dataset='mnist'):
    if dataset == "cifar100":
        w = h = 32
        c = 3
    else:
        w = h = 28
        c = 1
    return w * h * c


def compute_receiver_cost(no_queries, no_neighbors, data_pt_size, frequency=1):
    """
    Receiver first send `no_queries`,
    then sender sends back `no_queries * no_neighbors` data points.
    """
    return no_queries * (no_neighbors + 1) * data_pt_size * frequency


def compute_sender_cost(no_data_points, data_pt_size, frequency=1):
    """
    Sender first sends `no_data_points` points (high-dimensional vectors
    like images for each point),
    then receiver sends back `no_data_points` feedback (1D vector)
    """
    return (no_data_points + 1) * data_pt_size * frequency


def compute_mlp_module_size(layer_size=64):
    # import torch.nn as nn
    # def compute_module_size(layer_size):
    #     # Instantiate the module
    #     module = nn.Linear(layer_size, layer_size)
    #     # Compute the number of parameters
    #     num_weights = module.weight.numel()
    #     num_biases = module.bias.numel()

    #     total_parameters = num_weights + num_biases

    #     return total_parameters
    weights = layer_size * layer_size
    bias = layer_size
    return weights + bias


def compute_mlp_model_size(layer_size=64, num_modules=4):
    return compute_mlp_module_size(layer_size) * num_modules

# conv = nn.Conv2d(in_channel=self.channels, out_channel=self.channels,
#                                 self.conv_kernel, padding=self.padding)


def compute_cnn_model_size(in_channels=50, out_channels=50, kernel_size=3, num_modules=4):
    weights = in_channels * out_channels * kernel_size * kernel_size
    bias = out_channels
    return (weights + bias) * num_modules


def compute_fedavg_cost(model_size, frequency=1):
    return model_size * frequency


def compute_fedcurv_cost(model_size, frequency=1):
    return 2 * compute_fedavg_cost(model_size, frequency)


def compute_modular_cost(no_exchanged_components, component_size, frequency=1):
    return no_exchanged_components * component_size * frequency
