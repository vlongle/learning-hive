num_epochs = 100
comm_freq = num_epochs
# pre_or_post_comm = "pre"
pre_or_post_comm = "post"


def train(start_epoch, end_epoch):
    print("Training from epoch", start_epoch, "to", end_epoch)


def comm(start_epoch, end_epoch):
    print("Communicating from epoch", start_epoch, "to", end_epoch)


for start_epoch in range(0, num_epochs, comm_freq):
    end_epoch = min(start_epoch + comm_freq, num_epochs)
    if pre_or_post_comm == "pre" and comm_freq <= num_epochs and (end_epoch % comm_freq == 0):
        comm(start_epoch, end_epoch)
    train(start_epoch, end_epoch)
    if pre_or_post_comm == "post" and comm_freq <= num_epochs and (end_epoch % comm_freq == 0):
        comm(start_epoch, end_epoch)
