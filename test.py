# num_epochs = 100  # how long to train
# comm_freq = 100  # how often to communicate (for exampple, every 50 epochs)
# # num_coms_per_round = 1


# class Agent:
#     def communicate(self):
#         print('communicate')

#     def train(self, start_epoch, end_epoch):
#         print(f'train {start_epoch}->{end_epoch}')


# agent = Agent()
# for start_epoch in range(0, num_epochs, comm_freq):
#     # start_com_round=(start_epoch // comm_freq) * num_coms_per_round
#     end_epoch = min(start_epoch + comm_freq, num_epochs)
#     agent.train(start_epoch, end_epoch)
#     if comm_freq <= num_epochs and (end_epoch % comm_freq == 0):
#         agent.communicate()
#     # print('start', start_epoch, 'end', end_epoch)


# # import torch


# # a = torch.tensor([1,2,3])

# # for e in a:
# #     print(e)


from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss2/train', np.random.random())
