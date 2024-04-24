# # # # # # # # # # # # # # import ray


# # # # # # # # # # # # # # @ray.remote
# # # # # # # # # # # # # # class Actor:
# # # # # # # # # # # # # #     def ping(self):
# # # # # # # # # # # # # #         return "pong"


# # # # # # # # # # # # # # class Fleet():
# # # # # # # # # # # # # #     def __init__(self, num_agents):
# # # # # # # # # # # # # #         self.agents = [Actor.remote() for _ in range(num_agents)]

# # # # # # # # # # # # # #     def ping_agents(self):
# # # # # # # # # # # # # #         return ray.get([agent.ping.remote() for agent in self.agents])

# # # # # # # # # # # # # #     def delete_agent(self, agent_id):
# # # # # # # # # # # # # #         ray.kill(self.agents[agent_id])
# # # # # # # # # # # # # #         # del self.agents[agent_id]


# # # # # # # # # # # # # # fleet = Fleet(4)
# # # # # # # # # # # # # # print(fleet.ping_agents())
# # # # # # # # # # # # # # fleet.delete_agent(0)
# # # # # # # # # # # # # # print(fleet.ping_agents())


# # # # # # # # # # # # # # for i in range(6):
# # # # # # # # # # # # # #     data = datasets[(i // 2) % 3]
# # # # # # # # # # # # # #     print(i, data)

# # # # # # # # # # # # # datasets = ['m', 'k', 'f']
# # # # # # # # # # # # # algos = ['modular', 'mono']
# # # # # # # # # # # # # for i in range(6):
# # # # # # # # # # # # #     dataset_idx = (i // 2) % 3
# # # # # # # # # # # # #     algo_idx = i % 2
# # # # # # # # # # # # #     print(i, datasets[dataset_idx], algos[algo_idx])


# # # # # # # # # # # # import torch
# # # # # # # # # # # # import torch.nn as nn

# # # # # # # # # # # # # Step 1: Define a Linear layer A
# # # # # # # # # # # # A = nn.Linear(in_features=10, out_features=5)
# # # # # # # # # # # # print("Original A weights:", A.weight.data)

# # # # # # # # # # # # # Step 2: Clone A to create B. We clone the state_dict for demonstration.
# # # # # # # # # # # # # B_state_dict = {name: param.clone() for name, param in A.state_dict().items()}
# # # # # # # # # # # # B = nn.Linear(in_features=10, out_features=5)
# # # # # # # # # # # # # B.load_state_dict(B_state_dict)  # Load A's parameters into B
# # # # # # # # # # # # B.load_state_dict(A.state_dict())  # Load A's parameters into B


# # # # # # # # # # # # # Confirm B has the same parameters as A initially
# # # # # # # # # # # # print("Initial B weights:", B.weight.data)

# # # # # # # # # # # # # Step 3: Modify B's parameters (simulate optimization by adding 1 to B's weights)
# # # # # # # # # # # # with torch.no_grad():
# # # # # # # # # # # #     B.weight += 1.0

# # # # # # # # # # # # # Step 4: Check if A's parameters have changed
# # # # # # # # # # # # print("Modified B weights:", B.weight.data)
# # # # # # # # # # # # print("A weights after modifying B:", A.weight.data)


# # # # # # # # # # # pre_or_post = "post"
# # # # # # # # # # # comm_freq = 5
# # # # # # # # # # # num_epochs = 100
# # # # # # # # # # # component_update_freq = 100


# # # # # # # # # # # def train(start_epoch, comm_freq):
# # # # # # # # # # #     print(f"training from epoch {start_epoch} to {start_epoch + comm_freq}")
# # # # # # # # # # #     for i in range(start_epoch, start_epoch + comm_freq):
# # # # # # # # # # #         if (i + 1) % component_update_freq == 0:
# # # # # # # # # # #             update_modules()
# # # # # # # # # # #         else:
# # # # # # # # # # #             update_structure()


# # # # # # # # # # # def update_structure():
# # # # # # # # # # #     print('\t updating structure')


# # # # # # # # # # # def update_modules():
# # # # # # # # # # #     print('\t updating modules')


# # # # # # # # # # # def communicate():
# # # # # # # # # # #     print("! COMMUNICATE")


# # # # # # # # # # # for start_epoch in range(0, num_epochs, comm_freq):
# # # # # # # # # # #     if pre_or_post == "pre":
# # # # # # # # # # #         communicate()

# # # # # # # # # # #     train(start_epoch, comm_freq)
# # # # # # # # # # #     if pre_or_post == "post":
# # # # # # # # # # #         communicate()

# # # # # # # # # # # start = 3
# # # # # # # # # # # end = 9
# # # # # # # # # # # for t in range(start, end):
# # # # # # # # # # #     print(t)


# # # # # # # # # # num_init_tasks = 4
# # # # # # # # # # no_sparse_basis = True

# # # # # # # # # # for j in range(10):
# # # # # # # # # #     if j >= num_init_tasks or not no_sparse_basis:
# # # # # # # # # #         print('dropout at', j)


# # # # # # # # # from shell.datasets.datasets import CustomConcatTensorDataset
# # # # # # # # # import torch

# # # # # # # # # n, n2 = 10, 5
# # # # # # # # # X, Y = torch.rand(n, 3, 3), torch.randint(0, 10, (n,))
# # # # # # # # # X2, Y2 = torch.rand(n2, 3, 3), torch.randint(0, 10, (n2,))
# # # # # # # # # dataset = CustomConcatTensorDataset((X, Y), (X2, Y2))
# # # # # # # # # print(len(dataset))
# # # # # # # # # print(dataset[0])


# # # # # # # # # from shell.utils.utils import seed_everything
# # # # # # # # # import numpy as np

# # # # # # # # # seed_everything(0)

# # # # # # # # # print(np.random.permutation(100))


# # # # # # # # components = list(range(10))
# # # # # # # # num_candidate_modules = 5

# # # # # # # # for idx in range(-num_candidate_modules, 0, 1):
# # # # # # # #     print('idx', idx, 'comp', components[idx])

# # # # # # # import torch


# # # # # # # params = torch.nn.Parameter(torch.rand(3, 3))
# # # # # # # opt = torch.optim.Adam([params])
# # # # # # # # print(list(opt.state.values())[0])

# # # # # # # for epoch in range(5):  # Let's say we train for 10 epochs
# # # # # # #     # Dummy loss function (e.g., mean of all elements in params)
# # # # # # #     loss = params.mean()

# # # # # # #     # Zero gradients (clear previous gradients)
# # # # # # #     opt.zero_grad()

# # # # # # #     # Compute gradients
# # # # # # #     loss.backward()

# # # # # # #     # Perform optimization step
# # # # # # #     opt.step()
# # # # # # #     print('epoch', epoch)
# # # # # # #     print(list(opt.state.values())[0]['exp_avg'].mean())
# # # # # # #     # After the optimization step, print internal states
# # # # # # #     # for param_group in opt.param_groups:
# # # # # # #     #     for param in param_group['params']:
# # # # # # #     #         # Retrieve the Adam state
# # # # # # #     #         state = opt.state[param]
# # # # # # #     #         print(state['exp_avg'].mean())


# # # # # # # algos = ["modular", "monolithic"]
# # # # # # # for i in range(16):
# # # # # # #     seed = i % 8
# # # # # # #     algo = algos[i // 8]
# # # # # # #     print(i, seed, algo)

# # # # # # # num_tasks = 20
# # # # # # # for i in range(num_tasks-1, num_tasks):
# # # # # # #     print(i)


# # # # # a = [5, 1, 3, float('-inf')]
# # # # # k = 10
# # # # # sorted_tasks = sorted(range(len(a)), key=lambda x: (
# # # # #     a[x], -x), reverse=True)
# # # # # chosen = [task for task in sorted_tasks if a[task] != float('-inf')][:k]
# # # # # print('chosen', chosen, 'a of chosen', [a[task] for task in chosen])

# # # # # update_freq = 100

# # # # # for i in range(10):
# # # # #     if i + 1 % update_freq == 0:
# # # # #         print('up module')
# # # # #     else:
# # # # #         print('up struct')
# # # # # # a = [5, 1, 3, float('-inf')]
# # # # # # k = 10
# # # # # # sorted_tasks = sorted(range(len(a)), key=lambda x: (
# # # # # #     a[x], -x), reverse=True)
# # # # # # chosen = [task for task in sorted_tasks if a[task] != float('-inf')][:k]
# # # # # # print('chosen', chosen, 'a of chosen', [a[task] for task in chosen])

# # # # # # for i in range(4):
# # # # # #     print(i, trans_de[i // 2], trans_struct[i % 2])

# # # # # seeds = [0,1,2,3,4,5,6,7]
# # # # # trans_de = [0, 1]
# # # # # trans_struct = [0, 1]
# # # # # for i in range(32):
# # # # #     print(i, trans_de[i // 16], trans_struct[(i // 8) % 2], seeds[i % 8])
# # # # #     # print(i, i//2, i%2, i%8)


# # # # # num_cands = 3
# # # # # num_epochs = 10
# # # # # updates_per_candidate = num_epochs // num_cands
# # # # # no_module_update_freq = num_epochs // updates_per_candidate


# # # # # for i in range(num_epochs):
# # # # #     print('epoch', i)
# # # # #     if i % num_cands == 0:
# # # # #         print('UPDATE MOD')

# # # # # modules = ['m1', 'm2', 'm3', 'm4', 'm5']
# # # # # max=2
# # # # # for i in range(0, len(modules), max):
# # # # #     print(modules[i:i+max], i, i+max)


# # # # comm_freqs=["10" "20" "50" "100"]
# # # # algos=["modular" "monolithic"]

# # # # for i in range(8):
# # # #     print(i, i // 2, i % 2)
# # # #     print(comm_freqs[i // 2], algos[i % 2])


# # # import hashlib
# # # import torch


# # # def _compute_hash(X):
# # #     # Ensure the tensor is on CPU and in a consistent format (e.g., float32) for hashing
# # #     if X.is_cuda:
# # #         X = X.cpu()
# # #     # Flatten the tensor and ensure it's contiguous for consistent hashing
# # #     X = X.contiguous().view(-1)
# # #     return hashlib.sha256(X.numpy().tobytes()).hexdigest()


# # # a = torch.tensor([1.5, 2, 3.0])
# # # b = torch.tensor([1.5, 2, 3.0])
# # # a_h = _compute_hash(a)
# # # b_h = _compute_hash(b)
# # # print(a_h, b_h)
# # # print(a_h == b_h)  # True


# # import torch
# # import torch.nn as nn

# # # Define a simple neural network


# # class ToyModel(nn.Module):
# #     def __init__(self):
# #         super(ToyModel, self).__init__()
# #         self.dense1 = nn.Linear(10, 20)
# #         self.relu = nn.ReLU()
# #         self.dense2 = nn.Linear(20, 5)

# #     def forward(self, x):
# #         x = self.dense1(x)
# #         x = self.relu(x)
# #         x = self.dense2(x)
# #         return x


# # # Create an instance of the model
# # model = ToyModel()

# # # Test if changes to parameters from state_dict() are reflected in named_parameters()
# # for name, param in model.named_parameters():
# #     param_prime = model.state_dict()[name]
# #     print(f"Original {name}:", param[0])

# #     # Modify param_prime and check if param is updated
# #     param_prime += 1.0
# #     print(f"Modified {name} via param_prime:", model.state_dict()[name][0])
# #     print(f"Check if {name} in param updated:", param[0])

# #     # Reset and modify param, then check if reflected in param_prime
# #     param -= 1.0
# #     print(f"Reset and modified {name} via param:", param[0])
# #     print(f"Check if {name} in param_prime updated:",
# #           model.state_dict()[name][0])


# # num_epochs = 100
# # comm_freq = 5
# # for i in range(0, num_epochs, comm_freq):
# #     print('training from', i, 'to', i + comm_freq)


# import math


# comm_freqs = {
#     'data': 5,  # every 5 epochs
#     'gradient': 16,  # every 16 epochs
#     # 'modmod': 100,  # every 100 epochs
# }


# num_epochs = 100


# # Create a combined list of all unique communication epochs, including the last epoch
# unique_epochs = set()
# for freq in comm_freqs.values():
#     unique_epochs.update(range(freq, num_epochs + 1, freq))
# unique_epochs.add(num_epochs)  # Ensure the last epoch is always included
# sorted_epochs = sorted(unique_epochs)


# print(sorted_epochs)

# # max_comm_freq = max(comm_freqs.values())
# # num_coms = math.ceil(num_epochs / max_comm_freq)

# start_epoch = 0
# for end_epoch in sorted_epochs:
#     final = end_epoch == num_epochs
#     print('from', start_epoch, 'to', end_epoch, 'final', final)

# #     if self.sharing_strategy.pre_or_post_comm == "pre":
# #         for strategy, freq in comm_freqs.items():
# #             if end_epoch % freq == 0 and freq <= num_epochs:
# #                 print(f'>>> {strategy.upper()} COMM AT EPOCH', end_epoch)
# #                 self.communicate(task_id, end_epoch, freq, num_epochs, strategy=strategy, final=final)

# #     for agent in self.agents:
# #         agent.set_num_coms(task_id, num_coms)
# #         agent.train(task_id, start_epoch, end_epoch - start_epoch, final=final)

# #     if self.sharing_strategy.pre_or_post_comm == "post":
# #         for strategy, freq in comm_freqs.items():
# #             if end_epoch % freq == 0 and freq <= num_epochs:
# #                 print(f'>>> {strategy.upper()} COMM AT EPOCH', end_epoch)
# #                 self.communicate(task_id, end_epoch, freq, num_epochs, strategy=strategy, final=final)

#     start_epoch = end_epoch


# test = "mod"

# for s in test.split(","):
#     print(s)


freq = 100
num_epochs = 100


unique_epochs = set()

unique_epochs.update(range(freq, num_epochs + 1, freq))
unique_epochs.add(num_epochs)
sorted_epochs = sorted(unique_epochs)

print(unique_epochs)

start_epoch = 0
for end_epoch in sorted_epochs:
    print('from', start_epoch, 'to', end_epoch)
    start_epoch = end_epoch
