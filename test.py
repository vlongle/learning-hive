# # # # import ray


# # # # @ray.remote
# # # # class Actor:
# # # #     def ping(self):
# # # #         return "pong"


# # # # class Fleet():
# # # #     def __init__(self, num_agents):
# # # #         self.agents = [Actor.remote() for _ in range(num_agents)]

# # # #     def ping_agents(self):
# # # #         return ray.get([agent.ping.remote() for agent in self.agents])

# # # #     def delete_agent(self, agent_id):
# # # #         ray.kill(self.agents[agent_id])
# # # #         # del self.agents[agent_id]


# # # # fleet = Fleet(4)
# # # # print(fleet.ping_agents())
# # # # fleet.delete_agent(0)
# # # # print(fleet.ping_agents())


# # # # for i in range(6):
# # # #     data = datasets[(i // 2) % 3]
# # # #     print(i, data)

# # # datasets = ['m', 'k', 'f']
# # # algos = ['modular', 'mono']
# # # for i in range(6):
# # #     dataset_idx = (i // 2) % 3
# # #     algo_idx = i % 2
# # #     print(i, datasets[dataset_idx], algos[algo_idx])


# # import torch
# # import torch.nn as nn

# # # Step 1: Define a Linear layer A
# # A = nn.Linear(in_features=10, out_features=5)
# # print("Original A weights:", A.weight.data)

# # # Step 2: Clone A to create B. We clone the state_dict for demonstration.
# # # B_state_dict = {name: param.clone() for name, param in A.state_dict().items()}
# # B = nn.Linear(in_features=10, out_features=5)
# # # B.load_state_dict(B_state_dict)  # Load A's parameters into B
# # B.load_state_dict(A.state_dict())  # Load A's parameters into B


# # # Confirm B has the same parameters as A initially
# # print("Initial B weights:", B.weight.data)

# # # Step 3: Modify B's parameters (simulate optimization by adding 1 to B's weights)
# # with torch.no_grad():
# #     B.weight += 1.0

# # # Step 4: Check if A's parameters have changed
# # print("Modified B weights:", B.weight.data)
# # print("A weights after modifying B:", A.weight.data)


# pre_or_post = "post"
# comm_freq = 5
# num_epochs = 100
# component_update_freq = 100


# def train(start_epoch, comm_freq):
#     print(f"training from epoch {start_epoch} to {start_epoch + comm_freq}")
#     for i in range(start_epoch, start_epoch + comm_freq):
#         if (i + 1) % component_update_freq == 0:
#             update_modules()
#         else:
#             update_structure()


# def update_structure():
#     print('\t updating structure')


# def update_modules():
#     print('\t updating modules')


# def communicate():
#     print("! COMMUNICATE")


# for start_epoch in range(0, num_epochs, comm_freq):
#     if pre_or_post == "pre":
#         communicate()

#     train(start_epoch, comm_freq)
#     if pre_or_post == "post":
#         communicate()

start = 3
end = 9
for t in range(start, end):
    print(t)
