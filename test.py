# import ray


# @ray.remote
# class Actor:
#     def ping(self):
#         return "pong"


# class Fleet():
#     def __init__(self, num_agents):
#         self.agents = [Actor.remote() for _ in range(num_agents)]

#     def ping_agents(self):
#         return ray.get([agent.ping.remote() for agent in self.agents])

#     def delete_agent(self, agent_id):
#         ray.kill(self.agents[agent_id])
#         # del self.agents[agent_id]


# fleet = Fleet(4)
# print(fleet.ping_agents())
# fleet.delete_agent(0)
# print(fleet.ping_agents())


datasets = ['m', 'k', 'f']

for i in range(6):
    data = datasets[(i // 2) % 3]
    print(i, data)
