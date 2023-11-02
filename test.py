# num_epochs = 100
# comm_freq = 100
# #num_epochs = 10
# #comm_freq = 4
# num_coms_per_round = 1

# for start_epoch in range(0, num_epochs, comm_freq):
# 	start_com_round=(start_epoch // comm_freq) * num_coms_per_round
# 	end_epoch = min(start_epoch + comm_freq, num_epochs)
# 	print('epoch', start_epoch, '->', end_epoch, 'comm_round', start_com_round)
# 	if start_epoch + comm_freq >= num_epochs:
# 		print(start_epoch, 'END')
	


# class A:
#     def __init__(self) -> None:
#         print("A")

# class B:
# 	def __init__(self) -> None:
# 		print("B")
	
# 	def doB(self):
# 		print("doB")

# class Child(A, B):
# 	def __init__(self) -> None:
# 		print("child")
# 		super().__init__()
# 		self.doB()


# c = Child()


# class A:
#     def __init__(self, a, b):
#         print(a, b)


# c = (5, 6)

# a = A(*c)


datasets = ["M", "K", "F"]
seeds = [0, 1]

jobs = [0,1,2,3,4,5]

for job_id in jobs:
    d = datasets[job_id % 3]
    seed = seeds[job_id // 3]
    print(f"running --d {d} --seed {seed}")
