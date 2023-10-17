# num_epochs = 100
# comm_freq = 10
num_epochs = 10
comm_freq = 4
num_coms_per_round = 1

for start_epoch in range(0, num_epochs, comm_freq):
	start_com_round=(start_epoch // comm_freq) * num_coms_per_round
	print('epoch', start_epoch, 'comm_round', start_com_round)
	if start_epoch + comm_freq >= num_epochs:
		print(start_epoch, 'END')
	

