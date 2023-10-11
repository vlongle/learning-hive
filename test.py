num_epochs = 100
comm_freq = 20

for start_epoch in range(0, num_epochs, comm_freq):
	print(start_epoch)
	if start_epoch + comm_freq >= num_epochs:
		print(start_epoch, 'END')
	

