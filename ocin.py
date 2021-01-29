# Bryce Tuppurainen's personally written MNIST network model based on pyTorch
# OCIN (Obscure Character Identification Network)
# To run this on the majority of linux machines run the following command in a bash terminal: pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html && pip3 install matplotlib
# Also HUGE props to sentdex! It's an excellent tutorial as an absolute begginer to pyTorch : https://www.youtube.com/watch?v=i2yPxY2rOzs&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=2

# IMPORT MODULES FOR CONFIG
# Import pyTorch, matplotlib and torchvision (note that the install will be dependant on the CUDA of the GPU in your system usually)
import matplotlib.pyplot as pyplot
import torch
import torchvision
from torchvision import transforms, datasets

# CONFIGURE AND DEBUG DATASET
# The following lines are used for download, and appropriate configuration of the MNIST dataset 

# Pulls the MNIST to the local machine from the torchvision libraries datasets and converts it to a tensor so it can be used to train the network
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Similarly to the train tensor, except used to test against the network
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Batch size (number of test/train values used for each backpropogation) is limited as I found this is optimized for my machine at this value (although its not necessary for this dataset), note that shuffle rearranges the dataset randomly to assist in training precision and realism to other datasets
trainset = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=16, shuffle=True)

# DEBUGGING COMMENTED OUT

#valuesExist = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
#totalValues = 0

# Balancing dataset to ensure that the dataset is approximately an equal value of all outputs
#for data in trainset:
#	xs, ys = data
#	for y in ys:
#		valuesExist[int(y)]+= 1 # y will be equal to the actual value of each number in the set for the duration of this loop, therefore valuesExist[n] will identify the number of occurances of each value n (0, 1, 2, 3, ..., 9).
#		totalValues += 1

#print("Dataset true value distribution (check for balance):")
#for x in valuesExist:
#	print(f"{x}: ~{int(valuesExist[x]/totalValues*100)}%")



#for data in trainset:
	#print(data) # Commented out for now as this isn't required
	#pyplot.imshow(data[0][0].view(28, 28)) # Use the library matplotlib to show the first handwritten image to the user for debugging (note that it was randomised earlier)
	#pyplot.show()
	#break # this is a test, only run this loop once to demonstrate that the first batch has correctly been converted to tensors within tensors comment out the break to check the whole dataset, comment out the whole 

# IMPORT MODULES FOR NETWORK BUILDING
import torch.nn as nn
import torch.nn.functional as functional

# DEFINING THE LAYERS AND NETWORK
class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 64) # There's 748 pixels in the MNIST 28x28 images which act as the input to the first layer which is fully connected (fc) to the 64 nuerons in the first hidden layer
		self.fc2 = nn.Linear(64, 64) # HIDDEN LAYER
		self.fc3 = nn.Linear(64, 64) # HIDDEN LAYER
		self.fc4 = nn.Linear(64, 10) # This is the output layer

	# Defining neuron activation to create a feed forward for data through the network
	def forward(self, datain):
		datain = functional.relu(self.fc1(datain)) # relu is defining the activation function of these neurons as rectified linear in each hidden layer
		datain = functional.relu(self.fc2(datain))
		datain = functional.relu(self.fc3(datain))
		datain = self.fc4(datain)
		return functional.log_softmax(datain, dim=1)

liveNetwork = Network()

# SETTING UP THE OPTIMIZER TO TRAIN THE NETWORK
import torch.optim as optim

optimizer = optim.Adam(liveNetwork.parameters(), lr=0.001, ) # setting up a learning rate of 0.001

epochs = int(input("Number of epochs to run? > ")) # Number of times passing through the whole dataset
print("Beggining training of the MNIST dataset through "+str(epochs)+" epochs!")
for epoch in range(epochs):
	for data in trainset:
		pixels, target = data # Unpacking training data into the greyscale values of the pixels as tensors and scalar target values
		liveNetwork.zero_grad()
		output = liveNetwork(pixels.view(-1, 784))
		loss = functional.nll_loss(output, target) # Compare the scalar value provided as the output of the current network to the real value in order to determine loss at this point in time
		loss.backward() # The magic backpropogation method that pytorch has included in the Adam optimizer, documentation is online
		optimizer.step() # Adjust the bias and weights of the network based on the backpropogation from the previous line
	print("\nEpoch: "+str(epoch+1)+" of "+str(epochs)+" Current loss: "+str(loss.item()))
print("\nFinal loss after "+str(epochs)+" epochs was: "+str(loss.item()))