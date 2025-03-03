import torch
import torch.nn.functional as F
import torch.nn as nn


class convNN(nn.Module):

	def __init__(self):

		super(convNN, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
		#64x114x114 after conv1 so 2x2 max pooling
		self.maxPool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, stride= 2 , padding= 1)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3, stride= 2, padding=1)
		#128x29x29
		self.fc1 = nn.Linear(in_features= 4096, out_features= 1024)  
		self.fc2 = nn.Linear(in_features= 1024, out_features= 1024)  
		self.fc3 = nn.Linear(in_features= 1024, out_features= 2)  


		self.input_shape = (3, 227, 227)


	def forward(self, X):
		#print(X.size())
		X = self.conv1(X)
		#print (X.size())
		X = F.relu(X)
		X= self.maxPool(X)
		#print(X.size())
		X=self.conv2(X)
		#print(X.size())
		X = F.relu(X)
		X= self.maxPool(X)
		#print(X.size())
		X=self.conv3(X)
		#print(X.size())
		X = F.relu(X)
		X= self.maxPool(X)
		#print(X.size())
		#X= X.view(-1,4096)
		X=torch.flatten(X,1)
		#print(X.size())
		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)
		X = self.fc3(X)
		return X


# something = torch.rand(size=(1,3,227,227))
# print(something.size())
# model = convNN()
# out = model(something)
# print(out.size())


