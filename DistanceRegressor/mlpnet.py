# imports
import torch.nn as nn

# define our MLP model
class DistanceRegressor(nn.Module):
    def __init__(self, n_features):
        super(DistanceRegressor, self).__init__()
        
        # define our fully connected layers
        self.input = nn.Linear(n_features, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.output = nn.Linear(2048, 1)

        # define batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(2048)

        # define activation layers
        self.relu = nn.ReLU(inplace=True)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.input(x)
        x = self.relu(self.bn1(x))
        x = self.fc1(x)
        x = self.relu(self.bn2(x))
        x = self.fc2(x)
        x = self.relu(self.bn3(x))
        x = self.fc3(x)
        x = self.relu(self.bn4(x))
        x = self.fc4(x)
        x = self.relu(self.bn5(x))
        x = self.output(x)
        x = self.activation(x)
        return x