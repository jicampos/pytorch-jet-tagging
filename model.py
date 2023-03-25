import torch.nn as nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features=16, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=32, bias=True)
        self.fc4 = nn.Linear(in_features=32, out_features=5, bias=True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Implements the forward of the Jet Tagging neural network. 

        Args:
            x (torch.Tensor): Model inputs. 

        Returns:
            torch.Tensor: Model predictions.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.softmax(x)
