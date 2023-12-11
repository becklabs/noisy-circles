import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=6, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=6, stride=1, padding=1)

        self.fc1 = nn.Linear(3 * 3 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 16)
        self.fc5 = nn.Linear(16, 3)

        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv3(x)))
        x = self.max_pool(self.relu(self.conv4(x)))
        x = x.view(-1, 3 * 3 * 128)

        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x