import torch

class DemoModel(torch.nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.conv1_1c = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.conv1_3c = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 128, 3, 1, 1)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.conv1_1c(x)
        else:
            x = self.conv1_3c(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        # print(x.shape)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    
class DemoModel_bayes(torch.nn.Module):
    def __init__(self, p1=0.5, p2=0.5):
        super(DemoModel_bayes, self).__init__()
        self.conv1_1c = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv1_3c = torch.nn.Conv2d(3, 32, 3, 1)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 128, 3, 1)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.dropout1 = torch.nn.Dropout(p1)
        self.dropout2 = torch.nn.Dropout(p2)

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.conv1_1c(x)
        else:
            x = self.conv1_3c(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    

from multinomial import Dropout
    
class DemoModel_multi(torch.nn.Module):
    def __init__(self, p1=0.5, p2=0.5):
        super(DemoModel_multi, self).__init__()
        self.conv1_1c = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.conv1_3c = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 128, 3, 1, 1)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.dropout1 = Dropout(p1)
        self.dropout2 = Dropout(p2)

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.conv1_1c(x)
        else:
            x = self.conv1_3c(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DemoModel_bayes(0.5, 0.5)
    x = torch.randn(1, 1, 28, 28)
    print(model(x).shape)


