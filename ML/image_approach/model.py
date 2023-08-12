import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """The Alexnet model
    """
    def __init__(self, num_classes):
        """Constructor

        Parameters
        ----------
        num_classes : int
            the number of output/classes for the classification head
        """
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0)
        self.relu1 = nn.ReLU()
        self.response_Norm1 = nn.LocalResponseNorm(size = 5,k = 2)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5,  stride = 1, padding = 'same')
        self.relu2 = nn.ReLU()
        self.response_Norm2 = nn.LocalResponseNorm(size = 5,k = 2)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv3 =  nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 'same')
        self.relu3 = nn.ReLU()

        self.conv4 =  nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 'same')
        self.relu4 = nn.ReLU()

        self.conv5 =  nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 'same')
        self.relu5 = nn.ReLU()
        self.response_Norm3 = nn.LocalResponseNorm(size = 5,k = 2)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(4096,num_classes)
    
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.response_Norm1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.response_Norm2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.response_Norm3(x)
        x = self.max_pool3(x)

        x = torch.flatten(x, start_dim=1)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x
    
def evaluate(val_dataloader, model, criterion):
    """The evaluation function

    Parameters
    ----------
    val_dataloader : DataLoader
        the dataloader being evaluated/validated
    model : AlexNet model
        The alexnet model being evaluated
    criterion : PyTorch Loss Function
        The loss function

    Returns
    -------
    int, int
        number of true and predicted values
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        trues = []
        preds = []
        for images, labels in val_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            trues.extend(labels)
            preds.extend(labels)
    return trues, preds