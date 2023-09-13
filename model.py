import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile

class binary_classifier(nn.Module):
    def __init__(self):
        super(binary_classifier, self).__init__()
        self.model = models.resnet18()
        self.classifier = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # calulate flops and params
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = binary_classifier().to(device)
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))
