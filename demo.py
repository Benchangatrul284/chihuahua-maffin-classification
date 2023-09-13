import torch
import torch.nn as nn
import os
#import your model here
from model import binary_classifier
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def show_results(imgs, labels, preds):
    num_imgs = len(imgs)
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    correct = 0
    for i in range(num_imgs):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(imgs[i].permute(1, 2, 0))
        label = 'maf' if labels[i] == 1 else 'chi'
        pred = 'maf' if preds[i] == 1 else 'chi'
        correct += (label == pred)
        axs[row, col].set_title('label: {}, pred: {}'.format(label, pred),color='r' if label != pred else 'b')
    plt.tight_layout()
    plt.savefig('results.png')
    print('accuracy: {} out of {}'.format(correct, num_imgs))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                ])
    real_ds = ImageFolder(root=os.path.join(os.getcwd(),'real'), transform=trans) #real world images
    real_dl = DataLoader(real_ds, batch_size=16,shuffle=True)
    model = binary_classifier().to(device)
    model.load_state_dict(torch.load('checkpoint/model.pth')['model'],strict=False)
    for img,label in real_dl:
        img = img.to(device)
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        show_results(img.cpu(), label.cpu(),pred.cpu())