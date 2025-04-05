import torch
import torch.nn as nn
from torchvision import models


# Image Encoder (VGG16)
class ImageEncoder(nn.Module):
    def __init__(self, feat_dim=512):
        super(ImageEncoder, self).__init__()
        base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Modifying the first convolutional layer for grayscale input
        base.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Removing the classification layers
        self.vision_encoder = nn.Sequential(*list(base.children())[:-1])
        
        # Adaptive pooling and feature projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_feat = nn.Linear(512, feat_dim)

    def forward(self, x):
        x = self.vision_encoder(x)
        x = self.avgpool(x).view(x.size(0), -1)  # Flatten
        x = self.img_feat(x)
        return x  # Output shape: (batch_size, feat_dim)
    

if __name__ == '__main__':
    # test image encoder
    test_input = torch.randn(2, 1, 224, 224)
    model = ImageEncoder()
    output = model(test_input)
    print(model)    
    print(output.shape)

    