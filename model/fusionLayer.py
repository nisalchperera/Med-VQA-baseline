import torch
import torch.nn as nn

# Fusion Model
class FusionModel(nn.Module):
    def __init__(self, img_feat_dim=512, text_feat_dim=512, fusion_dim=512):
        super(FusionModel, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + text_feat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, img_feat, text_feat):
        fused_feat = torch.cat((img_feat, text_feat), dim=1)
        return self.fusion(fused_feat)  # (batch_size, fusion_dim)


class ClassificationLayer(nn.Module):
    def __init__(self, fusion_feat_dim=512, num_classes=557):
        super(ClassificationLayer, self).__init__()
        self.fc = nn.Linear(fusion_feat_dim, num_classes)

    def forward(self, fused_feat):
        return self.fc(fused_feat)  # Logits (apply softmax in loss function)


if __name__ == '__main__':
    # test fusion model
    img_feat = torch.randn(2, 512)
    text_feat = torch.randn(2, 512)
    model = FusionModel()
    output = model(img_feat, text_feat)
    print(model)
    print(output.shape)

    # test classification layer
    fused_feat = torch.randn(2, 512)
    model = ClassificationLayer()
    output = model(fused_feat)
    print(model)
    print(output.shape)