import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusionLayer import FusionModel, ClassificationLayer

# MedVQA Model
class MedVQA(nn.Module):
    def __init__(self, image_feat_dim=512, text_feat_dim=512, fusion_hidden_dim=512, num_classes=557):
        super(MedVQA, self).__init__()
        self.image_encoder = ImageEncoder(feat_dim=image_feat_dim)
        self.text_encoder = TextEncoder(feat_dim=text_feat_dim)
        self.fusion_model = FusionModel(img_feat_dim=image_feat_dim, text_feat_dim=text_feat_dim, fusion_dim=fusion_hidden_dim)
        self.classification_layer = ClassificationLayer(fusion_feat_dim=fusion_hidden_dim, num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        fused_features = self.fusion_model(image_features, text_features)
        return self.classification_layer(fused_features)  # Returns logits
