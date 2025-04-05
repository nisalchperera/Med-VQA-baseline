import torch
import torch.nn as nn
from transformers import AutoModel

# Text Encoder (PubMedBERT)
class TextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", feat_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.text_feat = nn.Sequential(
            nn.Linear(768, feat_dim),
            nn.BatchNorm1d(feat_dim),  # BatchNorm requires batch size > 1
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]  # Extract CLS token features
        return self.text_feat(text_features)  # (batch_size, feat_dim)

if __name__ == '__main__':
    # test text encoder
    test_input_ids = torch.randint(0, 30522, (2, 512))  # token ids
    test_attention_mask = torch.randint(0, 2, (2, 512))  # attention mask
    model = TextEncoder()
    output = model(test_input_ids, test_attention_mask)
    print(model)
    print(output.shape)