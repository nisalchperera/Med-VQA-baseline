import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MedVQADataset(Dataset):
    def __init__(self, annotations, image_path, text_tokenizer, ans2label, transform=None, max_seq_len=40):
        self.annotations = annotations
        self.image_path = image_path
        self.text_tokenizer = text_tokenizer  # PubMedBERT tokenizer
        self.ans2label = ans2label  # Answer to label mapping
        self.img_transform = transform
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        sample = self.annotations[index]
        image_name = sample.get("image_name", "")
        question = str(sample.get("question", ""))
        answer = sample.get("answer", "")

        #  Load and transform image
        try:
            image = Image.open(os.path.join(self.image_path, image_name)).convert("L")  # Grayscale
            if self.img_transform:
                image = self.img_transform(image)
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            image = torch.zeros(1, 224, 224)  # Default empty image tensor

        #  Tokenize question using PubMedBERT
        question_enc = self.text_tokenizer(
            question, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        #  Get the answer label from ans2label mapping
        answer_enc = torch.tensor(self.ans2label.get(answer, -1) ) # Using -1 for unknown answers

        return {
            "image": image,  # (1, 224, 224)
            "question_ids": question_enc["input_ids"].squeeze(0),  # PubMedBERT tokens
            "question_mask": question_enc["attention_mask"].squeeze(0),  # Attention mask
            "answer_label": answer_enc,  # Label for classification
        }

#  Collate function for DataLoader
def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    question_ids = torch.stack([item["question_ids"] for item in batch])
    question_masks = torch.stack([item["question_mask"] for item in batch])
    answer_label = torch.stack([item["answer_label"] for item in batch])
    return {
        "image": images,  
        "question_ids": question_ids,  
        "question_mask": question_masks,  
        "answer_label": answer_label  
    }

#  Mapping answers to labels
def map_answer_to_label(annotations):
    if os.path.exists('../ans2label.json') and os.path.exists('../label2ans.json'):
        answer2label, label2answer = map_answer_to_label(annotations)

        answer2label = json.load(open('ans2label.json'))
        label2answer = json.load(open('label2ans.json'))
        print("inside this ")
    else:
        answers = set()
        for sample in annotations:
            answers.add(sample["answer"])
        answers = list(answers)

        answer2label = {answer: label for label, answer in enumerate(answers)}
        label2answer = {label: answer for answer, label in answer2label.items()}
        json.dump(answer2label, open('ans2label.json', 'w'))
        json.dump(label2answer, open('label2ans.json', 'w'))

    return answer2label, label2answer

if __name__ == '__main__':
    from transformers import AutoTokenizer
    DATASET_PATH = "../data/vqa"
    IMAGE_PATH = os.path.join(DATASET_PATH, "image")
    ANNOTATION_FILE = os.path.join(DATASET_PATH, "question-answer.json")
    
    with open(ANNOTATION_FILE, "r") as file:
        annotations = json.load(file)
    print(f"Number of samples: {len(annotations)}")
    
    # Create answer to label mapping
    ans2label, label2ans = map_answer_to_label(annotations)
    print(label2ans[192])
    print(f"Number of unique answers: {len(ans2label)}")    
    # print(ans2label)
    transforms  = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Initialize dataset with ans2label passed
    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
  # Replace with actual PubMedBERT tokenizer
    dataset = MedVQADataset(annotations, IMAGE_PATH, text_tokenizer, ans2label, transforms)

    # Test dataset output
    sample = dataset[0]
    # print(sample)
    print(f"Image Shape: {sample['image'].shape}")
    print(f"Question Tokens: {sample['question_ids'].shape}")
    print(f"Answer Label: {sample['answer_label']}")

    for data in dataset:
        print(data)
        