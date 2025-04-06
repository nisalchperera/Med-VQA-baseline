import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer
from model import MedVQA # Ensure the model script is imported

# Load trained model
MODEL_PATH = "./models/medvqa_epoch_10.pth"  # Update with your checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
num_classes = 557  # Update based on your dataset
model = MedVQA().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)

# checkpoint = torch.load(MODEL_PATH, map_location=device)

# Image preprocessing (grayscale X-ray image)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values 
    ])
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Text preprocessing (PubMedBERT tokenizer)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

def preprocess_question(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

# Inference function
def infer(image_path, question):
    image_tensor = preprocess_image(image_path)
    input_ids, attention_mask = preprocess_question(question)

    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class  # Map this to the answer label

# Run inference on an X-ray image and a medical question
# image_path = "./synpic676.jpg"  # test  X-ray image path
image_path = "./image.jpg"  # test  X-ray image path

# question = "Is the lungs infected?"  # test question
# question = "What is the diagnosis?"  # test question
question = "Is there any abnormality in the image?"  # test question
import json
label2ans = json.load(open('./label2ans.json'))
predicted_label = infer(image_path, question)
print(f"Predicted Answer Label: {label2ans[str(predicted_label)]}")
