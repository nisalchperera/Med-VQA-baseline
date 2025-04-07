import os

# os.environ['HF_HOME'] = os.path.join(os.getcwd(), "models/huggingface/")
# os.makedirs(os.environ['HF_HOME'], exist_ok=True)

import streamlit as st

# # Retrieve Hugging Face token from secrets
# hf_token = st.secrets["secrets"]["TOKEN"]

# # Set the token as an environment variable (if needed)
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# transformers_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
# if os.path.exists(transformers_cache_dir):
#     shutil.rmtree(transformers_cache_dir)

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", cache_dir=os.environ['HF_HOME'], repo_type="model", force_download=True, token=hf_token, max_workers=1)

import torch
import torchvision.transforms as transforms
import gdown
import json

from PIL import Image
from transformers import AutoTokenizer

from model import MedVQA

cwd = os.getcwd()

@st.cache_resource
def download(file_id, output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", os.path.join(cwd, output_path), quiet=False)

# Model loading with caching
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedVQA().to(device)
    model.load_state_dict(torch.load("./models/medvqa_epoch_10.pth", map_location=device))
    model.eval()
    return model, device

# Load label mappings
with open('./label2ans.json') as f:
    label2ans = json.load(f)

# Downloading Models
# https://drive.google.com/file/d/1-eqG2ULS-zTTOsgTScbFJYzhhnCQAYnh/view?usp=sharing
# https://drive.google.com/file/d/1-eqG2ULS-zTTOsgTScbFJYzhhnCQAYnh/view?usp=sharing
if not os.path.exists(os.path.join(cwd, "models/medvqa_epoch_10.pth")):
    os.makedirs("models", exist_ok=True)
    download("1-eqG2ULS-zTTOsgTScbFJYzhhnCQAYnh", "models/medvqa_epoch_10.pth")

if not os.path.exists("models/BioMedNLP"):
    os.makedirs("models/BioMedNLP", exist_ok=True)
    # config.json https://drive.google.com/file/d/1saOo04pZt1FMX7EJ3ZVo7ynQL7e0cl74/view?usp=sharing
    # flax_model.msgpack https://drive.google.com/file/d/1jH6KcuTN8VlkBQXoVpPMSJHvYQ0hbo4j/view?usp=sharing
    # pytorch_model.bin https://drive.google.com/file/d/13HYBQ8lhOScq2vhi247JaCXjiRB1FNv9/view?usp=drive_link
    # tokenizer_config.json https://drive.google.com/file/d/1xfOyNtmxne-wmPlIcGHIqQvrFNst8rjL/view?usp=drive_link
    # vocab.txt https://drive.google.com/file/d/13Sl5V-c5mdm8KYDVAKFQdjF09F93SsKc/view?usp=drive_link

    download("1saOo04pZt1FMX7EJ3ZVo7ynQL7e0cl74", "models/BioMedNLP/config.json")
    download("1jH6KcuTN8VlkBQXoVpPMSJHvYQ0hbo4j", "models/BioMedNLP/flax_model.msgpack")
    download("13HYBQ8lhOScq2vhi247JaCXjiRB1FNv9", "models/BioMedNLP/pytorch_model.bin")
    download("1xfOyNtmxne-wmPlIcGHIqQvrFNst8rjL", "models/BioMedNLP/tokenizer_config.json")
    download("13Sl5V-c5mdm8KYDVAKFQdjF09F93SsKc", "models/BioMedNLP/vocab.txt")

# Initialize components
model, device = load_model()
tokenizer = AutoTokenizer.from_pretrained(os.path.join(cwd, "models/BioMedNLP/"))

def preprocess_question(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

# Image preprocessing (grayscale X-ray image)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values 
    ])
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("L")  # Convert to grayscale
    
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Streamlit UI
st.title("Medical Visual Question Answering System")
st.markdown("Upload a medical image and ask a question about it")

# Image upload section
with st.sidebar:
    st.header("Image Input")
    uploaded_file = st.file_uploader(
        "Choose a medical image (X-ray/CT/MRI)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Drag and drop or click to upload medical image"
    )

# Question input
question = st.text_input(
    "Enter your medical question:",
    placeholder="Is there any abnormality in the image?",
    help="Ask questions about findings, diagnoses, or observations"
)

# Display and processing
col1, col2 = st.columns([2, 3])

with col1:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(image, caption="Uploaded Medical Image", use_container_width=True)

with col2:
    if st.button("Run Inference") and uploaded_file and question:
        with st.spinner("Analyzing image and question..."):
            # Preprocessing pipeline            
            image_tensor = preprocess_image(image)
            input_ids, attention_mask = preprocess_question(question)
            
            # Model inference
            with torch.no_grad():
                logits = model(
                    image_tensor.to(device),
                    input_ids.to(device),
                    attention_mask.to(device)
                )
                predicted_label = torch.argmax(logits, dim=1).item()

            # Display results
            st.subheader("Analysis Results")
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Predicted Answer:** {label2ans[str(predicted_label)]}")
            st.success("Analysis complete!")
