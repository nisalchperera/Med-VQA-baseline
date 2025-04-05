import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm import tqdm

from data_pipeline import MedVQADataset, collate_fn, map_answer_to_label
from model import MedVQA

# Function to initialize the model
def initialize_model(image_feat_dim=512, text_feat_dim=512, fusion_hidden_dim=512, num_classes=557):
    model = MedVQA(image_feat_dim=image_feat_dim, text_feat_dim=text_feat_dim, fusion_hidden_dim=fusion_hidden_dim, num_classes=num_classes)
    return model

# Function to set up the loss function and optimizer
def setup_optimizer(model, learning_rate=1e-4):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    return criterion, optimizer, scheduler

# Function for training one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        images = batch['image'].to(device)
        input_ids = batch['question_ids'].to(device)
        attention_mask = batch['question_mask'].to(device)
        labels = batch['answer_label'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, input_ids, attention_mask)
        
        # Compute loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accuracy
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = (correct_preds / total_preds) * 100
    return avg_loss, accuracy

# Function for evaluating the model on the validation set
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            input_ids = batch['question_ids'].to(device)
            attention_mask = batch['question_mask'].to(device)
            labels = batch['answer_label'].to(device)

            # Forward pass
            outputs = model(images, input_ids, attention_mask)
            
            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = (val_correct_preds / val_total_preds) * 100
    return avg_val_loss, val_accuracy

# Function to save the model checkpoint
def save_model_checkpoint(model, epoch, save_path="models"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_path = os.path.join(save_path, f"medvqa_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)

# Full training loop
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    for epoch in range(num_epochs):
        # Train for one epoch
        avg_loss, accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

        # Validate the model
        avg_val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the model checkpoint
        save_model_checkpoint(model, epoch)

    print("Training completed!")

# Main function to prepare everything and start training
def main(train_dataset, val_dataset, num_epochs=10, batch_size=32, learning_rate=1e-4, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    # device = 'cpu'
    # DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the model
    model = initialize_model()
    model.to(device)

    # Set up loss function, optimizer, and scheduler
    criterion, optimizer, scheduler = setup_optimizer(model, learning_rate)

    # Start training and evaluation loop
    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

if __name__ == "__main__":
    
    from transformers import T5Tokenizer, AutoTokenizer
    import json

    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    
    # Define paths
    DATASET_PATH = "./data/vqa"
    IMAGE_PATH = os.path.join(DATASET_PATH, "image")
    ANNOTATION_FILE = os.path.join(DATASET_PATH, "question-answer.json")
    
    with open(ANNOTATION_FILE, "r") as file:
        annotations = json.load(file)
    print(f"Number of samples: {len(annotations)}")
    
    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # check if mapping file is present

    ans2label, label2ans = map_answer_to_label(annotations)
    # json.dump(ans2label, open('ans2label.json', 'w'))
    # json.dump(label2ans, open('label2ans.json', 'w'))

    # if not os.path.exists('ans2label.json') or not os.path.exists('label2ans.json'):
    #     ans2label, label2ans = map_answer_to_label(annotations)

    #     json.dump(ans2label, open('ans2label.json', 'w'))
    #     json.dump(label2ans, open('label2ans.json', 'w'))


    # ans2label = json.load(open('ans2label.json'))
    # label2ans = json.load(open('label2ans.json'))

    # Create dataset and dataloader
    dataset = MedVQADataset(annotations, IMAGE_PATH, text_tokenizer, ans2label, image_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(train_size, val_size)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedVQA().to(device)

    # Train model
    main(train_dataset, val_dataset)
    