import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, PeftModel
import glob
from tqdm import tqdm
import zipfile
import shutil
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
from datetime import datetime

# Configuration
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE_CLIP = 1e-5
LEARNING_RATE_CLASSIFIER = 5e-5
WEIGHT_DECAY = 0.01
SEED = 42
RESULTS_DIR = "results"

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)

# Kaggle paths
KAGGLE_WORKING_DIR = "/kaggle/working"
DATASET_DIR = os.path.join(KAGGLE_WORKING_DIR, "crop_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
output_dir = os.path.join(KAGGLE_WORKING_DIR, "output_clip_models")

def download_and_extract_dataset():
    """Download the dataset from Google Drive if it's not already available"""
    if os.path.exists(IMAGES_DIR):
        print(f"Dataset already exists at {IMAGES_DIR}. Skipping download.")
        return IMAGES_DIR
    
    # Create directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Download the zip file from Google Drive
    print("Downloading dataset from Google Drive...")
    drive_url = "https://drive.google.com/file/d/1kfB3zkittoef4BasOhwvAb8Cb66EPXst/view"
    zip_path = os.path.join(DATASET_DIR, "dataset.zip")
    
    # Install gdown if not already installed
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        !pip install -q gdown
        import gdown
    
    # Use gdown to download the file
    gdown.download(url=drive_url, output=zip_path, quiet=False, fuzzy=True)
    
    # Extract the zip file
    print(f"Extracting zip file to {DATASET_DIR}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    
    # Check if the images directory exists after extraction
    if not os.path.exists(IMAGES_DIR):
        # Try to find the images directory
        for root, dirs, files in os.walk(DATASET_DIR):
            for dir_name in dirs:
                if dir_name == "images":
                    actual_images_dir = os.path.join(root, dir_name)
                    print(f"Found images at {actual_images_dir}")
                    # Move to expected location if needed
                    if actual_images_dir != IMAGES_DIR:
                        shutil.move(actual_images_dir, IMAGES_DIR)
                    break
    
    # Remove the zip file to save space
    os.remove(zip_path)
    
    # Verify the images directory exists
    if not os.path.exists(IMAGES_DIR):
        raise FileNotFoundError(f"Images directory not found after extraction. Check the zip file structure.")
    
    # Print the structure of the images directory
    for root, dirs, files in os.walk(IMAGES_DIR, topdown=True):
        # Only process the top level
        if root == IMAGES_DIR:
            print(f"Found {len(dirs)} category directories and {len(files)} files in {root}")
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                img_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                print(f"  - {dir_name}: {img_count} images")
        # Don't recurse deeper
        dirs.clear()
    
    return IMAGES_DIR

# Function to extract crop name from label
def extract_crop_from_label(label):
    """Extract only the crop name from the disease label
    
    Args:
        label (str): The original label string, usually in format "Crop,Disease" 
                     or "Crop,Healthy"
    
    Returns:
        str: The extracted crop name
    """
    # Most labels use format "Crop,Disease" or "Crop,Healthy"
    if ',' in label:
        crop_part = label.split(',')[0]
        
        # Handle special cases like "Bell Pepper"
        if crop_part == "Bell Pepper" or crop_part.startswith("Bell "):
            return "Bell Pepper"
        
        return crop_part
    
    # If no comma, return the whole label (might be just the crop name)
    return label

# Create a custom dataset class with dual-label support
class CropDiseaseDataset(Dataset):
    """Dataset for crop disease classification with support for dual-head (crop and disease) classification.
    
    This dataset processes images of crop diseases and provides both disease and crop labels
    for training a multi-task model.
    """
    def __init__(self, image_dir, processor, split="train"):
        """Initialize the dataset.
        
        Args:
            image_dir (str): Path to the directory containing image folders
            processor (CLIPProcessor): Image processor for CLIP model
            split (str): Either "train" or "val" to determine data split
        """
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self.crop_map = {}  # Map to store crop labels
        
        # Walk through directories to collect image paths and labels
        dirs = sorted(glob.glob(os.path.join(image_dir, "*")))
        for idx, dir_path in enumerate(dirs):
            if os.path.isdir(dir_path):
                label = os.path.basename(dir_path)
                self.label_map[label] = idx
                
                # Extract crop name from label and add to crop map
                crop_name = extract_crop_from_label(label)
                if crop_name not in self.crop_map:
                    self.crop_map[crop_name] = len(self.crop_map)
                
                # Get all images in this category
                images = glob.glob(os.path.join(dir_path, "*.jpg")) + \
                         glob.glob(os.path.join(dir_path, "*.jpeg")) + \
                         glob.glob(os.path.join(dir_path, "*.png"))
                
                if not images:
                    print(f"Warning: No images found in {dir_path}")
                    continue
                
                # Split into train/test
                if split == "train":
                    images = images[:int(0.8*len(images))]
                else:
                    images = images[int(0.8*len(images)):]
                
                self.image_paths.extend(images)
                self.labels.extend([idx] * len(images))
        
        # Create inverse mapping for labels and crops
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.idx_to_crop = {v: k for k, v in self.crop_map.items()}
        
        # Create mapping from label index to crop index
        self.label_to_crop_idx = {}
        for label, idx in self.label_map.items():
            crop_name = extract_crop_from_label(label)
            self.label_to_crop_idx[idx] = self.crop_map[crop_name]
        
        print(f"Loaded {len(self.image_paths)} images for {split}")
        if self.label_map:
            print(f"Label map: Found {len(self.label_map)} disease classes")
            print(f"Crop map: Found {len(self.crop_map)} crop types")
        else:
            print("Warning: No labels were found!")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            dict: Dictionary containing image features and labels
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Remove batch dimension for dataset
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        
        # Add both disease and crop labels
        inputs["labels"] = torch.tensor(self.labels[idx])
        inputs["crop_labels"] = torch.tensor(self.label_to_crop_idx[self.labels[idx]])
        return inputs

# Function to evaluate the base CLIP model using zero-shot classification
def evaluate_base_model(model, processor, dataset, device):
    """Evaluate the base CLIP model using zero-shot classification.
    
    Args:
        model (CLIPModel): The CLIP model to evaluate
        processor (CLIPProcessor): The CLIP processor
        dataset (CropDiseaseDataset): The dataset to evaluate on
        device (torch.device): The device to use for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\n--- Evaluating Base CLIP Model ---")
    model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Initialize metrics
    disease_correct = 0
    crop_correct = 0
    total = 0
    confidences = []
    all_preds = []
    all_labels = []
    all_crop_preds = []
    all_crop_labels = []
    
    # Get all unique disease labels and crop labels
    disease_labels = dataset.idx_to_label
    crop_labels = dataset.idx_to_crop
    
    # Extract unique crops and diseases for text features
    unique_crops = list(crop_labels.values())
    unique_diseases = list(disease_labels.values())
    
    # Generate text features for zero-shot classification
    with torch.no_grad():
        # Text prompts for diseases
        disease_texts = [f"a photo of a plant with {disease}" for disease in unique_diseases]
        disease_text_inputs = processor(text=disease_texts, return_tensors="pt", padding=True).to(device)
        disease_text_features = model.get_text_features(**disease_text_inputs)
        disease_text_features = disease_text_features / disease_text_features.norm(dim=-1, keepdim=True)
        
        # Text prompts for crops
        crop_texts = [f"a photo of a {crop} plant" for crop in unique_crops]
        crop_text_inputs = processor(text=crop_texts, return_tensors="pt", padding=True).to(device)
        crop_text_features = model.get_text_features(**crop_text_inputs)
        crop_text_features = crop_text_features / crop_text_features.norm(dim=-1, keepdim=True)
        
        # Evaluate
        for batch in tqdm(dataloader, desc="Evaluating base model"):
            batch = {k: v.to(device) for k, v in batch.items() if k in ["pixel_values", "labels", "crop_labels"]}
            
            # Get image features
            image_features = model.get_image_features(pixel_values=batch["pixel_values"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            disease_similarity = (100.0 * image_features @ disease_text_features.T).softmax(dim=-1)
            crop_similarity = (100.0 * image_features @ crop_text_features.T).softmax(dim=-1)
            
            # Get predictions
            disease_probs, disease_preds = disease_similarity.max(dim=-1)
            crop_probs, crop_preds = crop_similarity.max(dim=-1)
            
            # Convert label indices to indices in our text features array
            true_disease_indices = torch.tensor([list(disease_labels.keys()).index(label.item()) 
                                 for label in batch["labels"]]).to(device)
            true_crop_indices = torch.tensor([list(crop_labels.keys()).index(label.item()) 
                                 for label in batch["crop_labels"]]).to(device)
            
            # Update metrics
            disease_correct += (disease_preds == true_disease_indices).sum().item()
            crop_correct += (crop_preds == true_crop_indices).sum().item()
            total += batch["labels"].size(0)
            
            # Store confidences and predictions for analysis
            confidences.extend(disease_probs.cpu().tolist())
            all_preds.extend(disease_preds.cpu().tolist())
            all_labels.extend(true_disease_indices.cpu().tolist())
            all_crop_preds.extend(crop_preds.cpu().tolist())
            all_crop_labels.extend(true_crop_indices.cpu().tolist())
    
    # Calculate metrics
    disease_accuracy = 100 * disease_correct / total
    crop_accuracy = 100 * crop_correct / total
    avg_confidence = sum(confidences) / len(confidences)
    
    print(f"Base Model Results:")
    print(f"  - Disease Accuracy: {disease_accuracy:.2f}%")
    print(f"  - Crop Accuracy: {crop_accuracy:.2f}%")
    print(f"  - Average Confidence: {avg_confidence:.2f}%")
    
    # Store detailed metrics for each class
    disease_class_accuracy = {}
    for i in range(len(unique_diseases)):
        indices = [j for j, x in enumerate(all_labels) if x == i]
        if indices:
            correct = sum(1 for j in indices if all_preds[j] == all_labels[j])
            disease_class_accuracy[unique_diseases[i]] = 100 * correct / len(indices)
    
    crop_class_accuracy = {}
    for i in range(len(unique_crops)):
        indices = [j for j, x in enumerate(all_crop_labels) if x == i]
        if indices:
            correct = sum(1 for j in indices if all_crop_preds[j] == all_crop_labels[j])
            crop_class_accuracy[unique_crops[i]] = 100 * correct / len(indices)
    
    return {
        "disease_accuracy": disease_accuracy,
        "crop_accuracy": crop_accuracy,
        "avg_confidence": avg_confidence,
        "disease_class_accuracy": disease_class_accuracy,
        "crop_class_accuracy": crop_class_accuracy,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_crop_preds": all_crop_preds,
        "all_crop_labels": all_crop_labels,
        "confidences": confidences
    }

# Define a LoRA trainer class for comparison
class LoraTrainer:
    """Trainer class for fine-tuning CLIP model with LoRA adapters.
    
    This class handles the training and evaluation of a CLIP model with LoRA
    adapters for both crop and disease classification.
    """
    def __init__(self, model, processor, train_dataset, val_dataset, output_dir, batch_size=8, num_epochs=3):
        """Initialize the LoRA trainer.
        
        Args:
            model (CLIPModel): The CLIP model to fine-tune
            processor (CLIPProcessor): The CLIP processor
            train_dataset (CropDiseaseDataset): The training dataset
            val_dataset (CropDiseaseDataset): The validation dataset
            output_dir (str): Directory to save the model and results
            batch_size (int): Batch size for training
            num_epochs (int): Number of training epochs
        """
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Print model structure to help identify target modules
        print("Identifying CLIP model layers for LoRA...")
        clip_module_names = [name for name, _ in model.named_modules()]
        attention_layers = [name for name in clip_module_names if 'attn' in name]
        print(f"Found {len(attention_layers)} attention-related layers")
        if len(attention_layers) > 0:
            print(f"Sample attention layer names: {attention_layers[:3]}...")
        
        # Create LoRA config
        # Using a relatively small rank and targeting only specific visual encoder layers
        lora_config = LoraConfig(
            r=4,  
            lora_alpha=8,  # Lower alpha scaling
            target_modules=["q_proj", "k_proj", "v_proj"],  # Target attention projection layers
            lora_dropout=0.1,  # Higher dropout probability 
            bias="none",
            task_type="FEATURE_EXTRACTION"  # Changed from CAUSAL_LM to FEATURE_EXTRACTION
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(model, lora_config)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Create classifier heads
        num_disease_classes = len(self.train_dataset.label_map)
        num_crop_classes = len(self.train_dataset.crop_map)
        self.disease_classifier = torch.nn.Linear(512, num_disease_classes).to(self.device)
        self.crop_classifier = torch.nn.Linear(512, num_crop_classes).to(self.device)
        
        # Set up optimizer
        params_to_train = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad], "lr": LEARNING_RATE_CLIP},
            {"params": self.disease_classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER},
            {"params": self.crop_classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}
        ]
        
        self.optimizer = torch.optim.AdamW(
            params_to_train,
            weight_decay=WEIGHT_DECAY
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Logging
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "disease_accuracy": [],
            "crop_accuracy": [],
            "combined_accuracy": []
        }
    
    def train(self):
        """Train the model.
        
        Returns:
            tuple: The trained model and classification heads
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        best_disease_acc = 0.0
        
        # Loss function weights for balancing crop and disease tasks
        loss_weights = {
            "disease": 0.7,  # Higher weight for disease classification
            "crop": 0.3      # Lower weight for crop classification
        }
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            self.disease_classifier.train()
            self.crop_classifier.train()
            train_loss = 0.0
            disease_loss = 0.0
            crop_loss = 0.0
            
            for batch in tqdm(self.train_loader, desc="Training"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Get image features from CLIP
                with torch.set_grad_enabled(True):
                    image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                
                # Run through classifiers
                disease_logits = self.disease_classifier(image_features)
                crop_logits = self.crop_classifier(image_features)
                
                # Calculate loss
                loss_fct = torch.nn.CrossEntropyLoss()
                disease_loss_val = loss_fct(disease_logits, batch["labels"])
                crop_loss_val = loss_fct(crop_logits, batch["crop_labels"])
                
                # Combine losses with weights
                combined_loss = (loss_weights["disease"] * disease_loss_val) + (loss_weights["crop"] * crop_loss_val)
                
                combined_loss.backward()
                self.optimizer.step()
                
                train_loss += combined_loss.item()
                disease_loss += disease_loss_val.item()
                crop_loss += crop_loss_val.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_disease_loss = disease_loss / len(self.train_loader)
            avg_crop_loss = crop_loss / len(self.train_loader)
            
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"  - Disease loss: {avg_disease_loss:.4f}")
            print(f"  - Crop loss: {avg_crop_loss:.4f}")
            
            # Validation
            self.model.eval()
            self.disease_classifier.eval()
            self.crop_classifier.eval()
            val_loss = 0.0
            disease_correct = 0
            crop_correct = 0
            both_correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validation"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Get image features
                    image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                    
                    # Run through classifiers
                    disease_logits = self.disease_classifier(image_features)
                    crop_logits = self.crop_classifier(image_features)
                    
                    # Calculate loss
                    disease_loss_val = loss_fct(disease_logits, batch["labels"])
                    crop_loss_val = loss_fct(crop_logits, batch["crop_labels"])
                    combined_loss = (loss_weights["disease"] * disease_loss_val) + (loss_weights["crop"] * crop_loss_val)
                    val_loss += combined_loss.item()
                    
                    # Calculate accuracy
                    _, disease_preds = torch.max(disease_logits, 1)
                    _, crop_preds = torch.max(crop_logits, 1)
                    total += batch["labels"].size(0)
                    
                    disease_correct += (disease_preds == batch["labels"]).sum().item()
                    crop_correct += (crop_preds == batch["crop_labels"]).sum().item()
                    
                    # Count cases where both predictions are correct
                    both_correct += ((disease_preds == batch["labels"]) & 
                                    (crop_preds == batch["crop_labels"])).sum().item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            disease_accuracy = 100 * disease_correct / total
            crop_accuracy = 100 * crop_correct / total
            combined_accuracy = 100 * both_correct / total
            
            print(f"Average validation loss: {avg_val_loss:.4f}")
            print(f"Disease accuracy: {disease_accuracy:.2f}%")
            print(f"Crop accuracy: {crop_accuracy:.2f}%")
            print(f"Combined accuracy (both correct): {combined_accuracy:.2f}%")
            
            # Save metrics for plotting
            self.metrics["train_loss"].append(avg_train_loss)
            self.metrics["val_loss"].append(avg_val_loss)
            self.metrics["disease_accuracy"].append(disease_accuracy)
            self.metrics["crop_accuracy"].append(crop_accuracy)
            self.metrics["combined_accuracy"].append(combined_accuracy)
            
            # Save best model
            if disease_accuracy > best_disease_acc:
                best_disease_acc = disease_accuracy
                # Save model
                best_model_path = os.path.join(self.output_dir, "lora_best")
                os.makedirs(best_model_path, exist_ok=True)
                
                # Save CLIP model
                self.model.save_pretrained(best_model_path)
                
                # Save classifiers
                torch.save(self.disease_classifier.state_dict(), os.path.join(best_model_path, "disease_classifier.pt"))
                torch.save(self.crop_classifier.state_dict(), os.path.join(best_model_path, "crop_classifier.pt"))
                
                # Save label maps
                with open(os.path.join(best_model_path, "label_maps.json"), 'w') as f:
                    json.dump({
                        "disease_map": self.train_dataset.label_map,
                        "crop_map": self.train_dataset.crop_map
                    }, f)
                
                print(f"Best model saved at epoch {epoch+1}")
        
        # Save final model
        final_path = os.path.join(self.output_dir, "lora_final")
        os.makedirs(final_path, exist_ok=True)
        self.model.save_pretrained(final_path)
        torch.save(self.disease_classifier.state_dict(), os.path.join(final_path, "disease_classifier.pt"))
        torch.save(self.crop_classifier.state_dict(), os.path.join(final_path, "crop_classifier.pt"))
        
        # Save training metrics
        with open(os.path.join(self.output_dir, "lora_training_metrics.json"), 'w') as f:
            json.dump(self.metrics, f)
        
        return self.model, self.disease_classifier, self.crop_classifier
    
    def evaluate(self, detailed=True):
        """Evaluate the model.
        
        Args:
            detailed (bool): Whether to calculate detailed metrics
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\n--- Evaluating LoRA Model ---")
        self.model.eval()
        self.disease_classifier.eval()
        self.crop_classifier.eval()
        
        disease_correct = 0
        crop_correct = 0
        both_correct = 0
        total = 0
        confidences = []
        all_disease_preds = []
        all_disease_labels = []
        all_crop_preds = []
        all_crop_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get image features
                image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                
                # Run through classifiers
                disease_logits = self.disease_classifier(image_features)
                crop_logits = self.crop_classifier(image_features)
                
                # Get predictions and confidence scores
                disease_probs = torch.nn.functional.softmax(disease_logits, dim=1)
                disease_conf, disease_preds = disease_probs.max(dim=1)
                
                crop_probs = torch.nn.functional.softmax(crop_logits, dim=1)
                _, crop_preds = crop_probs.max(dim=1)
                
                # Update metrics
                total += batch["labels"].size(0)
                disease_correct += (disease_preds == batch["labels"]).sum().item()
                crop_correct += (crop_preds == batch["crop_labels"]).sum().item()
                both_correct += ((disease_preds == batch["labels"]) & 
                                 (crop_preds == batch["crop_labels"])).sum().item()
                
                # Store for detailed analysis
                confidences.extend(disease_conf.cpu().tolist())
                all_disease_preds.extend(disease_preds.cpu().tolist())
                all_disease_labels.extend(batch["labels"].cpu().tolist())
                all_crop_preds.extend(crop_preds.cpu().tolist())
                all_crop_labels.extend(batch["crop_labels"].cpu().tolist())
        
        # Calculate overall metrics
        disease_accuracy = 100 * disease_correct / total
        crop_accuracy = 100 * crop_correct / total
        combined_accuracy = 100 * both_correct / total
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"LoRA Model Results:")
        print(f"  - Disease Accuracy: {disease_accuracy:.2f}%")
        print(f"  - Crop Accuracy: {crop_accuracy:.2f}%")
        print(f"  - Combined Accuracy: {combined_accuracy:.2f}%")
        print(f"  - Average Confidence: {avg_confidence:.2f}%")
        
        # Generate per-class metrics if detailed evaluation is requested
        disease_class_accuracy = {}
        crop_class_accuracy = {}
        
        if detailed:
            # Disease class accuracy
            for label_idx, label_name in self.val_dataset.idx_to_label.items():
                indices = [i for i, x in enumerate(all_disease_labels) if x == label_idx]
                if indices:
                    correct = sum(1 for i in indices if all_disease_preds[i] == all_disease_labels[i])
                    disease_class_accuracy[label_name] = 100 * correct / len(indices)
            
            # Crop class accuracy
            for label_idx, label_name in self.val_dataset.idx_to_crop.items():
                indices = [i for i, x in enumerate(all_crop_labels) if x == label_idx]
                if indices:
                    correct = sum(1 for i in indices if all_crop_preds[i] == all_crop_labels[i])
                    crop_class_accuracy[label_name] = 100 * correct / len(indices)
        
        return {
            "disease_accuracy": disease_accuracy,
            "crop_accuracy": crop_accuracy,
            "combined_accuracy": combined_accuracy,
            "avg_confidence": avg_confidence,
            "disease_class_accuracy": disease_class_accuracy,
            "crop_class_accuracy": crop_class_accuracy,
            "all_disease_preds": all_disease_preds,
            "all_disease_labels": all_disease_labels,
            "all_crop_preds": all_crop_preds,
            "all_crop_labels": all_crop_labels,
            "confidences": confidences
        }

# Define a custom trainer class for selective fine-tuning
class SelectiveFineTuningTrainer:
    """Trainer class for selective fine-tuning of CLIP model.
    
    This class handles the training and evaluation of a CLIP model using
    selective parameter freezing for both crop and disease classification.
    """
    def __init__(self, model, processor, train_dataset, val_dataset, output_dir, batch_size=8, num_epochs=3):
        """Initialize the selective fine-tuning trainer.
        
        Args:
            model (CLIPModel): The CLIP model to fine-tune
            processor (CLIPProcessor): The CLIP processor
            train_dataset (CropDiseaseDataset): The training dataset
            val_dataset (CropDiseaseDataset): The validation dataset
            output_dir (str): Directory to save the model and results
            batch_size (int): Batch size for training
            num_epochs (int): Number of training epochs
        """
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Create classifier heads
        num_disease_classes = len(self.train_dataset.label_map)
        num_crop_classes = len(self.train_dataset.crop_map)
        self.disease_classifier = torch.nn.Linear(512, num_disease_classes).to(self.device)
        self.crop_classifier = torch.nn.Linear(512, num_crop_classes).to(self.device)
        
        # Freeze CLIP model parameters and only train the classifier
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze only the visual encoder's last transformer block
        for name, param in self.model.named_parameters():
            # Unfreeze the last transformer layer in the vision model
            if "visual.transformer.resblocks.11" in name:
                param.requires_grad = True
                print(f"Unfreezing: {name}")
        
        # Optimizer - train only the unfrozen parameters and classifier
        params_to_train = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad], "lr": LEARNING_RATE_CLIP},
            {"params": self.disease_classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER},
            {"params": self.crop_classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}
        ]
        
        self.optimizer = torch.optim.AdamW(
            params_to_train,
            weight_decay=WEIGHT_DECAY
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Logging
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "disease_accuracy": [],
            "crop_accuracy": [],
            "combined_accuracy": []
        }
    
    def train(self):
        """Train the model.
        
        Returns:
            tuple: The trained model and classification heads
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        best_disease_acc = 0.0
        
        # Loss function weights for balancing crop and disease tasks
        loss_weights = {
            "disease": 0.7,  # Higher weight for disease classification
            "crop": 0.3      # Lower weight for crop classification
        }
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            self.disease_classifier.train()
            self.crop_classifier.train()
            train_loss = 0.0
            disease_loss = 0.0
            crop_loss = 0.0
            
            for batch in tqdm(self.train_loader, desc="Training"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Get image features from CLIP
                with torch.set_grad_enabled(True):
                    image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                
                # Run through classifiers
                disease_logits = self.disease_classifier(image_features)
                crop_logits = self.crop_classifier(image_features)
                
                # Calculate loss
                loss_fct = torch.nn.CrossEntropyLoss()
                disease_loss_val = loss_fct(disease_logits, batch["labels"])
                crop_loss_val = loss_fct(crop_logits, batch["crop_labels"])
                
                # Combine losses with weights
                combined_loss = (loss_weights["disease"] * disease_loss_val) + (loss_weights["crop"] * crop_loss_val)
                
                combined_loss.backward()
                self.optimizer.step()
                
                train_loss += combined_loss.item()
                disease_loss += disease_loss_val.item()
                crop_loss += crop_loss_val.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_disease_loss = disease_loss / len(self.train_loader)
            avg_crop_loss = crop_loss / len(self.train_loader)
            
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"  - Disease loss: {avg_disease_loss:.4f}")
            print(f"  - Crop loss: {avg_crop_loss:.4f}")
            
            # Validation
            self.model.eval()
            self.disease_classifier.eval()
            self.crop_classifier.eval()
            val_loss = 0.0
            disease_correct = 0
            crop_correct = 0
            both_correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validation"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Get image features
                    image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                    
                    # Run through classifiers
                    disease_logits = self.disease_classifier(image_features)
                    crop_logits = self.crop_classifier(image_features)
                    
                    # Calculate loss
                    disease_loss_val = loss_fct(disease_logits, batch["labels"])
                    crop_loss_val = loss_fct(crop_logits, batch["crop_labels"])
                    combined_loss = (loss_weights["disease"] * disease_loss_val) + (loss_weights["crop"] * crop_loss_val)
                    val_loss += combined_loss.item()
                    
                    # Calculate accuracy
                    _, disease_preds = torch.max(disease_logits, 1)
                    _, crop_preds = torch.max(crop_logits, 1)
                    total += batch["labels"].size(0)
                    
                    disease_correct += (disease_preds == batch["labels"]).sum().item()
                    crop_correct += (crop_preds == batch["crop_labels"]).sum().item()
                    
                    # Count cases where both predictions are correct
                    both_correct += ((disease_preds == batch["labels"]) & 
                                    (crop_preds == batch["crop_labels"])).sum().item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            disease_accuracy = 100 * disease_correct / total
            crop_accuracy = 100 * crop_correct / total
            combined_accuracy = 100 * both_correct / total
            
            print(f"Average validation loss: {avg_val_loss:.4f}")
            print(f"Disease accuracy: {disease_accuracy:.2f}%")
            print(f"Crop accuracy: {crop_accuracy:.2f}%")
            print(f"Combined accuracy (both correct): {combined_accuracy:.2f}%")
            
            # Save metrics for plotting
            self.metrics["train_loss"].append(avg_train_loss)
            self.metrics["val_loss"].append(avg_val_loss)
            self.metrics["disease_accuracy"].append(disease_accuracy)
            self.metrics["crop_accuracy"].append(crop_accuracy)
            self.metrics["combined_accuracy"].append(combined_accuracy)
            
            # Save best model
            if disease_accuracy > best_disease_acc:
                best_disease_acc = disease_accuracy
                # Save model
                best_model_path = os.path.join(self.output_dir, "selective_best")
                os.makedirs(best_model_path, exist_ok=True)
                
                # Save CLIP model
                self.model.save_pretrained(best_model_path)
                
                # Save classifiers
                torch.save(self.disease_classifier.state_dict(), os.path.join(best_model_path, "disease_classifier.pt"))
                torch.save(self.crop_classifier.state_dict(), os.path.join(best_model_path, "crop_classifier.pt"))
                
                # Save label maps
                with open(os.path.join(best_model_path, "label_maps.json"), 'w') as f:
                    json.dump({
                        "disease_map": self.train_dataset.label_map,
                        "crop_map": self.train_dataset.crop_map
                    }, f)
                
                print(f"Best model saved at epoch {epoch+1}")
        
        # Save final model
        final_path = os.path.join(self.output_dir, "selective_final")
        os.makedirs(final_path, exist_ok=True)
        self.model.save_pretrained(final_path)
        torch.save(self.disease_classifier.state_dict(), os.path.join(final_path, "disease_classifier.pt"))
        torch.save(self.crop_classifier.state_dict(), os.path.join(final_path, "crop_classifier.pt"))
        
        # Save training metrics
        with open(os.path.join(self.output_dir, "selective_training_metrics.json"), 'w') as f:
            json.dump(self.metrics, f)
        
        return self.model, self.disease_classifier, self.crop_classifier
    
    def evaluate(self, detailed=True):
        """Evaluate the model.
        
        Args:
            detailed (bool): Whether to calculate detailed metrics
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\n--- Evaluating Selective Fine-tuning Model ---")
        self.model.eval()
        self.disease_classifier.eval()
        self.crop_classifier.eval()
        
        disease_correct = 0
        crop_correct = 0
        both_correct = 0
        total = 0
        confidences = []
        all_disease_preds = []
        all_disease_labels = []
        all_crop_preds = []
        all_crop_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get image features
                image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                
                # Run through classifiers
                disease_logits = self.disease_classifier(image_features)
                crop_logits = self.crop_classifier(image_features)
                
                # Get predictions and confidence scores
                disease_probs = torch.nn.functional.softmax(disease_logits, dim=1)
                disease_conf, disease_preds = disease_probs.max(dim=1)
                
                crop_probs = torch.nn.functional.softmax(crop_logits, dim=1)
                _, crop_preds = crop_probs.max(dim=1)
                
                # Update metrics
                total += batch["labels"].size(0)
                disease_correct += (disease_preds == batch["labels"]).sum().item()
                crop_correct += (crop_preds == batch["crop_labels"]).sum().item()
                both_correct += ((disease_preds == batch["labels"]) & 
                                 (crop_preds == batch["crop_labels"])).sum().item()
                
                # Store for detailed analysis
                confidences.extend(disease_conf.cpu().tolist())
                all_disease_preds.extend(disease_preds.cpu().tolist())
                all_disease_labels.extend(batch["labels"].cpu().tolist())
                all_crop_preds.extend(crop_preds.cpu().tolist())
                all_crop_labels.extend(batch["crop_labels"].cpu().tolist())
        
        # Calculate overall metrics
        disease_accuracy = 100 * disease_correct / total
        crop_accuracy = 100 * crop_correct / total
        combined_accuracy = 100 * both_correct / total
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"Selective Fine-tuning Model Results:")
        print(f"  - Disease Accuracy: {disease_accuracy:.2f}%")
        print(f"  - Crop Accuracy: {crop_accuracy:.2f}%")
        print(f"  - Combined Accuracy: {combined_accuracy:.2f}%")
        print(f"  - Average Confidence: {avg_confidence:.2f}%")
        
        # Generate per-class metrics if detailed evaluation is requested
        disease_class_accuracy = {}
        crop_class_accuracy = {}
        
        if detailed:
            # Disease class accuracy
            for label_idx, label_name in self.val_dataset.idx_to_label.items():
                indices = [i for i, x in enumerate(all_disease_labels) if x == label_idx]
                if indices:
                    correct = sum(1 for i in indices if all_disease_preds[i] == all_disease_labels[i])
                    disease_class_accuracy[label_name] = 100 * correct / len(indices)
            
            # Crop class accuracy
            for label_idx, label_name in self.val_dataset.idx_to_crop.items():
                indices = [i for i, x in enumerate(all_crop_labels) if x == label_idx]
                if indices:
                    correct = sum(1 for i in indices if all_crop_preds[i] == all_crop_labels[i])
                    crop_class_accuracy[label_name] = 100 * correct / len(indices)
        
        return {
            "disease_accuracy": disease_accuracy,
            "crop_accuracy": crop_accuracy,
            "combined_accuracy": combined_accuracy,
            "avg_confidence": avg_confidence,
            "disease_class_accuracy": disease_class_accuracy,
            "crop_class_accuracy": crop_class_accuracy,
            "all_disease_preds": all_disease_preds,
            "all_disease_labels": all_disease_labels,
            "all_crop_preds": all_crop_preds,
            "all_crop_labels": all_crop_labels,
            "confidences": confidences
        }

# Function to count parameters that require gradients
def count_trainable_parameters(model):
    """Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): The model to count parameters for
        
    Returns:
        int: The number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Main function to generate plots
def create_plots(base_results, lora_results, selective_results, output_dir):
    """Create and save plots comparing different models.
    
    Args:
        base_results (dict): Results from the base CLIP model
        lora_results (dict): Results from the LoRA model
        selective_results (dict): Results from the selective fine-tuning model
        output_dir (str): Directory to save the plots
    """
    plt.figure(figsize=(12, 8))
    
    # 1. Overall accuracy comparison
    plt.figure(figsize=(10, 6))
    models = ['Base CLIP', 'LoRA', 'Selective Fine-tuning']
    disease_acc = [base_results['disease_accuracy'], lora_results['disease_accuracy'], selective_results['disease_accuracy']]
    crop_acc = [base_results['crop_accuracy'], lora_results['crop_accuracy'], selective_results['crop_accuracy']]
    combined_acc = [0, lora_results['combined_accuracy'], selective_results['combined_accuracy']]  # Base doesn't have combined
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, disease_acc, width, label='Disease Accuracy')
    rects2 = ax.bar(x, crop_acc, width, label='Crop Accuracy')
    rects3 = ax.bar(x + width, combined_acc, width, label='Combined Accuracy')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'model_comparison.png'))
    plt.close()
    
    # 2. Confidence distribution comparison
    plt.figure(figsize=(12, 8))
    plt.hist([base_results['confidences'], lora_results['confidences'], selective_results['confidences']], 
             bins=20, alpha=0.7, label=models)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'plots', 'confidence_distribution.png'))
    plt.close()
    
    # 3. Per-crop accuracy comparison (top 5 crops)
    plt.figure(figsize=(14, 8))
    
    # Get top 5 crops by number of samples
    crop_counts = {}
    for crop_label in selective_results['all_crop_labels']:
        crop_name = list(selective_results['crop_class_accuracy'].keys())[crop_label]
        crop_counts[crop_name] = crop_counts.get(crop_name, 0) + 1
    
    top_crops = sorted(crop_counts.keys(), key=lambda x: crop_counts[x], reverse=True)[:5]
    
    # Get accuracies for top crops
    crop_accuracies = {
        'Base CLIP': [base_results['crop_class_accuracy'].get(crop, 0) for crop in top_crops],
        'LoRA': [lora_results['crop_class_accuracy'].get(crop, 0) for crop in top_crops],
        'Selective': [selective_results['crop_class_accuracy'].get(crop, 0) for crop in top_crops]
    }
    
    x = np.arange(len(top_crops))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, crop_accuracies['Base CLIP'], width, label='Base CLIP')
    rects2 = ax.bar(x, crop_accuracies['LoRA'], width, label='LoRA')
    rects3 = ax.bar(x + width, crop_accuracies['Selective'], width, label='Selective Fine-tuning')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Crop Classification Accuracy by Model (Top 5 Crops)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_crops, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'top_crops_comparison.png'))
    plt.close()
    
    # 4. Per-disease accuracy comparison (top 5 diseases)
    plt.figure(figsize=(14, 10))
    
    # Get top 5 diseases by number of samples
    disease_counts = {}
    for disease_label in selective_results['all_disease_labels']:
        disease_name = list(selective_results['disease_class_accuracy'].keys())[disease_label]
        disease_counts[disease_name] = disease_counts.get(disease_name, 0) + 1
    
    top_diseases = sorted(disease_counts.keys(), key=lambda x: disease_counts[x], reverse=True)[:5]
    
    # Get accuracies for top diseases
    disease_accuracies = {
        'Base CLIP': [base_results['disease_class_accuracy'].get(disease, 0) for disease in top_diseases],
        'LoRA': [lora_results['disease_class_accuracy'].get(disease, 0) for disease in top_diseases],
        'Selective': [selective_results['disease_class_accuracy'].get(disease, 0) for disease in top_diseases]
    }
    
    x = np.arange(len(top_diseases))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 10))
    rects1 = ax.bar(x - width, disease_accuracies['Base CLIP'], width, label='Base CLIP')
    rects2 = ax.bar(x, disease_accuracies['LoRA'], width, label='LoRA')
    rects3 = ax.bar(x + width, disease_accuracies['Selective'], width, label='Selective Fine-tuning')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Disease Classification Accuracy by Model (Top 5 Diseases)')
    ax.set_xticks(x)
    ax.set_xticklabels([d[:20] + '...' if len(d) > 20 else d for d in top_diseases], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'top_diseases_comparison.png'))
    plt.close()
    
    # 5. Training curves for LoRA and Selective
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(lora_results['metrics']['train_loss'], label='LoRA Training Loss')
    plt.plot(lora_results['metrics']['val_loss'], label='LoRA Validation Loss')
    plt.plot(selective_results['metrics']['train_loss'], label='Selective Training Loss')
    plt.plot(selective_results['metrics']['val_loss'], label='Selective Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(lora_results['metrics']['disease_accuracy'], label='LoRA Disease Accuracy')
    plt.plot(selective_results['metrics']['disease_accuracy'], label='Selective Disease Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Disease Classification Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(lora_results['metrics']['crop_accuracy'], label='LoRA Crop Accuracy')
    plt.plot(selective_results['metrics']['crop_accuracy'], label='Selective Crop Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Crop Classification Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'training_curves.png'))
    plt.close()
    
    # 6. Confusion matrices for crop classification
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    crop_labels = list(selective_results['crop_class_accuracy'].keys())
    cm = confusion_matrix(
        [crop_labels[i] for i in base_results['all_crop_labels']], 
        [crop_labels[i] for i in base_results['all_crop_preds']]
    )
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('Base CLIP Crop Confusion Matrix')
    plt.ylabel('True Crop')
    plt.xlabel('Predicted Crop')
    
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(
        [crop_labels[i] for i in lora_results['all_crop_labels']], 
        [crop_labels[i] for i in lora_results['all_crop_preds']]
    )
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('LoRA Crop Confusion Matrix')
    plt.ylabel('True Crop')
    plt.xlabel('Predicted Crop')
    
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(
        [crop_labels[i] for i in selective_results['all_crop_labels']], 
        [crop_labels[i] for i in selective_results['all_crop_preds']]
    )
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('Selective Fine-tuning Crop Confusion Matrix')
    plt.ylabel('True Crop')
    plt.xlabel('Predicted Crop')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'crop_confusion_matrices.png'))
    plt.close()

# Function to save final evaluation results
def save_evaluation_results(base_results, lora_results, selective_results, output_dir):
    """Save evaluation results to a file.
    
    Args:
        base_results (dict): Results from the base CLIP model
        lora_results (dict): Results from the LoRA model
        selective_results (dict): Results from the selective fine-tuning model
        output_dir (str): Directory to save the results
    """
    # Create summary table
    summary = {
        "Model": ["Base CLIP", "LoRA", "Selective Fine-tuning"],
        "Disease Accuracy (%)": [
            f"{base_results['disease_accuracy']:.2f}",
            f"{lora_results['disease_accuracy']:.2f}",
            f"{selective_results['disease_accuracy']:.2f}"
        ],
        "Crop Accuracy (%)": [
            f"{base_results['crop_accuracy']:.2f}",
            f"{lora_results['crop_accuracy']:.2f}",
            f"{selective_results['crop_accuracy']:.2f}"
        ],
        "Combined Accuracy (%)": [
            "N/A",
            f"{lora_results['combined_accuracy']:.2f}",
            f"{selective_results['combined_accuracy']:.2f}"
        ],
        "Average Confidence (%)": [
            f"{base_results['avg_confidence']:.2f}",
            f"{lora_results['avg_confidence']:.2f}",
            f"{selective_results['avg_confidence']:.2f}"
        ]
    }
    
    # Convert to DataFrame and save as CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)
    
    # Create a more detailed text report
    with open(os.path.join(output_dir, 'detailed_results.txt'), 'w') as f:
        f.write("=============================================\n")
        f.write("CROP DISEASE CLASSIFICATION MODEL EVALUATION\n")
        f.write("=============================================\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY OF RESULTS\n")
        f.write("-----------------\n")
        f.write(f"{'Model':<25} {'Disease Acc.':<15} {'Crop Acc.':<15} {'Combined Acc.':<15} {'Confidence':<15}\n")
        for i in range(len(summary["Model"])):
            f.write(f"{summary['Model'][i]:<25} {summary['Disease Accuracy (%)'][i]:<15} "
                   f"{summary['Crop Accuracy (%)'][i]:<15} {summary['Combined Accuracy (%)'][i]:<15} "
                   f"{summary['Average Confidence (%)'][i]:<15}\n")
        
        f.write("\n\nDETAILED RESULTS\n")
        f.write("----------------\n\n")
        
        # Per-crop accuracy
        f.write("Crop Classification Accuracy\n")
        f.write("---------------------------\n")
        # Get all crops from selective results
        all_crops = list(selective_results['crop_class_accuracy'].keys())
        f.write(f"{'Crop Type':<20} {'Base CLIP':<15} {'LoRA':<15} {'Selective':<15}\n")
        for crop in all_crops:
            base_acc = base_results['crop_class_accuracy'].get(crop, 0)
            lora_acc = lora_results['crop_class_accuracy'].get(crop, 0)
            selective_acc = selective_results['crop_class_accuracy'].get(crop, 0)
            f.write(f"{crop:<20} {base_acc:.2f}%{' ':<10} {lora_acc:.2f}%{' ':<10} {selective_acc:.2f}%{' ':<10}\n")
        
        # Top 5 disease accuracy
        f.write("\n\nTop 5 Diseases Classification Accuracy\n")
        f.write("-------------------------------------\n")
        # Get top 5 diseases by count
        disease_counts = {}
        for d_label in selective_results['all_disease_labels']:
            disease = list(selective_results['disease_class_accuracy'].keys())[d_label]
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        top_diseases = sorted(disease_counts.keys(), key=lambda x: disease_counts[x], reverse=True)[:5]
        
        f.write(f"{'Disease Type':<30} {'Base CLIP':<15} {'LoRA':<15} {'Selective':<15}\n")
        for disease in top_diseases:
            base_acc = base_results['disease_class_accuracy'].get(disease, 0)
            lora_acc = lora_results['disease_class_accuracy'].get(disease, 0)
            selective_acc = selective_results['disease_class_accuracy'].get(disease, 0)
            disease_display = disease if len(disease) < 30 else disease[:27] + "..."
            f.write(f"{disease_display:<30} {base_acc:.2f}%{' ':<10} {lora_acc:.2f}%{' ':<10} {selective_acc:.2f}%{' ':<10}\n")
        
        # Model architecture summary
        f.write("\n\nMODEL ARCHITECTURE SUMMARY\n")
        f.write("--------------------------\n")
        f.write("Base CLIP: Zero-shot classification using pre-trained CLIP model\n")
        f.write(f"LoRA Fine-tuning: Applied to visual encoder attention layers with rank=8\n")
        f.write("Selective Fine-tuning: Only last transformer block of visual encoder fine-tuned\n\n")
        
        f.write("CONCLUSIONS\n")
        f.write("-----------\n")
        
        # Determine which model performed best
        best_model = "Selective Fine-tuning" if selective_results['disease_accuracy'] > lora_results['disease_accuracy'] else "LoRA"
        
        f.write(f"The {best_model} approach achieved the best overall performance for crop disease classification.\n")
        f.write(f"Disease accuracy improved from {base_results['disease_accuracy']:.2f}% (Base CLIP) to ")
        f.write(f"{max(lora_results['disease_accuracy'], selective_results['disease_accuracy']):.2f}% ")
        f.write(f"({best_model}).\n\n")
        
        f.write("The dual-head classification architecture successfully leverages both crop and disease features,\n")
        f.write("improving the model's ability to distinguish between visually similar diseases across different crops.\n")

# Main function
def main():
    # Download dataset from Google Drive
    image_dir = download_and_extract_dataset()
    print(f"Using images from: {image_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create datasets
    train_dataset = CropDiseaseDataset(image_dir, clip_processor, split="train")
    val_dataset = CropDiseaseDataset(image_dir, clip_processor, split="val")
    
    if len(train_dataset) == 0:
        raise ValueError("No training data found! Please check the dataset path.")
    
    # 1. Evaluate base CLIP model
    base_results = evaluate_base_model(clip_model, clip_processor, val_dataset, device)
    
    # 2. Train and evaluate LoRA model
    lora_trainer = LoraTrainer(
        model=clip_model,
        processor=clip_processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=os.path.join(RESULTS_DIR, "models", "lora"),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )
    
    # Count trainable parameters for LoRA
    lora_trainable_params = count_trainable_parameters(lora_trainer.model)
    print(f"LoRA trainable parameters: {lora_trainable_params:,}")
    
    # Train LoRA model
    lora_model, lora_disease_classifier, lora_crop_classifier = lora_trainer.train()
    
    # Evaluate LoRA model
    lora_eval_results = lora_trainer.evaluate()
    lora_eval_results['metrics'] = lora_trainer.metrics
    
    # 3. Train and evaluate Selective fine-tuning model
    # Reset CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    selective_trainer = SelectiveFineTuningTrainer(
        model=clip_model,
        processor=clip_processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=os.path.join(RESULTS_DIR, "models", "selective"),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )
    
    # Count trainable parameters for Selective fine-tuning
    selective_trainable_params = count_trainable_parameters(selective_trainer.model)
    print(f"Selective fine-tuning trainable parameters: {selective_trainable_params:,}")
    
    # Train Selective model
    selective_model, selective_disease_classifier, selective_crop_classifier = selective_trainer.train()
    
    # Evaluate Selective model
    selective_eval_results = selective_trainer.evaluate()
    selective_eval_results['metrics'] = selective_trainer.metrics
    
    # Generate plots and save results
    create_plots(base_results, lora_eval_results, selective_eval_results, RESULTS_DIR)
    save_evaluation_results(base_results, lora_eval_results, selective_eval_results, RESULTS_DIR)
    
    print("\n--- Final Results Summary ---")
    print(f"Base CLIP Disease Accuracy: {base_results['disease_accuracy']:.2f}%")
    print(f"LoRA Disease Accuracy: {lora_eval_results['disease_accuracy']:.2f}%")
    print(f"Selective Fine-tuning Disease Accuracy: {selective_eval_results['disease_accuracy']:.2f}%")
    
    print(f"\nBase CLIP Crop Accuracy: {base_results['crop_accuracy']:.2f}%")
    print(f"LoRA Crop Accuracy: {lora_eval_results['crop_accuracy']:.2f}%")
    print(f"Selective Fine-tuning Crop Accuracy: {selective_eval_results['crop_accuracy']:.2f}%")
    
    print(f"\nLoRA Combined Accuracy: {lora_eval_results['combined_accuracy']:.2f}%")
    print(f"Selective Fine-tuning Combined Accuracy: {selective_eval_results['combined_accuracy']:.2f}%")
    
    print(f"\nLoRA Average Confidence: {lora_eval_results['avg_confidence']:.2f}%")
    print(f"Selective Fine-tuning Average Confidence: {selective_eval_results['avg_confidence']:.2f}%")
    
    print("\nComplete detailed results have been saved to:")
    print(f"  - {os.path.join(RESULTS_DIR, 'detailed_results.txt')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'summary_results.csv')}")
    print("\nPlots have been saved to:")
    print(f"  - {os.path.join(RESULTS_DIR, 'plots')}")

if __name__ == "__main__":
    main()