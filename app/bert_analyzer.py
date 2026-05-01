"""
BERT Classification Module
Handles question classification and analysis using BERT model
"""

import os

# Must be set BEFORE importing transformers to prevent it from loading TensorFlow.
# TensorFlow conflicts with paddlepaddle-gpu due to incompatible protobuf versions.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict, Tuple


class QuestionDataset(Dataset):
    """PyTorch Dataset for question classification"""
    
    def __init__(self, questions: List[str], labels: List[int], 
                 tokenizer, max_length: int = 64):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    """BERT-based question classifier"""
    
    def __init__(self, bert_model_name: str, num_classes: int):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.fc(cls_output)


class BERTQuestionAnalyzer:
    """Main analyzer combining BERT model with question classification"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None):
        """
        Initialize BERT analyzer
        
        Args:
            model_name: Pre-trained BERT model name
            device: 'cuda' or 'cpu'
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_map = {}
        self.reverse_label_map = {}
        self.max_length = 64
    
    def load_training_data(self, csv_path: str) -> Tuple[List[str], List[int]]:
        """
        Load training data from CSV
        
        Args:
            csv_path: Path to CSV with 'Question Text' and 'Chapter_name' columns
            
        Returns:
            Tuple of (questions, labels_encoded)
        """
        df = pd.read_csv(csv_path)
        questions = df['Question Text'].tolist()
        chapter_labels = df['Chapter_name'].tolist()
        
        # Create label mappings
        unique_labels = sorted(set(chapter_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
        
        # Encode labels
        labels_encoded = [self.label_map[label] for label in chapter_labels]
        
        return questions, labels_encoded
    
    def train(self, questions: List[str], labels_encoded: List[int], 
              batch_size: int = 8, num_epochs: int = 13, learning_rate: float = 4e-5,
              val_split: float = 0.2):
        """
        Train the BERT classifier
        
        Args:
            questions: List of question texts
            labels_encoded: List of encoded labels
            batch_size: Training batch size
            num_epochs: Number of epochs
            learning_rate: Learning rate
            val_split: Validation split ratio
        """
        from torch.utils.data import DataLoader, random_split
        
        num_classes = len(self.label_map)
        
        # Initialize model
        self.model = BERTClassifier(self.model_name, num_classes).to(self.device)
        
        # Create dataset
        dataset = QuestionDataset(questions, labels_encoded, self.tokenizer, self.max_length)
        
        # Split into train/val
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    def predict(self, questions: List[str]) -> List[Dict[str, str]]:
        """
        Predict chapter labels for questions
        
        Args:
            questions: List of question texts
            
        Returns:
            List of dicts with 'question', 'predicted_chapter', and 'confidence'
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a model.")
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for question in questions:
                # Tokenize
                encoding = self.tokenizer.encode_plus(
                    question,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids, attention_mask)
                logits = outputs[0]
                probabilities = torch.softmax(logits, dim=1)
                
                predicted_label_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_label_idx].item()
                
                predicted_chapter = self.reverse_label_map[predicted_label_idx]
                
                results.append({
                    'question': question,
                    'predicted_chapter': predicted_chapter,
                    'confidence': f"{confidence:.2%}"
                })
        
        return results
    
    def save_model(self, save_path: str):
        """Save trained model"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save label maps
        import json
        with open(os.path.join(save_path, 'label_map.json'), 'w') as f:
            json.dump(self.label_map, f)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        import json
        num_classes = len(self.label_map) if self.label_map else 29  # Default from your data
        
        self.model = BERTClassifier(self.model_name, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
        
        with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
            self.label_map = json.load(f)
            self.reverse_label_map = {int(v): k for k, v in self.label_map.items()}
