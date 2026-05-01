import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'final_questions.csv'
df = pd.read_csv(file_path)
questions = df['Question Text'].tolist()
labels = df['Chapter_name'].tolist()

#label_map = {label: idx for idx, label in enumerate(set(labels))}
#map_to_label ={idx:label for idx, label in enumerate(set(labels))}
map_to_label={'Generic Algorithms and STL': 0,
 'Associative Containers': 1,
 'Virtual Functions': 2,
 'Templates': 3,
 'Templates ': 4,
 'Inheritance': 5,
 'Copy Control': 6,
 'Sequential Containers': 7,
 'Multifile Programs': 8,
 'Functions': 9,
 'Objects and Classes': 10,
 'Structures': 11,
 'Loops and Decisions': 12,
 'Pointers and Dynamic Memory': 13,
 'String,Vectors, and Arrays': 14,
 'Getting started': 15,
 'Object-Oriented Programming': 16,
 'Operator Overloading': 17,
 'Expressions': 18,
 'Statements': 19,
 'Specialised Tools and Techniques': 20,
 'C++ Programming Basics': 21,
 'Streams and IO Library': 22,
 'Tools for Large Programs': 23,
 'Strings, Vectors, and Arrays': 24,
 'Generic Algorithms': 25,
 'Pointers': 26,
 'Variable and Basic types': 27,
 'Specialised Library Facilities': 28}
new_map_to_label={label:idx for idx,label in map_to_label.items()}
#labels = [label_map[label] for label in labels]
num_classes = len(map_to_label)
# Load BERT tokenizer and define constants
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 64
batch_size = 8
learning_rate = 4e-5 # (1e-5 =55%) (3e-5 = 65%) (8e-5=67%)
num_epochs = 13
class QuestionDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_length):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        label = self.labels[idx]

        encoding = (self.tokenizer.encode_plus(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ))

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.fc(cls_output)
model = BERTClassifier('bert-base-uncased', num_classes).to(device)
dataset = QuestionDataset(questions, labels, tokenizer, max_length)
train_size = int(0.8* len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)# why we used adamW
#why we used cross entropy loss
criterion = nn.CrossEntropyLoss()
def predict_question(question, model, tokenizer, device, max_length=32):
    # Preprocess the question (tokenize)
    encoding = tokenizer.encode_plus(
        question,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()
SModel =torch.load('model_complete.pth')
def question_to_class(question):
    predicted_class = predict_question(question, SModel, tokenizer, device)
    return new_map_to_label[predicted_class]

# Streamlit app layout
st.title("Question  Analysis App")
st.write("Ask me any question, and I'll do my best to classify it!")

# Input field for the user's question
question = st.text_input("Enter your question:")

# Generate and display the answer
if question:
    answer = question_to_class(question)
    st.write("### Name of the chapter:")
    st.write(answer)