import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from flask import Flask, request, jsonify
from preprocess import preprocessed_data, tfidf_matrix
from chatbot import Chatbot

app = Flask(__name__)

# Initialize the chatbot with the path to the CSV file
chatbot = Chatbot('responses.csv')

@app.route('/')
def home():
    return "Welcome to the Restaurant Chatbot!"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    user_id = request.json.get('user_id', 'default_user')  # Default to 'default_user' if no user_id provided
    
    # Pass both user_input and user_id to the get_response method
    response = chatbot.get_response(user_input, user_id)
    
    return jsonify({"response": response})

# Define your PyTorch model
class ChatbotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Example Dataset class
class ChatDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx])

# Dummy labels for example
labels = [0] * len(preprocessed_data)  # Replace with actual labels

# Create dataset and dataloader
dataset = ChatDataset(tfidf_matrix.toarray(), labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model, criterion, and optimizer
model = ChatbotModel(input_dim=tfidf_matrix.shape[1], hidden_dim=128, output_dim=5)  # Adjust output_dim as needed
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model():
    for epoch in range(10):  # Adjust the number of epochs as needed
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == '__main__':
    # You can choose to train the model before running the app or during runtime
    train_model()  # Call this function to start training
    app.run(debug=True)
