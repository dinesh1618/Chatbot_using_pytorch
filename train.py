import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import TextProcessing
from dataset import ChatBot
from models import NeuralNet


from dotenv import load_dotenv
load_dotenv()

with open("intents.json", "r") as f:
    data = json.load(f)

model = os.getenv("MODEL")
tags = []
all_words = []
xy = []
TP = TextProcessing(model)
for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = TP.SentTokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
all_words = [TP.Stemmimg (token) for token in all_words]
all_words = TP.StopWordsRemoval(all_words, extra_tokens={'?', '!', '.'})
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for pattern, tag in xy:
    bow = TP.BogofWords(pattern, all_words)
    X_train.append(bow)
    label = tags.index(tag)
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

INPUT_SIZE=len(X_train[0])
HIDDEN_SIZE=8
OUTPUT_SIZE=len(tags)
TRAIN_BATCH = eval(os.getenv("TRAIN_BATCH"))
EPOCHS = eval(os.getenv("EPOCHS"))
LEARNING_RATE = eval(os.getenv("LEARNING_RATE"))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = ChatBot(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=0)
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.to(DEVICE)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in tqdm(range(EPOCHS)):
    model.train()
    for words, label in train_loader:
        words = words.to(DEVICE)
        label = torch.tensor(label, dtype=torch.long).to(DEVICE)
        output = model(words)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
        
print(f'final loss: {loss.item():.4f}')
data = {
"model_state": model.state_dict(),
"input_size": INPUT_SIZE,
"hidden_size": HIDDEN_SIZE,
"output_size": OUTPUT_SIZE,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')