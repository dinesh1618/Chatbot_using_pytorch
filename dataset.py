import torch
from torch.utils.data import Dataset

class ChatBot(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, item):
        return self.X_train[item], self.y_train[item]