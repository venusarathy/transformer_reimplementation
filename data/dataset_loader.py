import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader

# DatasetLoader for IMDb (example)
class IMDbDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(list(self.dataset))
    
    def __getitem__(self, idx):
        text, label = list(self.dataset)[idx]
        text = self.tokenizer(text).numpy()
        label = torch.tensor(label.numpy(), dtype=torch.float32)
        return torch.tensor(text, dtype=torch.long), label

def load_dataset():
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    return train_dataset, test_dataset

