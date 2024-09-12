import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorflow_datasets as tfds
import tensorflow as tf
from models.transformer_model import TransformerClassifier
from data.dataset_loader import IMDbDataset, load_dataset

# Prepare device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer using TensorFlow's TextVectorization
tokenizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_mode='int', output_sequence_length=250)

def prepare_data():
    """
    Loads and tokenizes the dataset.
    """
    # Load dataset using TensorFlow Datasets
    train_dataset, test_dataset = load_dataset()

    # Tokenize the training data
    train_text = train_dataset.map(lambda text, label: text)
    tokenizer.adapt(train_text)

    # Wrap TensorFlow Datasets into PyTorch-compatible datasets
    train_data = IMDbDataset(train_dataset, tokenizer)
    test_data = IMDbDataset(test_dataset, tokenizer)

    # Prepare PyTorch DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """
    The training loop for the model.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(texts).squeeze(1)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

def evaluate_model(model, test_loader):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze(1)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {correct / total:.4f}")

def main():
    """
    Main function to run the training and evaluation of the Transformer model.
    """
    # Load the dataset
    train_loader, test_loader = prepare_data()

    # Model hyperparameters
    vocab_size = 10000
    embedding_dim = 128
    num_heads = 8
    ff_hidden_dim = 512
    num_layers = 2

    # Initialize the Transformer model
    model = TransformerClassifier(vocab_size, embedding_dim, num_heads, ff_hidden_dim, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for binary classification (sentiment analysis)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, epochs=5)

    # Evaluate the model on the test set
    print("Evaluating the model...")
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()

