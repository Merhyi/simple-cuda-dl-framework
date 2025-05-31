import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt # For optional visualization
import time

# --- 1. Data Generation (Similar to SimpleBinaryDataset) ---
def generate_simple_binary_data(n_samples_total=250, noise=0.1, non_linear=False, random_state=None):
    """
    Generates a simple 2D binary classification dataset.
    Args:
        n_samples_total (int): Total number of samples.
        noise (float): Standard deviation of Gaussian noise.
        non_linear (bool): If True, generates a slightly non-linear dataset.
        random_state (int, optional): Seed for reproducibility.
    Returns:
        X (torch.Tensor): Features (n_samples, 2).
        y (torch.Tensor): Labels (n_samples, 1).
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    X_np = np.random.uniform(-2.0, 2.0, size=(n_samples_total, 2))
    y_np = np.zeros((n_samples_total, 1))

    for i in range(n_samples_total):
        x1_base, x2_base = X_np[i, 0], X_np[i, 1]
        label_int = 0
        if not non_linear: # Linearly separable (approx)
            offset = 0.2
            if x1_base > x2_base + offset:
                label_int = 1
            elif x1_base < x2_base - offset:
                label_int = 0
            else:
                label_int = 1 if np.random.rand() > 0.5 else 0 # Fuzzy boundary
        else: # Non-linearly separable
            if (x1_base > 0 and x2_base > 0) or (x1_base < -0.5 and x2_base < -0.5):
                label_int = 1
            else:
                label_int = 0
        y_np[i] = label_int

    X_np += np.random.normal(0, noise, X_np.shape)

    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)

# --- PyTorch Dataset and DataLoader (Optional but good practice) ---
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. Neural Network Definition ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # Shape(2, 4) -> input_size=2, hidden_size=4
        self.tanh = nn.Tanh()                             # TanhActivation
        self.linear2 = nn.Linear(hidden_size, output_size)# Shape(4, 1) -> hidden_size=4, output_size=1
        self.sigmoid = nn.Sigmoid()                       # SigmoidActivation

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# --- 3. Training Parameters and Setup ---
# Corresponds to C++:
# size_t num_batches = 5; (This will be handled by DataLoader if used, or n_samples_total / batch_size)
# size_t samples_per_batch = 50;
# float noise = 0.1f;
# const float learning_rate = 0.01f;

# Python/PyTorch parameters
n_total_samples = 250 # num_batches * samples_per_batch = 5 * 50 (from C++ example logic)
batch_size_py = 50    # samples_per_batch
noise_py = 0.15         # Matches C++ example in question for SimpleBinaryDataset usage
generate_non_linear_data = True # Matches C++: SimpleBinaryDataset(..., false) -> actually non-linear based on prompt, true for linear
learning_rate_py = 0.01
num_epochs_py = 1001

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 4. Generate Data and Create DataLoader ---
print("Generating dataset...")
X_data, y_data = generate_simple_binary_data(
    n_samples_total=n_total_samples,
    noise=noise_py,
    non_linear=generate_non_linear_data, # C++ code used 'false' for dataset constructor, meaning non_linear
    random_state=42 # For reproducibility
)
print(f"Dataset generated. X shape: {X_data.shape}, y shape: {y_data.shape}")

# Split into training and a small test set (mimicking C++ last batch for accuracy)
# For a more robust setup, use sklearn.model_selection.train_test_split
num_test_samples = batch_size_py # Size of the last batch in C++
X_train, y_train = X_data[:-num_test_samples], y_data[:-num_test_samples]
X_test, y_test = X_data[-num_test_samples:], y_data[-num_test_samples:]

print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

train_dataset = SimpleDataset(X_train, y_train)
# num_batches for training loop: len(train_dataset) // batch_size_py
# C++ code uses `dataset.getNumOfBatches() - 1` for training loop,
# which is (n_total_samples / batch_size_py) - 1
# The PyTorch DataLoader will handle batching automatically.
# Number of batches in C++ was 5 (total), training on 4. Total samples 250. Training on 200.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_py, shuffle=True)
num_training_batches = len(train_loader) # This is equivalent to dataset.getNumOfBatches() - 1 from C++

# --- 5. Initialize Model, Loss, Optimizer ---
input_dim = 2
hidden_dim = 4 # From LinearLayer("linear_1", Shape(2, 4))
output_dim = 1

model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.BCELoss() # BCECost
optimizer = optim.Adam(model.parameters(), lr=learning_rate_py) # AdamOptimizer

print("\nModel Architecture:")
print(model)
print(f"\nTraining with {num_training_batches} batches per epoch.")

# --- 6. Training Loop ---
print("\nStarting Training...")
total_train_start_time = time.time()
for epoch in range(num_epochs_py):
    epoch_loss = 0.0
    model.train() # Set model to training mode

    for i, (batch_features, batch_labels) in enumerate(train_loader):
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        # Backward and optimize
        optimizer.zero_grad() # Corresponds to zeroing gradients
        loss.backward()       # Corresponds to nn.backprop()
        optimizer.step()      # Corresponds to optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0 or epoch == 0:
        avg_epoch_loss = epoch_loss / num_training_batches
        print(f"Epoch [{epoch+1}/{num_epochs_py}], Loss: {avg_epoch_loss:.6f}")
total_train_end_time = time.time()
total_train_duration = total_train_end_time - total_train_start_time
print(f"Total training time: {total_train_duration:.4f} seconds")


# --- 7. Compute Accuracy on the Test Set ---
model.eval() # Set model to evaluation mode
with torch.no_grad(): # No need to track gradients for evaluation
    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)

    test_outputs = model(X_test_dev)
    # Convert outputs to 0 or 1 predictions
    predicted_labels = (test_outputs > 0.5).float()

    correct_predictions = (predicted_labels == y_test_dev).sum().item()
    total_test_samples = y_test_dev.size(0)
    accuracy = correct_predictions / total_test_samples

    print(f"\nAccuracy on the test set ({total_test_samples} samples): {accuracy * 100:.2f}%")


# --- Optional: Visualize the decision boundary (for 2D data) ---
def plot_decision_boundary(model, X, y, device_to_use):
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Convert meshgrid to tensor and predict
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device_to_use)
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = (Z > 0.5).float() # Apply threshold
        Z = Z.cpu().numpy()
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu().squeeze(), cmap=plt.cm.Spectral, edgecolors='k', s=30)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot for the entire dataset (train + test) to see how well it learned
# plot_decision_boundary(model, X_data.to("cpu"), y_data.to("cpu"), device) # Pass the device model is on
plot_decision_boundary(model, X_data, y_data, device)