import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt # For optional visualization
import time # Import the time module

# --- 1. Data Generation (Similar to LinearRegressionDataset) ---
def generate_linear_regression_data(n_samples_total=200,
                                    n_features=2,
                                    true_weights=None, # List or array of true weights
                                    true_bias=0.5,
                                    noise_std=0.2,
                                    feature_range=(-5.0, 5.0),
                                    random_state=None):
    """
    Generates data for a linear regression task.
    y = Xw + b + noise
    Args:
        n_samples_total (int): Total number of samples.
        n_features (int): Number of input features for X.
        true_weights (list/np.array, optional): True weights for X. If None, generated.
        true_bias (float): True bias term.
        noise_std (float): Standard deviation of Gaussian noise added to y.
        feature_range (tuple): (min, max) for generating X features.
        random_state (int, optional): Seed for reproducibility.
    Returns:
        X (torch.Tensor): Features (n_samples, n_features).
        y (torch.Tensor): Targets (n_samples, 1).
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    X_np = np.random.uniform(feature_range[0], feature_range[1], size=(n_samples_total, n_features))

    if true_weights is None:
        # Generate some example weights if not provided
        # For C++ example: true_w_param=2.5f, for 2 features, second weight was derived
        if n_features == 1:
            w_np = np.array([2.5]).reshape(-1, 1) # Example default
        elif n_features == 2:
             w_np = np.array([2.5, 2.5 * 0.75]).reshape(-1,1) # Matches C++ example implicit generation
        else:
            w_np = np.random.randn(n_features, 1) * 2.0 # Generic random weights
    elif isinstance(true_weights, (list, tuple)):
        w_np = np.array(true_weights).reshape(-1, 1)
    else: # Assume it's already a numpy array
        w_np = true_weights.reshape(-1, 1)

    if w_np.shape[0] != n_features:
        raise ValueError(f"Number of true_weights ({w_np.shape[0]}) must match n_features ({n_features})")

    y_true_noiseless_np = X_np @ w_np + true_bias
    noise_np = np.random.normal(0, noise_std, size=(n_samples_total, 1))
    y_np = y_true_noiseless_np + noise_np

    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)


# --- PyTorch Dataset and DataLoader (Good practice) ---
from torch.utils.data import Dataset, DataLoader

class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. Neural Network Definition ---
class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # Shape(2, 10) -> input=2, hidden=10
        self.tanh = nn.Tanh()                             # TanhActivation
        self.linear2 = nn.Linear(hidden_size, output_size)# Shape(10, 1) -> hidden=10, output=1
        # No output activation for linear regression

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        return out

# --- 3. Training Parameters and Setup ---
# C++ values:
# size_t num_batches = 10;
# size_t samples_per_batch = 20;
# float noise_level = 0.2f;
# size_t features_2d = 2;
# true_w = 2.5f, true_b = 1.0f (for 2D LinearRegressionDataset)
# const float learning_rate = 0.1f; (This is likely too high for MSE with SGD)
# const float momentum = 0.0f;

# Python/PyTorch parameters
n_features_py = 2
n_total_samples_py = 200 # num_batches * samples_per_batch = 10 * 20
batch_size_py = 20       # samples_per_batch
noise_std_py = 0.2
true_w1 = 2.5
true_w2 = 2.5 * 0.75 # Derived in C++ LinearRegressionDataset for 2 features
true_b_py = 1.0
# learning_rate_py = 0.1f # Original from C++, likely too high
learning_rate_py = 0.005 # A more reasonable starting point for Adam with MSE
# For SGD, might need even smaller like 0.001 or 0.0001
momentum_py = 0.0 # For SGD if used
num_epochs_py = 2001 # Matches C++

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 4. Generate Data and Create DataLoader ---
print("Generating dataset...")
# For 2D, pass the true weights explicitly or ensure generator logic matches
true_weights_py = [true_w1, true_w2]
X_data_reg, y_data_reg = generate_linear_regression_data(
    n_samples_total=n_total_samples_py,
    n_features=n_features_py,
    true_weights=true_weights_py,
    true_bias=true_b_py,
    noise_std=noise_std_py,
    random_state=int(np.random.rand() * 10000) # Similar to srand(time(NULL))
)
print(f"Dataset generated. X shape: {X_data_reg.shape}, y shape: {y_data_reg.shape}")

# Split into training and a small test set (mimicking C++ last batch for testing)
num_test_samples_reg = batch_size_py # C++ used last batch for test_cost
X_train_reg, y_train_reg = X_data_reg[:-num_test_samples_reg], y_data_reg[:-num_test_samples_reg]
X_test_reg, y_test_reg = X_data_reg[-num_test_samples_reg:], y_data_reg[-num_test_samples_reg:]

print(f"Training set size: {X_train_reg.shape[0]}, Test set size: {X_test_reg.shape[0]}")

train_dataset_reg = RegressionDataset(X_train_reg, y_train_reg)
train_loader_reg = DataLoader(dataset=train_dataset_reg, batch_size=batch_size_py, shuffle=True)
num_training_batches_reg = len(train_loader_reg)

# --- 5. Initialize Model, Loss, Optimizer ---
input_dim_reg = n_features_py
hidden_dim_reg = 10 # From LinearLayer("linear_1", Shape(2, 10))
output_dim_reg = 1

model_reg = RegressionNN(input_dim_reg, hidden_dim_reg, output_dim_reg).to(device)

# Initialize weights (important for Tanh)
def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model_reg.apply(init_weights_xavier_uniform) # Apply Xavier initialization

criterion_reg = nn.MSELoss() # MSECost

# Optimizer: SGD or Adam
# C++ used SGD, but Adam is often more robust for initial tuning.
# optimizer_reg = optim.SGD(model_reg.parameters(), lr=learning_rate_py, momentum=momentum_py)
optimizer_reg = optim.Adam(model_reg.parameters(), lr=learning_rate_py)


print("\nModel Architecture:")
print(model_reg)
print(f"\nTraining with {num_training_batches_reg} batches per epoch.")

# --- 6. Training Loop ---
print("\nStarting Training...")
total_train_start_time = time.time()
for epoch in range(num_epochs_py):
    epoch_loss = 0.0
    model_reg.train() # Set model to training mode

    # The C++ code iterates `dataset.getNumOfBatches() - 1` times.
    # Our train_loader is based on X_train_reg, which already excludes the last batch.
    for i, (batch_features, batch_targets) in enumerate(train_loader_reg):
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass
        outputs = model_reg(batch_features)
        loss = criterion_reg(outputs, batch_targets)

        # Backward and optimize
        optimizer_reg.zero_grad() # Clear previous gradients
        loss.backward()           # Compute gradients
        optimizer_reg.step()          # Update weights

        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0 or epoch == 0 :
        # The C++ code divides by total batches, not just training batches for print.
        # To match C++ printed cost: cost / (n_total_samples_py / batch_size_py)
        # More accurate average training loss for the epoch: epoch_loss / num_training_batches_reg
        avg_epoch_loss_printed_cpp_style = epoch_loss * batch_size_py / n_total_samples_py # Approximation
        avg_epoch_loss_train = epoch_loss / num_training_batches_reg
        print(f"Epoch [{epoch+1}/{num_epochs_py}], Avg Train Loss: {avg_epoch_loss_train:.6f}")
        # C++ cost was sum of batch costs, then divided by total num batches.
        # So epoch_loss is sum of batch losses. total_num_batches_cpp = n_total_samples_py / batch_size_py
        # print(f"Epoch [{epoch+1}/{num_epochs_py}], Cpp-Style Cost: {epoch_loss / (n_total_samples_py / batch_size_py):.6f}")
total_train_end_time = time.time()
total_train_duration = total_train_end_time - total_train_start_time
print(f"Total training time: {total_train_duration:.4f} seconds")

# --- 7. Compute Test Cost ---
model_reg.eval() # Set model to evaluation mode
with torch.no_grad(): # No need to track gradients for evaluation
    X_test_dev_reg = X_test_reg.to(device)
    y_test_dev_reg = y_test_reg.to(device)

    test_outputs = model_reg(X_test_dev_reg)
    test_loss = criterion_reg(test_outputs, y_test_dev_reg)

    print(f"\nTest Cost (MSE): {test_loss.item():.6f}")


# --- Optional: Plot predictions vs actual for the test set (if 1D features for X) ---
if n_features_py == 1:
    model_reg.eval()
    with torch.no_grad():
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test_reg.cpu().numpy(), y_test_reg.cpu().numpy(), color='blue', label='Actual data', s=10)
        plt.scatter(X_test_reg.cpu().numpy(), test_outputs.cpu().numpy(), color='red', label='Predictions', s=10, marker='x')
        # To plot the learned line:
        # x_line = torch.linspace(X_data_reg[:,0].min(), X_data_reg[:,0].max(), 100).unsqueeze(1).to(device)
        # y_line = model_reg(x_line)
        # plt.plot(x_line.cpu().numpy(), y_line.cpu().numpy(), color='green', label='Learned Regression Line')

        plt.xlabel("Feature X1")
        plt.ylabel("Target Y")
        plt.title("Linear Regression: Actual vs. Predicted (Test Set)")
        plt.legend()
        plt.grid(True)
        plt.show()
elif n_features_py == 2:
    print("\nTo visualize 2D input regression, you would typically plot a 3D surface or contour plot of predictions,")
    print("or plot predicted vs actual y values in a scatter plot.")

    model_reg.eval()
    with torch.no_grad():
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_dev_reg.cpu().numpy(), test_outputs.cpu().numpy(), alpha=0.7, edgecolors='k', linewidths=0.5)
        min_val = min(y_test_dev_reg.min().item(), test_outputs.min().item())
        max_val = max(y_test_dev_reg.max().item(), test_outputs.max().item())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel("Actual Y values")
        plt.ylabel("Predicted Y values")
        plt.title("Actual vs. Predicted Y (Test Set)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()