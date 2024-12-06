import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import optuna
from scipy import stats

# Define the neural network model with variable hidden layers
class MultiLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(MultiLayerNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Load data from spreadsheet
data = pd.read_excel('data.xlsx')

# Assume the last column is the target and the rest are features
X = data.iloc[:, 2:-1].values
y = data.iloc[:, -1].values

# Normalize the features between 0 and 1
X_max = X.max(axis=0)
X = X / X_max

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define k-fold cross-validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Early stopping patience threshold
patience = 10

# Define the Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
    hidden_size = trial.suggest_int('hidden_size', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 500, 2000)

    # Initialize lists to store metrics
    auc_list = []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = MultiLayerNN(input_size=X.shape[1], hidden_size=hidden_size, output_size=1, num_hidden_layers=num_hidden_layers)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Early stopping variables
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Training
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test).item()

            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} with best validation loss: {best_loss:.4f}")
                break

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            y_pred_prob = test_outputs.numpy()
            y_test_numpy = y_test.numpy()
            auc = roc_auc_score(y_test_numpy, y_pred_prob)
            auc_list.append(auc)
    
    return np.mean(auc_list)

# Run hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print('Best hyperparameters:', best_params)

# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc_roc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    
    return sensitivity, specificity, ppv, npv, accuracy, auc_roc, brier

# Train the final model with the best hyperparameters
hidden_size = best_params['hidden_size']
learning_rate = best_params['learning_rate']
weight_decay = best_params['weight_decay']
num_epochs = best_params['num_epochs']
num_hidden_layers = best_params['num_hidden_layers']

# Initialize lists to store metrics
metrics_list = []
all_y_test = []
all_y_pred = []

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f'Fold {fold+1}/{k}')
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = MultiLayerNN(input_size=X.shape[1], hidden_size=hidden_size, output_size=1, num_hidden_layers=num_hidden_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} with best validation loss: {best_loss:.4f}")
            break

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        y_pred_prob = test_outputs.numpy()
        y_test_numpy = y_test.numpy()

        metrics = calculate_metrics(y_test_numpy, y_pred_prob)
        metrics_list.append(metrics)
        
        all_y_test.extend(y_test_numpy)
        all_y_pred.extend(y_pred_prob)

# Calculate and print average metrics
metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUC', 'Brier Score']
for i, name in enumerate(metric_names):
    mean = np.mean([m[i] for m in metrics_list])
    print(f'{name}: {mean:.4f}')

# Save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'X_max': X_max,
    'best_params': best_params
}, 'tuned_nn_model.pth')

print("Model saved to tuned_nn_model.pth")





