import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, brier_score_loss,
                             f1_score, matthews_corrcoef, cohen_kappa_score, log_loss,
                             balanced_accuracy_score, precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Custom logistic regression model
class CustomLogisticRegression:
    def __init__(self):
        self.coefficients = np.array([-0.3510, 0.6127, 0.1224, 0.2121])
    
    def fit(self, X, y):
        # This method doesn't actually fit the model, as coefficients are pre-defined
        pass
    
    def predict_proba(self, X):
        # Add a column of 1s for the intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        z = np.dot(X_with_intercept, self.coefficients)
        probabilities = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Load data from spreadsheet
data = pd.read_excel('filepath.xlsx')

# Extract the required features and target
X = data[['RT Boost (0 = No Boost, 1 = Boost)', 'Reduced Motivation P1M - D1', 'IL_17 P1M - D1']].values
y = data.iloc[:, -1].values

# Define k-fold cross-validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Perform manual cross-validation
y_pred_cv = np.zeros_like(y, dtype=float)
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = CustomLogisticRegression()
    model.fit(X_train, y_train)
    y_pred_cv[test_index] = model.predict_proba(X_test)[:, 1]

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
    
    f1 = f1_score(y_true, y_pred_binary)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    kappa = cohen_kappa_score(y_true, y_pred_binary)
    logloss = log_loss(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_binary)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    
    youdens_j = sensitivity + specificity - 1
    diagnostic_odds_ratio = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.inf
    lift = (tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn)) if (tp + fp) > 0 else np.inf
    gini = 2 * auc_roc - 1

    return (sensitivity, specificity, ppv, npv, accuracy, auc_roc, brier,
            f1, mcc, kappa, logloss, balanced_acc, auc_pr, youdens_j,
            diagnostic_odds_ratio, lift, gini)

# Function to calculate confidence intervals using bootstrapping
def bootstrap_ci(y_true, y_pred, n_iterations=1000, alpha=0.05):
    stats = []
    for _ in range(n_iterations):
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]
        stats.append(calculate_metrics(y_true_resampled, y_pred_resampled))
    
    ci_lower = np.percentile(stats, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(stats, (1 - alpha/2) * 100, axis=0)
    
    return ci_lower, ci_upper

# Calculate performance metrics
metrics = calculate_metrics(y, y_pred_cv)
ci_lower, ci_upper = bootstrap_ci(y, y_pred_cv)

metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUC ROC', 'Brier Score',
                'F1 Score', 'Matthews Correlation Coefficient', 'Cohen\'s Kappa', 'Log Loss',
                'Balanced Accuracy', 'AUC Precision-Recall', 'Youden\'s J', 'Diagnostic Odds Ratio',
                'Lift', 'Gini Coefficient']

print("\nPerformance of logistic regression model (cross-validated) with 95% CI:")
for name, value, lower, upper in zip(metric_names, metrics, ci_lower, ci_upper):
    print(f'{name}: {value:.4f} (95% CI: {lower:.4f} - {upper:.4f})')

# Print full confusion matrix
cm = confusion_matrix(y, (y_pred_cv >= 0.5).astype(int))
print("\nConfusion Matrix:")
print(cm)

# Plot the AUC curve
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--')
fpr, tpr, _ = roc_curve(y, y_pred_cv)
plt.plot(fpr, tpr, label=f'AUC = {metrics[5]:.4f}', linewidth=3.0, color = 'purple')
plt.fill_between(fpr, tpr, color = "purple", alpha=0.2)  # Shade the area under the curve
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Multivariable Logistic Regression ROC Curve', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc='best', fontsize=16)
plt.show()

# Calibration plot
plt.figure(figsize=(10, 8))
fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred_cv, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Logistic Regression Calibration plot (cross-validated)", fontsize=20)
plt.legend()
plt.show()

