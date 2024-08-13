import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib
from scipy.stats import randint, uniform
from scipy import stats
import shap
from sklearn.utils import resample

# Load data from spreadsheet
data = pd.read_excel('filepath.xlsx')

# Assume the last column is the target and the rest are features
X = data.iloc[:, 2:-1]
y = data.iloc[:, -1]

# Normalize the features between 0 and 1
X_max = X.max()
X = X / X_max

# Define k-fold cross-validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Define the hyperparameter space
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=500,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Perform random search
random_search.fit(X, y)

# Get the best model
best_rf = random_search.best_estimator_

print("Best hyperparameters:", random_search.best_params_)

# Use cross_val_predict to get predictions
y_pred = cross_val_predict(best_rf, X, y, cv=k, method='predict')
y_pred_prob = cross_val_predict(best_rf, X, y, cv=k, method='predict_proba')[:, 1]

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    brier = brier_score_loss(y_true, y_pred_prob)
    return sensitivity, specificity, ppv, npv, accuracy, auc, brier

# Function to calculate bootstrapped confidence intervals
def bootstrap_ci(y_true, y_pred, y_pred_prob, n_iterations=1000, alpha=0.05):
    stats = []
    for _ in range(n_iterations):
        # Resample with replacement
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_resampled = y_true.iloc[indices]
        y_pred_resampled = y_pred[indices]
        y_pred_prob_resampled = y_pred_prob[indices]
        
        # Calculate metrics for this sample
        stats.append(calculate_metrics(y_true_resampled, y_pred_resampled, y_pred_prob_resampled))
    
    # Calculate confidence intervals
    ci_lower = np.percentile(stats, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(stats, (1 - alpha/2) * 100, axis=0)
    
    return ci_lower, ci_upper

# Calculate metrics
metrics = calculate_metrics(y, y_pred, y_pred_prob)
ci_lower, ci_upper = bootstrap_ci(y, y_pred, y_pred_prob)

# Print metrics with confidence intervals
metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUC', 'Brier Score']
for i, metric_name in enumerate(metric_names):
    print(f'{metric_name}: {metrics[i]:.4f} (95% CI: {ci_lower[i]:.4f} - {ci_upper[i]:.4f})')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y, y_pred_prob)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'AUC = {metrics[5]:.4f})', linewidth=3.0, color = "g")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.fill_between(fpr, tpr, color = "g", alpha=0.2)  # Shade the area under the curve
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.title('Random Forest ROC Curve', fontsize = 20)
plt.legend(loc='best', fontsize = 16)
plt.show()

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred_prob, n_bins=10)
plt.figure()
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Calibration curve')
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Plot")
plt.legend()
plt.show()

# Save the final model and feature names
joblib.dump({
    'model': best_rf,
    'feature_names': X.columns.tolist(),
    'X_max': X_max
}, 'random_forest_model_tuned.pkl')

print("Tuned Random Forest model and feature names saved to random_forest_model_tuned.pkl")