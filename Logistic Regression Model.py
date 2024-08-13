import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, brier_score_loss,
                             f1_score, matthews_corrcoef, cohen_kappa_score, log_loss,
                             balanced_accuracy_score, precision_recall_curve, auc, make_scorer)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import joblib
from scipy.stats import uniform, randint
import shap

# Load data from spreadsheet
data = pd.read_excel('filepath.xlsx')

# Assume the last column is the target and the rest are features
X = data.iloc[:, 2:-1].values
# X = data[['Age', 'BMI', 'Mammographic Breast Density', 'Number of Pathological Lymph Nodes', 'ALND']].values
y = data.iloc[:, -1].values
print(y)

# Normalize the features between 0 and 1
X_max = X.max(axis=0)
X = X / X_max

# Define k-fold cross-validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Define the parameter space for random search
param_dist = [
    {
        'penalty': ['l1', 'l2'],
        'C': uniform(0.1, 10),
        'solver': ['saga']
    },
    {
        'penalty': ['elasticnet'],
        'C': uniform(0.1, 10),
        'l1_ratio': uniform(0, 1),
        'solver': ['saga']
    }
]

# Dictionary to store hyperparameters and their performance across folds
hyperparameters_performance = {}

# Cross-validation for hyperparameter tuning
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f'Fold {fold+1}/{k}')
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize the Logistic Regression model
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=500,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring=make_scorer(roc_auc_score)
    )
    
    # Perform random search
    random_search.fit(X_train, y_train)
    
    print(f'Best parameters for fold {fold+1}: {random_search.best_params_}')
    
    # Store the performance of these hyperparameters
    param_key = tuple(random_search.best_params_.items())
    if param_key not in hyperparameters_performance:
        hyperparameters_performance[param_key] = []
    hyperparameters_performance[param_key].append(random_search.best_score_)

# Find the best overall hyperparameters
best_params = max(hyperparameters_performance, key=lambda x: np.mean(hyperparameters_performance[x]))
best_params = dict(best_params)

print("Best overall hyperparameters:")
print(best_params)

# Evaluate the model with best hyperparameters using cross-validation
best_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)

# Perform cross-validation and get predictions
y_pred_cv = cross_val_predict(best_model, X, y, cv=kf, method='predict_proba')[:, 1]

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
        # Resample with replacement
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]
        
        # Calculate metrics for this sample
        stats.append(calculate_metrics(y_true_resampled, y_pred_resampled))
    
    # Calculate confidence intervals
    ci_lower = np.percentile(stats, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(stats, (1 - alpha/2) * 100, axis=0)
    
    return ci_lower, ci_upper

# Determine performance metrics
metrics = calculate_metrics(y, y_pred_cv)
ci_lower, ci_upper = bootstrap_ci(y, y_pred_cv)

metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUC ROC', 'Brier Score',
                'F1 Score', 'Matthews Correlation Coefficient', 'Cohen\'s Kappa', 'Log Loss',
                'Balanced Accuracy', 'AUC Precision-Recall', 'Youden\'s J', 'Diagnostic Odds Ratio',
                'Lift', 'Gini Coefficient']

print("\nPerformance of best model (cross-validated) with 95% CI:")
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
plt.plot(fpr, tpr, label='AUC = {:.4f}'.format(metrics[5]), linewidth=3.0)  # AUC ROC is at index 5
plt.fill_between(fpr, tpr, alpha=0.2)  # Shade the area under the curve
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('Logistic Regression ROC Curve', fontsize = 22)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best', fontsize = 18)
plt.show()

# Calibration plot
plt.figure(figsize=(10, 8))
fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred_cv, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Logistic Regression Calibration plot (cross-validated)", fontsize = 20)
plt.legend()
plt.show()

# Train final model on all data
final_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
final_model.fit(X, y)

# Save the model and the maximum values used for normalization
joblib.dump({
    'model': final_model,
    'X_max': X_max
}, 'model_logistic_regression_tuned.pkl')

print("Model saved to model_logistic_regression_tuned.pkl")

# SHAP explanations
explainer = shap.Explainer(final_model, X)
shap_values = explainer(X)

# Plot SHAP summary
shap.summary_plot(shap_values, features=X, feature_names=data.columns[2:-1], plot_size = (6, 8), show = False, color_bar = False)
plt.yticks(fontsize=17,color='black')
plt.xticks(fontsize=16)
plt.xlabel('SHAP value (impact on model output)',fontsize=18)
cbar = plt.colorbar(aspect = 40)
cbar.ax.tick_params(labelsize = 14)
cbar.set_label('Relative Feature Value', fontsize = 16)
plt.savefig('LR SHAP Poster.png', dpi = 1000)
plt.show()