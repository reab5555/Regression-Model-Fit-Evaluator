import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, kstest
import tempfile
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.inspection import permutation_importance

def plot_model_bars(sorted_models):
    top_models = sorted_models[:5]
    names = [name for name, _ in top_models]
    r_squared_values = [info['R2'] for _, info in top_models]
    plt.figure(figsize=(12, 8), dpi=400)
    sns.barplot(x=r_squared_values, y=names, hue=names, palette="Set1", legend=False)
    plt.axvline(0, color='black', linewidth=1.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    plt.gca().set_axisbelow(True)
    for i, v in enumerate(r_squared_values):
        plt.text(v, i, f"{v:.3f}", color='black', ha='left', va='center')
    plt.xlabel('R-squared')
    plt.title('Top 5 Best Models')
    plt.tight_layout()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name

def check_residual_distribution(residuals, best_model_name, test_size=0.2):
    plt.figure(figsize=(12, 8), dpi=400)
    sns.histplot(residuals, kde=True, stat="density", bins=50, color='black', label='Residual Distribution')
    mean, std = norm.fit(residuals)
    normal_best_fit_data = np.linspace(min(residuals), max(residuals), 1000)
    normal_pdf = norm.pdf(normal_best_fit_data, mean, std)
    p_value = kstest(residuals, 'norm', args=(mean, std)).pvalue
    p_value_text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    plt.plot(normal_best_fit_data, normal_pdf, color='red', lw=2, label=f'Normal Fit (p-value={p_value_text})')
    plt.title(f"Residuals Comparison with Normal Distribution\n(Best Model: {best_model_name}, Test Split: {test_size*100}%)")
    plt.legend()
    temp_file_residuals = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file_residuals.name)
    plt.close()
    return temp_file_residuals.name, p_value_text

def plot_all_variables_scatter(data, independent_vars, dependent_var):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    plt.figure(figsize=(12, 8), dpi=400)
    colors = sns.color_palette("husl", n_colors=len(independent_vars))
    for i, var in enumerate(independent_vars):
        plt.scatter(scaled_df[var], scaled_df[dependent_var],
                    color=colors[i], label=var, alpha=0.6)

    plt.xlabel('Scaled Independent Variables')
    plt.ylabel(f'Scaled {dependent_var}')
    plt.title(f'All Variables vs {dependent_var} (Scaled)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, bbox_inches='tight')
    plt.close()
    return temp_file.name

def plot_feature_importance(X, y, best_model, model_name):
    perm_importance = permutation_importance(best_model, X, y, n_repeats=15, random_state=42)
    importance_values = np.abs(perm_importance.importances_mean)
    sorted_idx = importance_values.argsort()[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importance = importance_values[sorted_idx]
    plt.figure(figsize=(12, 8), dpi=400)
    sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis")
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name

    