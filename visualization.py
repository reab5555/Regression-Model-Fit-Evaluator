import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, kstest
import tempfile
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from statsmodels.nonparametric.smoothers_lowess import lowess

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

def check_residual_distribution(residuals, best_model_name):
    plt.figure(figsize=(12, 8), dpi=400)
    sns.histplot(residuals, kde=True, stat="density", bins=50, color='black', label='Residual Distribution')
    mean, std = norm.fit(residuals)
    normal_best_fit_data = np.linspace(min(residuals), max(residuals), 1000)
    normal_pdf = norm.pdf(normal_best_fit_data, mean, std)
    p_value = kstest(residuals, 'norm', args=(mean, std)).pvalue
    p_value_text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    plt.plot(normal_best_fit_data, normal_pdf, color='red', lw=2, label=f'Normal Fit (p-value={p_value_text})')
    plt.title(f"Residuals Comparison with Normal Distribution\n(Best Model: {best_model_name})")
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
    
    # Combine all independent variables for the black line
    all_data = np.concatenate([scaled_df[var].values.reshape(-1, 1) for var in independent_vars])
    all_dependent = np.tile(scaled_df[dependent_var].values, len(independent_vars))
    
    # Plot black line for all variables combined
    lowess_result_all = lowess(all_dependent, all_data.flatten(), frac=0.6)
    plt.plot(lowess_result_all[:, 0], lowess_result_all[:, 1], color='black', linewidth=3, label='All Variables')
    
    for i, var in enumerate(independent_vars):
        plt.scatter(scaled_df[var], scaled_df[dependent_var],
                    color=colors[i], label=var, alpha=0.6)
        
        # Add colored LOESS line for each variable
        lowess_result = lowess(scaled_df[dependent_var], scaled_df[var], frac=0.6)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], color=colors[i], linewidth=1)
    
    plt.xlabel('Scaled Independent Variables')
    plt.ylabel(f'Scaled {dependent_var}')
    plt.title(f'All Variables vs {dependent_var} (Scaled) with LOESS')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, bbox_inches='tight')
    plt.close()
    
    return temp_file.name


def plot_feature_importance(X, y, best_model, model_name, best_params):
    plt.figure(figsize=(12, 8), dpi=400)
    
    if hasattr(best_model, 'coef_'):
        # For models with coefficients (e.g., Linear Regression, Lasso, Ridge)
        importance_values = best_model.coef_
        if importance_values.ndim == 1:
            importance_values = importance_values.reshape(1, -1)
        sorted_idx = np.argsort(np.abs(importance_values).flatten())[::-1]
        sorted_features = [X.columns[i] for i in sorted_idx]
        sorted_importance = importance_values.flatten()[sorted_idx]
        title_suffix = ""
    else:
        # Use permutation importance for other models, including SVR
        perm_importance = permutation_importance(best_model, X, y, n_repeats=35, random_state=42)
        importance_values = perm_importance.importances_mean
        sorted_idx = np.argsort(importance_values)[::-1]
        sorted_features = [X.columns[i] for i in sorted_idx]
        sorted_importance = importance_values[sorted_idx]
        title_suffix = f" (Kernel: {best_params.get('kernel', 'N/A')})" if model_name == "Support Vector Machine" else ""

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_importance)))
    bars = plt.barh(range(len(sorted_importance)), sorted_importance, color=colors, align='center')
    plt.yticks(range(len(sorted_importance)), sorted_features)
    plt.xlabel("Coefficient/Importance")
    plt.title(f"Feature Importance/Coefficients ({model_name}){title_suffix}")

    # Annotate each bar with the importance value
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width < 0:
            plt.text(width, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                     ha='right', va='center', color='black')
        else:
            plt.text(width, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                     ha='left', va='center', color='black')

    plt.axvline(x=0, color='k', linestyle='--')
    plt.ylabel("Features")
    plt.tight_layout()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name
