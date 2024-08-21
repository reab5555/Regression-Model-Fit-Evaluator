import gradio as gr
import pandas as pd
import numpy as np
from ml_fit import find_best_ml_model
from visualization import plot_model_bars, check_residual_distribution, plot_all_variables_scatter, plot_feature_importance

def load_file(file):
    df = pd.read_csv(file.name)
    numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    return gr.update(choices=numeric_columns), gr.update(choices=numeric_columns), df
    
def analyze_columns(target_var, independent_vars, df):
    if not target_var or not independent_vars:
        return None, None, None, None, "Please select a target variable and at least one independent variable."

    selected_vars = independent_vars + [target_var]
    selected_data = df[selected_vars].dropna()
    X = selected_data[independent_vars]
    y = selected_data[target_var]

    sorted_models, complexity_level, residuals, best_model_object = find_best_ml_model(X.values, y.values)
    model_plot = plot_model_bars(sorted_models)
    best_model_name = sorted_models[0][0]
    residual_plot, residual_p_value = check_residual_distribution(residuals, best_model_name, test_size=0.2)
    all_vars_scatter = plot_all_variables_scatter(selected_data, independent_vars, target_var)
    feature_importance_plot = plot_feature_importance(X, y, best_model_object, best_model_name)

    result_text = f"# Analysis Results\n\n"
    result_text += f"## Data Overview\n"
    result_text += f"- **Number of Samples**: {len(y)}\n"
    result_text += f"- **Number of Features**: {X.shape[1]}\n"
    result_text += f"- **Target Variable**: {target_var}\n"
    result_text += f"- **Independent Variables**: {', '.join(independent_vars)}\n"
    result_text += f"- **Relationship Complexity**: {complexity_level}/10\n\n"

    result_text += f"## Top 10 Models\n"
    for i, (name, info) in enumerate(sorted_models[:10], 1):
        result_text += f"{i}. **{name}**\n"
        result_text += f"   - R²: {info['R2']:.3f}\n"
        result_text += f"   - MSE: {info['MSE']:.3f}\n"
        if info['Best Parameters']:
            result_text += f"   - Best Parameters: {info['Best Parameters']}\n"
        result_text += "\n"

    result_text += f"## Residuals Analysis\n"
    result_text += f"- **Normality Test P-value**: {residual_p_value}\n"
    result_text += "  (A p-value < 0.05 suggests non-normal distribution)\n\n"

    result_text += f"## Conclusion\n"
    result_text += f"The most appropriate model is **{best_model_name}** with an R² of {sorted_models[0][1]['R2']:.3f}."

    return model_plot, residual_plot, all_vars_scatter, feature_importance_plot, result_text

with gr.Blocks() as demo:
    gr.Markdown("# Regression Model Fit Evaluator\n"
                "This tool analyzes various regression models and visualizes relationships between variables.")
    file_input = gr.File(label="Upload CSV File")
    with gr.Row():
        target_selector = gr.Dropdown(label="Select Target (Dependent) Variable", choices=[])
        independent_selector = gr.Dropdown(label="Select Independent Variables", choices=[], multiselect=True)
    analyze_button = gr.Button("Analyze")
    with gr.Row():
        model_plot_output = gr.Image(label="Model Comparison")
        feature_importance_output = gr.Image(label="Feature Importance")
    with gr.Row():
        all_vars_scatter_output = gr.Image(label="Scatter Plot (Scaled)")
        residual_plot_output = gr.Image(label="Residuals Distribution")

    output_text = gr.Markdown(label="Results")
    df_state = gr.State(None)

    file_input.upload(load_file, inputs=file_input, outputs=[target_selector, independent_selector, df_state])
    analyze_button.click(analyze_columns,
                         inputs=[target_selector, independent_selector, df_state],
                         outputs=[model_plot_output, residual_plot_output, all_vars_scatter_output, feature_importance_output, output_text])

if __name__ == "__main__":
    demo.launch()