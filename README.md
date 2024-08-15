<img src="appendix/icon.webp" width="100" height="auto">

# Advanced Model Fit Evaluator

## Overview

The Advanced Model Fit Evaluator is a powerful, user-friendly tool designed to analyze and visualize relationships between variables in datasets. It employs various regression models and provides comprehensive visualizations to help users understand the complex relationships within their data.

## Features

- **Multiple Regression Models**: Evaluates and compares various regression models including Linear Regression, Polynomial Regression, Decision Trees, Random Forests, and more.
- **Model Comparison**: Visualizes the performance of top models using R-squared values.
- **Residual Analysis**: Provides a detailed analysis of residuals, including distribution and normality tests.
- **Variable Relationship Visualization**: Creates scatter plots to show relationships between independent variables and the target variable.
- **Data Scaling**: Implements feature scaling for better comparison and visualization.
- **Complexity Assessment**: Evaluates and reports on the complexity of relationships in the data.
- **User-Friendly Interface**: Built with Gradio for an intuitive, web-based user experience.

## How to Use

1. **Upload Data**: Start by uploading your CSV file containing the dataset.
2. **Select Variables**: Choose your target (dependent) variable and one or more independent variables.
3. **Analyze**: Click the "Analyze" button to process your data.
4. **Interpret Results**: Review the generated visualizations and text analysis:
   - Model Comparison Chart
   - Residuals Distribution Plot
   - Combined Scatter Plot of all variables
   - Detailed text analysis of models and relationships

## Installation

```bash
git clone https://github.com/reab5555/advanced-model-fit-evaluator.git
cd advanced-model-fit-evaluator
pip install -r requirements.txt
