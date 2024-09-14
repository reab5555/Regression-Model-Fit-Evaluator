<img src="icon.jpeg" width="100" height="auto">

# ML Regression Model Fit Evaluator

The Advanced Model Fit Evaluator is a powerful, user-friendly tool designed to analyze and visualize relationships between variables in datasets. It employs various regression models and provides comprehensive visualizations to help users understand the complex relationships within their data.

[App](https://huggingface.co/spaces/reab5555/Regression-Model-Fit-Evaluator)

## Features

- **Multiple Regression Models**: Evaluates and compares various regression models from simplier model to more complex.
- **Model Comparison**: Visualizes the performance of top models using R-squared values, with absence of any regularization technics in order to determine the best fit.
- **Grid Search**: Utilizing grid-search for finding the best hyperparameters.
- **Data Scaling**: Implements feature scaling for better comparison and visualization.
- **Cross Validation** Built in cross-validation with k number of folds that can be set.
- **Features Importance**: Use permutations method to asses or rank features importance.
- **Residual Analysis**: Provides a detailed analysis of residuals, including distribution and normality tests.
- **Variable Relationship Visualization**: Creates scatter plots to show relationships between independent variables and the target variable.
- **Complexity Assessment**: Evaluates and reports on the complexity of relationships in the data.
- **User-Friendly Interface**: Built with Gradio for an intuitive, web-based user experience.

## List of Models:
  - Linear Regression
  - Polynomial Regression
  - Generalized Linear Model
  - Decision Tree Regressor
  - K-Nearest Neighbors Regressor
  - Support Vector Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
    
## How to Use

1. **Upload Data**: Start by uploading your CSV file containing the dataset.
2. **Select Variables**: Choose your target (dependent) variable and one or more independent variables.
3. **Analyze**: Click the "Analyze" button to process your data.
4. **Interpret Results**: Review the generated visualizations and text analysis:
   - Model Comparison Chart
   - Residuals Distribution Plot
   - Combined Scatter Plot of all variables
   - Detailed analysis of models and relationships

