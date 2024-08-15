from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def find_best_ml_model(X, y):
    models = {
        'Linear Regression': LinearRegression(),
        'GLM': TweedieRegressor(power=0),
        'Decision Tree': DecisionTreeRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Support Vector Machine': SVR(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
    }

    if X.shape[1] == 1:
        models['Polynomial (degree 2)'] = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        models['Polynomial (degree 3)'] = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

    param_grids = {
        'Linear Regression': {},
        'GLM': {'power': [0, 1, 2], 'alpha': [0.01, 0.1, 1]},
        'Decision Tree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 10]},
        'K-Nearest Neighbors': {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']},
        'Support Vector Machine': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
        'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.01]},
    }

    if X.shape[1] == 1:
        param_grids['Polynomial (degree 2)'] = {}
        param_grids['Polynomial (degree 3)'] = {}

    best_model_info = {}

    # Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    residuals = None  # Store residuals for the best model

    for name, model in models.items():
        try:
            if name == 'GLM':
                # Special handling for GLM to avoid invalid y values
                y_min, y_max = np.min(y), np.max(y)
                if y_min <= 0:
                    y_train = y_train - y_min + 1e-4  # Ensure all y values are positive
                    y_test = y_test - y_min + 1e-4

            grid_search = GridSearchCV(model, param_grids[name], cv=6, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            best_model_info[name] = {
                'Best Parameters': grid_search.best_params_,
                'MSE': round(mse, 4),
                'R2': round(r2, 4)
            }

            if name == sorted(best_model_info.items(), key=lambda item: item[1]['R2'], reverse=True)[0][0]:
                residuals = y_test - y_pred  # Store residuals of the best model
        except Exception as e:
            print(f"Skipping model {name} due to error: {str(e)}")

    # Sort models by R2 score
    sorted_models = sorted(best_model_info.items(), key=lambda item: item[1]['R2'], reverse=True)

    # Determine complexity level based on R2 score of the best model
    max_r2 = sorted_models[0][1]['R2'] if sorted_models else 0
    complexity_level = int((1 - max_r2) * 10)

    return sorted_models, complexity_level, residuals