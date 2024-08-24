from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import numpy as np

def find_best_ml_model(X, y, cv_folds=6):

    models = {
        'Linear Regression': LinearRegression(),
        'GLM': TweedieRegressor(power=0),
        'Decision Tree': DecisionTreeRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Support Vector Machine': SVR(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    if X.shape[1] == 1:
        models['Polynomial (degree 2)'] = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        models['Polynomial (degree 3)'] = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

    param_grids = {
        'Linear Regression': {},
        'GLM': {
            'power': [0, 1, 2, 3],
            'link': ['auto']
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 5],
            'max_features': ['sqrt', 'log2', None]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 10, 15], 
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],
            'leaf_size': [15, 30, 50] 
        },
        'Support Vector Machine': {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 125],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 5],
            'bootstrap': [True, False]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 1.0],  
            'max_features': ['sqrt', 'log2', None] 
        }
    }


    if X.shape[1] == 1:
        param_grids['Polynomial (degree 2)'] = {}
        param_grids['Polynomial (degree 3)'] = {}
        
    best_model_info = {}
    best_model_object = None
    best_r2 = -float('inf')
    overall_best_residuals = None

    # Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model in models.items():
        try:
            if name == 'GLM':
                # Special handling for GLM to avoid invalid y values
                y_min, y_max = np.min(y), np.max(y)
                if y_min <= 0:
                    y_adjusted = y - y_min + 1e-4  # Ensure all y values are positive
                else:
                    y_adjusted = y

                grid_search = GridSearchCV(model, param_grids[name], cv=cv_folds, scoring='r2', n_jobs=-1)
                grid_search.fit(X_scaled, y_adjusted)
            else:
                grid_search = GridSearchCV(model, param_grids[name], cv=cv_folds, scoring='r2', n_jobs=-1)
                grid_search.fit(X_scaled, y)

            best_model = grid_search.best_estimator_
            
            # Perform 7-fold cross-validation predictions
            cross_val_preds = cross_val_predict(best_model, X_scaled, y, cv=cv_folds)
            
            rmse = np.sqrt(mean_squared_error(y, cross_val_preds))
            r2 = np.mean(cross_val_score(best_model, X_scaled, y, cv=cv_folds, scoring='r2'))  # Use mean of cross-validation R2 scores

            best_model_info[name] = {
                'Best Parameters': grid_search.best_params_,
                'RMSE': round(rmse, 4),
                'R2': round(r2, 4),
                'model': best_model,
                'Cross-Val Predictions': cross_val_preds
            }

            if r2 > best_r2:
                best_r2 = r2
                best_model_object = best_model
                overall_best_residuals = y - cross_val_preds

        except Exception as e:
            print(f"Skipping model {name} due to error: {str(e)}")

    # Sort models by R2 score
    sorted_models = sorted(best_model_info.items(), key=lambda item: item[1]['R2'], reverse=True)

    # Determine complexity level based on R2 score of the best model
    max_r2 = sorted_models[0][1]['R2'] if sorted_models else 0
    complexity_level = int((1 - max_r2) * 10)

    return sorted_models, complexity_level, overall_best_residuals, best_model_object