"""
Machine Learning Engine for BDA Project
Includes Supervised Regression, Supervised Classification, 
Unsupervised K-Means Clustering, and Real-Time Prediction Logic.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

RISK_CLASSES = ['Low Risk', 'Moderate Risk', 'High Risk']

class BdaMLSuite:
    def __init__(self, ml_data, test_size=0.25, random_state=42):
        self.ml_data = ml_data
        self.test_size = test_size
        self.random_state = random_state
        
        X = ml_data['X_scaled']
        y_reg = ml_data['y_reg']
        y_clf = ml_data['y_clf']
        
        # Split data
        # Handle small dataset stratification safely
        try:
            self.X_train, self.X_test, self.y_reg_train, self.y_reg_test = train_test_split(
                X, y_reg, test_size=test_size, random_state=random_state
            )
            self.y_clf_train = y_clf.loc[self.X_train.index]
            self.y_clf_test = y_clf.loc[self.X_test.index]
        except Exception:
            self.X_train, self.X_test = X, X
            self.y_reg_train, self.y_reg_test = y_reg, y_reg
            self.y_clf_train, self.y_clf_test = y_clf, y_clf

        self.reg_models = {}
        self.clf_models = {}
        self.clustering_model = None
        self.pca_model = None

    def train_regression_models(self):
        """
        Trains and benchmarks Linear Regression, Decision Tree, and Random Forest.
        """
        candidates = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regressor': DecisionTreeRegressor(random_state=self.random_state, max_depth=4),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=50, random_state=self.random_state, max_depth=4)
        }
        
        results = {}
        for name, model in candidates.items():
            model.fit(self.X_train, self.y_reg_train)
            preds_train = model.predict(self.X_train)
            preds_test = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_reg_test, preds_test)
            mse = mean_squared_error(self.y_reg_test, preds_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_reg_test, preds_test)
            
            results[name] = {
                'model': model,
                'mae': round(mae, 3),
                'rmse': round(rmse, 3),
                'r2': round(r2, 3),
                'y_test': self.y_reg_test.values,
                'y_pred': preds_test
            }
            self.reg_models[name] = model
            
        return results

    def train_classification_models(self):
        """
        Trains and benchmarks Logistic Regression, Decision Tree, and Random Forest Classifier.
        """
        candidates = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=200),
            'Decision Tree Classifier': DecisionTreeClassifier(random_state=self.random_state, max_depth=4),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=50, random_state=self.random_state, max_depth=4)
        }
        
        results = {}
        for name, model in candidates.items():
            model.fit(self.X_train, self.y_clf_train)
            preds = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_clf_test, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                self.y_clf_test, preds, average='weighted', zero_division=0
            )
            cm = confusion_matrix(self.y_clf_test, preds)
            
            results[name] = {
                'model': model,
                'accuracy': round(acc * 100, 2),
                'precision': round(prec, 3),
                'recall': round(rec, 3),
                'f1_score': round(f1, 3),
                'confusion_matrix': cm,
                'y_test': self.y_clf_test.values,
                'y_pred': preds
            }
            self.clf_models[name] = model
            
        return results

    def train_clustering(self, k=3):
        """
        Performs unsupervised K-Means clustering and PCA 2D decomposition.
        """
        features_for_clustering = self.ml_data['X_scaled']
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(features_for_clustering)
        
        # PCA projection to 2 dimensions for visualization
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(features_for_clustering)
        
        self.clustering_model = kmeans
        self.pca_model = pca
        
        df_clustered = self.ml_data['encoded_df'].copy()
        df_clustered['cluster'] = clusters
        df_clustered['pca_x'] = pca_coords[:, 0]
        df_clustered['pca_y'] = pca_coords[:, 1]
        
        cluster_summary = df_clustered.groupby('cluster').agg({
            'suicides_per_100k': ['mean', 'min', 'max'],
            'suicides_no': ['sum', 'mean'],
            'population': 'mean'
        }).round(2)
        
        return {
            'kmeans': kmeans,
            'clusters': clusters,
            'pca_coords': pca_coords,
            'pca_variance_ratio': pca.explained_variance_ratio_,
            'df_clustered': df_clustered,
            'cluster_summary': cluster_summary
        }

    def predict_sample(self, country, sex, age, year, population, gdp_per_capita=5000, hdi=0.65):
        """
        Real-Time What-If Scenario Prediction.
        """
        feature_cols = self.ml_data['feature_cols']
        numeric_cols = self.ml_data['numeric_cols']
        scaler = self.ml_data['scaler']
        
        input_dict = {col: 0 for col in feature_cols}
        
        # Set categorical one-hot values
        country_col = f"country_{country}"
        sex_col = f"sex_{sex.lower()}"
        age_col = f"age_{age}"
        
        if country_col in input_dict:
            input_dict[country_col] = 1
        if sex_col in input_dict:
            input_dict[sex_col] = 1
        if age_col in input_dict:
            input_dict[age_col] = 1
            
        # Set numeric values
        input_dict['year'] = year
        input_dict['population'] = population
        if 'gdp_per_capita' in input_dict:
            input_dict['gdp_per_capita'] = gdp_per_capita
        if 'hdi' in input_dict:
            input_dict['hdi'] = hdi
            
        input_df = pd.DataFrame([input_dict])[feature_cols]
        
        # Scale numeric features
        input_scaled = input_df.copy()
        input_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Use best or Random Forest / Decision Tree model
        best_reg = self.reg_models.get('Random Forest Regressor') or self.reg_models.get('Linear Regression')
        best_clf = self.clf_models.get('Random Forest Classifier') or self.clf_models.get('Logistic Regression')
        
        pred_rate = float(best_reg.predict(input_scaled)[0])
        pred_rate = max(0.0, round(pred_rate, 2))
        pred_count = int(round((pred_rate / 100000.0) * population))
        
        pred_class_idx = int(best_clf.predict(input_scaled)[0])
        pred_class = RISK_CLASSES[min(pred_class_idx, len(RISK_CLASSES) - 1)]
        
        # Probabilities if available
        proba = None
        if hasattr(best_clf, 'predict_proba'):
            proba = best_clf.predict_proba(input_scaled)[0]
            
        return {
            'predicted_rate_per_100k': pred_rate,
            'predicted_suicides_no': pred_count,
            'predicted_risk_level': pred_class,
            'probabilities': proba,
            'model_used_reg': type(best_reg).__name__,
            'model_used_clf': type(best_clf).__name__
        }
