"""
Multi-Model Training and Comparison Module
Trains and compares: Logistic Regression, Random Forest, XGBoost, SVM, Neural Network
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix,
                            classification_report)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and compare multiple ML models"""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_logistic_regression(self, **kwargs):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        model = LogisticRegression(random_state=self.random_state, max_iter=1000, **kwargs)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = model
        return model
    
    def train_random_forest(self, n_estimators=100, **kwargs):
        """Train Random Forest model"""
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        return model
    
    def train_xgboost(self, **kwargs):
        """Train XGBoost model"""
        print("Training XGBoost...")
        model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            **kwargs
        )
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        return model
    
    def train_svm(self, **kwargs):
        """Train Support Vector Machine model"""
        print("Training SVM...")
        model = SVC(probability=True, random_state=self.random_state, **kwargs)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = model
        return model
    
    def train_neural_network(self, hidden_layers=(100, 50), **kwargs):
        """Train Neural Network model"""
        print("Training Neural Network...")
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            random_state=self.random_state,
            max_iter=1000,
            **kwargs
        )
        model.fit(self.X_train_scaled, self.y_train)
        self.models['Neural Network'] = model
        return model
    
    def train_all_models(self):
        """Train all models"""
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_svm()
        self.train_neural_network()
        print("All models trained successfully!")
    
    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        # Determine if model needs scaled data
        if model_name in ['Logistic Regression', 'SVM', 'Neural Network']:
            X_test = self.X_test_scaled
            X_train = self.X_train_scaled
        else:
            X_test = self.X_test
            X_train = self.X_train
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        for model_name, model in self.models.items():
            self.results[model_name] = self.evaluate_model(model, model_name)
        return self.results
    
    def get_best_model(self, metric='f1_score'):
        """Get the best model based on specified metric"""
        if not self.results:
            self.evaluate_all_models()
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x][metric])
        best_model = self.models[best_model_name]
        best_results = self.results[best_model_name]
        
        return best_model_name, best_model, best_results
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        for model_name, results in self.results.items():
            plt.plot(results['fpr'], results['tpr'], 
                    label=f"{model_name} (AUC = {results['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return plt
    
    def get_comparison_dataframe(self):
        """Get comparison dataframe of all models"""
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        return df
    
    def save_model(self, model, model_name, filepath):
        """Save trained model"""
        joblib.dump(model, filepath)
        print(f"{model_name} saved to {filepath}")
    
    def save_scaler(self, filepath):
        """Save scaler"""
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        return joblib.load(filepath)
    
    def predict(self, model_name, X, scaled=True):
        """Make prediction using specified model"""
        model = self.models[model_name]
        
        if scaled and model_name in ['Logistic Regression', 'SVM', 'Neural Network']:
            X = self.scaler.transform(X)
        
        prediction = model.predict(X)
        prediction_proba = model.predict_proba(X)
        
        return prediction, prediction_proba

