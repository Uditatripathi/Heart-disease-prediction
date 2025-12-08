"""
Explainable AI Module
Provides SHAP values, LIME explanations, and feature importance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Suppress SHAP warnings
import logging
logging.getLogger('shap').setLevel(logging.ERROR)


class XAIExplainer:
    """Explainable AI class for model interpretability"""
    
    def __init__(self, model, X_train, X_test, feature_names, model_name='Model'):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.model_name = model_name
        self.shap_explainer = None
        self.lime_explainer = None
        
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            return feature_importance_df
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance = np.abs(self.model.coef_[0])
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            return feature_importance_df
        else:
            return None
    
    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """Plot feature importance"""
        importance_df = self.get_feature_importance()
        if importance_df is None:
            return None
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Feature Importance - {self.model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        return plt
    
    def initialize_shap_explainer(self, sample_size=100):
        """Initialize SHAP explainer"""
        try:
            # Sample data for faster computation
            X_sample = self.X_train[:sample_size] if len(self.X_train) > sample_size else self.X_train
            
            if hasattr(self.model, 'predict_proba'):
                # Tree-based models
                if hasattr(self.model, 'feature_importances_'):
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # For other models, use KernelExplainer
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, X_sample
                    )
            else:
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict, X_sample
                )
            return True
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            return False
    
    def get_shap_values(self, X_explain=None, max_evals=100):
        """Calculate SHAP values"""
        if self.shap_explainer is None:
            if not self.initialize_shap_explainer():
                return None
        
        if X_explain is None:
            X_explain = self.X_test[:10]  # Explain first 10 test samples
        
        try:
            if isinstance(self.shap_explainer, shap.TreeExplainer):
                shap_values = self.shap_explainer.shap_values(X_explain)
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, use positive class
            else:
                shap_values = self.shap_explainer.shap_values(
                    X_explain, nsamples=max_evals
                )
            
            return shap_values, X_explain
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None, None
    
    def plot_shap_summary(self, X_explain=None, max_evals=100, figsize=(10, 8)):
        """Plot SHAP summary plot"""
        shap_values, X_explain = self.get_shap_values(X_explain, max_evals)
        if shap_values is None:
            return None
        
        try:
            plt.figure(figsize=figsize)
            shap.summary_plot(shap_values, X_explain, 
                            feature_names=self.feature_names, 
                            show=False, plot_type='bar')
            plt.title(f'SHAP Feature Importance - {self.model_name}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            return plt
        except Exception as e:
            print(f"Error plotting SHAP summary: {e}")
            return None
    
    def plot_shap_waterfall(self, instance_idx=0, figsize=(10, 8)):
        """Plot SHAP waterfall plot for a single instance"""
        shap_values, X_explain = self.get_shap_values()
        if shap_values is None:
            return None
        
        try:
            if instance_idx >= len(X_explain):
                instance_idx = 0
            
            instance = X_explain[instance_idx]
            instance_shap = shap_values[instance_idx]
            
            # Create explanation object
            explanation = shap.Explanation(
                values=instance_shap,
                base_values=self.shap_explainer.expected_value if hasattr(self.shap_explainer, 'expected_value') else 0,
                data=instance,
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=figsize)
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}', 
                     fontsize=16, fontweight='bold')
            plt.tight_layout()
            return plt
        except Exception as e:
            print(f"Error plotting SHAP waterfall: {e}")
            return None
    
    def initialize_lime_explainer(self):
        """Initialize LIME explainer"""
        try:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=['No Heart Disease', 'Heart Disease'],
                mode='classification'
            )
            return True
        except Exception as e:
            print(f"Error initializing LIME explainer: {e}")
            return False
    
    def get_lime_explanation(self, instance, num_features=10):
        """Get LIME explanation for a single instance"""
        if self.lime_explainer is None:
            if not self.initialize_lime_explainer():
                return None
        
        try:
            explanation = self.lime_explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features
            )
            return explanation
        except Exception as e:
            print(f"Error getting LIME explanation: {e}")
            return None
    
    def get_lime_explanation_dict(self, instance, num_features=10):
        """Get LIME explanation as dictionary"""
        explanation = self.get_lime_explanation(instance, num_features)
        if explanation is None:
            return None
        
        try:
            exp_list = explanation.as_list()
            exp_dict = {
                'features': [item[0] for item in exp_list],
                'contributions': [item[1] for item in exp_list]
            }
            return exp_dict
        except Exception as e:
            print(f"Error converting LIME explanation: {e}")
            return None
    
    def plot_lime_explanation(self, instance, num_features=10, figsize=(10, 8)):
        """Plot LIME explanation"""
        explanation = self.get_lime_explanation(instance, num_features)
        if explanation is None:
            return None
        
        try:
            fig = explanation.as_pyplot_figure()
            fig.set_size_inches(figsize)
            plt.title(f'LIME Explanation - {self.model_name}', 
                     fontsize=16, fontweight='bold')
            plt.tight_layout()
            return plt
        except Exception as e:
            print(f"Error plotting LIME explanation: {e}")
            return None
    
    def get_global_explanation_summary(self):
        """Get comprehensive global explanation summary"""
        summary = {
            'feature_importance': self.get_feature_importance(),
            'model_name': self.model_name
        }
        return summary
    
    def explain_prediction(self, instance, model_type='best'):
        """Comprehensive explanation for a single prediction"""
        explanation = {
            'prediction': self.model.predict([instance])[0],
            'prediction_proba': self.model.predict_proba([instance])[0],
            'feature_importance': self.get_feature_importance(),
            'lime_explanation': self.get_lime_explanation_dict(instance)
        }
        
        # Add SHAP if available
        shap_values, _ = self.get_shap_values(np.array([instance]))
        if shap_values is not None:
            explanation['shap_values'] = shap_values[0].tolist()
        
        return explanation

