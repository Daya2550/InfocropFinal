"""
Ensemble Model Training for Crop Recommendation System
Trains multiple ML models and creates an ensemble for more accurate predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsembleCropRecommendation:
    """
    Ensemble learning system using multiple ML models for crop recommendation.
    """
    
    def __init__(self):
        """Initialize the ensemble system with multiple models."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=3,
                random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            )
        }
        
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
    def load_data(self, dataset_path='Crop_recommendation.csv'):
        """Load and preprocess the dataset."""
        print("=" * 80)
        print("ENSEMBLE MODEL TRAINING - MULTI-MODEL APPROACH")
        print("=" * 80)
        print("\n[*] Loading dataset...")
        
        df = pd.read_csv(dataset_path)
        print(f"[OK] Dataset loaded: {len(df)} records, {df['label'].nunique()} crops")
        
        # Remove any missing values
        df = df.dropna()
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df['label']
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2):
        """Train all models and evaluate their performance."""
        print("\n" + "=" * 80)
        print("TRAINING MULTIPLE MODELS")
        print("=" * 80)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n[OK] Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n[*] Training {model_name.replace('_', ' ').title()}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_score': cv_mean
            }
            
            print(f"[OK] {model_name.replace('_', ' ').title()}")
            print(f"    - Accuracy: {accuracy * 100:.2f}%")
            print(f"    - Cross-validation: {cv_mean * 100:.2f}%")
        
        # Ensemble prediction test
        print("\n" + "=" * 80)
        print("ENSEMBLE PERFORMANCE")
        print("=" * 80)
        
        ensemble_predictions = self.predict_ensemble(X_test, return_details=False)
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        
        print(f"\n[OK] Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")
        print(f"[OK] Improvement: +{(ensemble_accuracy - max([r['accuracy'] for r in results.values()])) * 100:.2f}%")
        
        return results, X_test, y_test
    
    def predict_ensemble(self, X, return_details=True):
        """
        Make predictions using all models and return ensemble result.
        
        Args:
            X: Input features
            return_details: If True, return detailed predictions from all models
            
        Returns:
            If return_details=True: dict with predictions from all models
            If return_details=False: array of final predictions
        """
        predictions = {}
        
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        
        if not return_details:
            # Voting mechanism - return majority vote
            pred_array = np.array([predictions[m] for m in predictions.keys()])
            final_predictions = []
            
            for i in range(pred_array.shape[1]):
                votes = pred_array[:, i]
                # Get most common prediction
                unique, counts = np.unique(votes, return_counts=True)
                final_predictions.append(unique[np.argmax(counts)])
            
            return np.array(final_predictions)
        
        return predictions
    
    def save_models(self):
        """Save all trained models."""
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            filename = f'{model_name}_model.pkl'
            joblib.dump(model, filename)
            print(f"[OK] Saved {model_name} to {filename}")
        
        # Save ensemble info
        ensemble_info = {
            'models': list(self.models.keys()),
            'feature_columns': self.feature_columns
        }
        joblib.dump(ensemble_info, 'ensemble_info.pkl')
        print(f"[OK] Saved ensemble info to ensemble_info.pkl")
        
    def load_models(self):
        """Load all trained models."""
        try:
            for model_name in self.models.keys():
                filename = f'{model_name}_model.pkl'
                self.models[model_name] = joblib.load(filename)
            print("[OK] All models loaded successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            return False

def main():
    """Main function to train ensemble models."""
    # Initialize ensemble system
    ensemble = EnsembleCropRecommendation()
    
    # Load data
    X, y = ensemble.load_data()
    
    # Train models
    results, X_test, y_test = ensemble.train_models(X, y)
    
    # Save models
    ensemble.save_models()
    
    # Detailed performance report
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Test Accuracy: {result['accuracy'] * 100:.2f}%")
        print(f"  CV Score: {result['cv_score'] * 100:.2f}%")
    
    # Test ensemble prediction
    print("\n" + "=" * 80)
    print("ENSEMBLE PREDICTION TEST")
    print("=" * 80)
    
    # Test with a sample (rice example)
    test_input = np.array([[90, 42, 43, 20.8, 82, 6.5, 202.9]])
    
    print("\nTest Input: N=90, P=42, K=43, Temp=20.8, Humidity=82, pH=6.5, Rainfall=202.9")
    print("\nIndividual Model Predictions:")
    
    for model_name, model in ensemble.models.items():
        pred = model.predict(test_input)
        proba = model.predict_proba(test_input)
        confidence = np.max(proba) * 100
        print(f"  {model_name.replace('_', ' ').title()}: {pred[0]} ({confidence:.1f}% confidence)")
    
    # Ensemble prediction
    predictions = ensemble.predict_ensemble(test_input, return_details=False)
    print(f"\nEnsemble Prediction: {predictions[0]}")
    
    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
