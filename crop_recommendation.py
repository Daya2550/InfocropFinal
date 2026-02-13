"""
Crop Recommendation System Using Machine Learning
This module provides a complete crop recommendation system based on soil nutrients and weather conditions.

Author: ML Engineer
Date: 2026-01-14
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


class CropRecommendationSystem:
    """
    A comprehensive crop recommendation system using Random Forest Classifier.
    
    This system validates input parameters, trains an ML model, and recommends crops
    based on soil nutrients and weather conditions.
    """
    
    # Input validation ranges
    VALIDATION_RANGES = {
        'N': (0, 150),        # Nitrogen
        'P': (0, 150),        # Phosphorus
        'K': (0, 150),        # Potassium
        'temperature': (0, 60),   # Temperature in Celsius
        'humidity': (0, 100),     # Humidity in percentage
        'ph': (0, 14),            # pH value
        'rainfall': (0, 500)      # Rainfall in mm
    }
    
    def __init__(self, model_path='crop_model.pkl'):
        """
        Initialize the Crop Recommendation System.
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model = None
        self.model_path = model_path
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
    def load_and_preprocess_data(self, dataset_path):
        """
        Load and preprocess the dataset.
        
        Args:
            dataset_path (str): Path to the CSV dataset
            
        Returns:
            tuple: X (features), y (labels)
        """
        print("=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"[OK] Dataset loaded successfully")
        print(f"  - Total records: {len(df)}")
        print(f"  - Unique crops: {df['label'].nunique()}")
        print(f"  - Crop types: {sorted(df['label'].unique())}")
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print("[!] Warning: Missing values detected. Handling...")
            df = df.dropna()
            print(f"  - Records after cleaning: {len(df)}")
        else:
            print("[OK] No missing values detected")
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df['label']
        
        print(f"\n[OK] Data preprocessing complete")
        print(f"  - Features shape: {X.shape}")
        print(f"  - Target shape: {y.shape}")
        print("=" * 60)
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Random Forest Classifier.
        
        Args:
            X: Feature data
            y: Target labels
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            float: Model accuracy on test set
        """
        print("\nTRAINING MODEL")
        print("=" * 60)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"[OK] Data split complete")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Testing samples: {len(X_test)}")
        
        # Initialize and train Random Forest Classifier
        print("\n[*] Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.model.fit(X_train, y_train)
        print("[OK] Model training complete")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("\n" + "=" * 60)
        print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
        print("=" * 60)
        
        # Display detailed classification report
        print("\nDETAILED PERFORMANCE METRICS:")
        print("-" * 60)
        print(classification_report(y_test, y_pred))
        
        # Save the model
        self.save_model()
        
        return accuracy
    
    def save_model(self):
        """Save the trained model to disk."""
        try:
            joblib.dump(self.model, self.model_path)
            print(f"\n[OK] Model saved successfully to '{self.model_path}'")
        except Exception as e:
            print(f"\n[ERROR] Error saving model: {e}")
    
    def load_model(self):
        """Load a trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"[OK] Model loaded successfully from '{self.model_path}'")
                return True
            else:
                print(f"[!] Model file '{self.model_path}' not found")
                return False
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            return False
    
    def validate_input(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Validate input parameters against defined ranges.
        
        Args:
            N, P, K, temperature, humidity, ph, rainfall: Input parameters
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Parameter names mapping for user-friendly messages
        param_names = {
            'N': 'Nitrogen',
            'P': 'Phosphorus',
            'K': 'Potassium',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'ph': 'pH',
            'rainfall': 'Rainfall'
        }
        
        # Input values
        inputs = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        
        # Validate each parameter
        for param, value in inputs.items():
            # Check if value is numeric
            if not isinstance(value, (int, float)):
                return False, f"Invalid input detected for {param_names[param]}. Please enter valid numeric data."
            
            # Check if value is within valid range
            min_val, max_val = self.VALIDATION_RANGES[param]
            if not min_val <= value <= max_val:
                return False, f"Invalid input detected for {param_names[param]}. Please enter valid data (Range: {min_val}-{max_val})."
        
        return True, "Valid"
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Recommend the best crop based on input parameters.
        
        Args:
            N (float): Nitrogen content (0-150)
            P (float): Phosphorus content (0-150)
            K (float): Potassium content (0-150)
            temperature (float): Temperature in Celsius (0-60)
            humidity (float): Humidity percentage (0-100)
            ph (float): pH value (0-14)
            rainfall (float): Rainfall in mm (0-500)
            
        Returns:
            str: Recommended crop name or error message
        """
        # Validate input
        is_valid, message = self.validate_input(N, P, K, temperature, humidity, ph, rainfall)
        
        if not is_valid:
            return message
        
        # Check if model is loaded
        if self.model is None:
            return "Error: Model not trained or loaded. Please train the model first."
        
        # Prepare input data
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Make prediction
        try:
            prediction = self.model.predict(input_data)
            return f"Recommended Crop: {prediction[0]}"
        except Exception as e:
            return f"Error making prediction: {e}"


def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Standalone function to recommend crops using a pre-trained model.
    
    This is a convenience function that wraps the CropRecommendationSystem class.
    
    Args:
        N (float): Nitrogen content (0-150)
        P (float): Phosphorus content (0-150)
        K (float): Potassium content (0-150)
        temperature (float): Temperature in Celsius (0-60)
        humidity (float): Humidity percentage (0-100)
        ph (float): pH value (0-14)
        rainfall (float): Rainfall in mm (0-500)
        
    Returns:
        str: Recommended crop name or error message
        
    Example:
        >>> recommend_crop(90, 42, 43, 20.8, 82, 6.5, 202.9)
        'Recommended Crop: rice'
    """
    # Initialize system and load model
    system = CropRecommendationSystem()
    
    if not system.load_model():
        return "Error: Model not found. Please train the model first by running main()."
    
    # Get recommendation
    return system.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)


def main():
    """
    Main function to train the model and run test cases.
    """
    print("\n" + "=" * 60)
    print("CROP RECOMMENDATION SYSTEM")
    print("Machine Learning Based Agricultural Solution")
    print("=" * 60 + "\n")
    
    # Initialize the system
    system = CropRecommendationSystem()
    
    # Load and preprocess data
    dataset_path = 'Crop_recommendation.csv'
    X, y = system.load_and_preprocess_data(dataset_path)
    
    # Train the model
    accuracy = system.train_model(X, y)
    
    # Test cases
    print("\n" + "=" * 60)
    print("RUNNING TEST CASES")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Valid Input - Rice',
            'params': (90, 42, 43, 20.8, 82, 6.5, 202.9),
            'expected': 'Valid'
        },
        {
            'name': 'Invalid Input - Temperature Out of Range',
            'params': (90, 42, 43, 120, 82, 6.5, 202.9),
            'expected': 'Invalid'
        },
        {
            'name': 'Invalid Input - Humidity Out of Range',
            'params': (90, 42, 43, 25, 150, 6.5, 200),
            'expected': 'Invalid'
        },
        {
            'name': 'Valid Input - Maize',
            'params': (80, 43, 16, 23.5, 71.5, 6.6, 66.7),
            'expected': 'Valid'
        },
        {
            'name': 'Invalid Input - Negative Nitrogen',
            'params': (-10, 42, 43, 25, 82, 6.5, 200),
            'expected': 'Invalid'
        },
        {
            'name': 'Invalid Input - pH Out of Range',
            'params': (90, 42, 43, 25, 82, 15, 200),
            'expected': 'Invalid'
        },
        {
            'name': 'Valid Input - Chickpea',
            'params': (40, 72, 77, 17, 17, 7.5, 88.5),
            'expected': 'Valid'
        },
        {
            'name': 'Edge Case - Minimum Values',
            'params': (0, 0, 0, 0, 0, 0, 0),
            'expected': 'Valid'
        },
        {
            'name': 'Edge Case - Maximum Values',
            'params': (150, 150, 150, 60, 100, 14, 500),
            'expected': 'Valid'
        }
    ]
    
    print("\nTest Case Results:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Input: N={test_case['params'][0]}, P={test_case['params'][1]}, "
              f"K={test_case['params'][2]}, Temp={test_case['params'][3]}, "
              f"Humidity={test_case['params'][4]}, pH={test_case['params'][5]}, "
              f"Rainfall={test_case['params'][6]}")
        
        result = system.recommend_crop(*test_case['params'])
        print(f"   Output: {result}")
        
        # Verify expected behavior
        if test_case['expected'] == 'Valid':
            if result.startswith('Recommended Crop'):
                print("   Status: [PASS]")
            else:
                print("   Status: [FAIL] (Expected valid recommendation)")
        else:  # Invalid expected
            if result.startswith('Invalid input'):
                print("   Status: [PASS]")
            else:
                print("   Status: [FAIL] (Expected validation error)")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    
    # Interactive demo
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO")
    print("=" * 60)
    print("\nYou can now use the recommend_crop() function to get recommendations.")
    print("\nExample usage:")
    print(">>> recommend_crop(90, 42, 43, 20.8, 82, 6.5, 202.9)")
    print("'Recommended Crop: rice'")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run the main function
    main()
