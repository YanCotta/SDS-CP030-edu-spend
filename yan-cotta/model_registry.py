"""
MLflow Model Registry Setup for EduSpend TCA Prediction
Registers the best performing model for production deployment
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class TCAPredictor:
    """
    Production-ready TCA (Total Cost of Attendance) Predictor
    Integrated with MLflow Model Registry for deployment
    """
    
    def __init__(self, model_name="EduSpend-TCA-Predictor"):
        self.model_name = model_name
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.top_cities = None
        self.model_version = None
        
    def load_data(self, data_path='data/International_Education_Costs.csv'):
        """Load and prepare the dataset"""
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Dataset loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Calculate TCA if not present
        if 'TCA' not in df.columns:
            df['TCA'] = df['Tuition_USD'] + (df['Rent_USD'] * 12)
            if 'Visa_Fee_USD' in df.columns:
                df['TCA'] += df['Visa_Fee_USD']
            if 'Insurance_USD' in df.columns:
                df['TCA'] += df['Insurance_USD']
        
        # Create simplified city feature for top cities
        city_counts = df['City'].value_counts()
        top_15_cities = city_counts.head(15).index.tolist()
        self.top_cities = top_15_cities
        
        df['City_Simplified'] = df['City'].apply(
            lambda x: x if x in top_15_cities else 'Other'
        )
        
        # Select features
        feature_columns = ['Country', 'Program', 'Level', 'City_Simplified', 
                          'Duration_Years', 'Living_Cost_Index', 'Rent_USD', 
                          'Visa_Fee_USD', 'Insurance_USD']
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in df.columns:
                print(f"⚠️  Warning: Column {col} not found in dataset")
        
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df['TCA'].copy()
        
        self.feature_names = available_features
        
        return X, y
    
    def create_preprocessor(self, X):
        """Create preprocessing pipeline"""
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                ('num', StandardScaler(), numerical_features)
            ],
            remainder='passthrough'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the Gradient Boosting model with MLflow tracking"""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create preprocessor
        preprocessor = self.create_preprocessor(X)
        
        # Create the best model pipeline (Gradient Boosting)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            ))
        ])
        
        # Start MLflow run
        with mlflow.start_run(run_name="TCA-Predictor-Production") as run:
            # Log parameters
            mlflow.log_param("model_type", "GradientBoostingRegressor")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_features", len(self.feature_names))
            
            # Train the model
            print("🚀 Training Gradient Boosting model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_r2 = r2_score(y_train, y_pred_train)
            
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=self.model_name
            )
            
            # Save artifacts
            with open("top_cities_list.pkl", "wb") as f:
                pickle.dump(self.top_cities, f)
            mlflow.log_artifact("top_cities_list.pkl")
            
            # Log feature names
            with open("feature_names.pkl", "wb") as f:
                pickle.dump(self.feature_names, f)
            mlflow.log_artifact("feature_names.pkl")
            
            self.model = model
            run_id = run.info.run_id
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Test R² Score: {test_r2:.4f}")
            print(f"💰 Test MAE: ${test_mae:,.0f}")
            print(f"📈 Test RMSE: ${test_rmse:,.0f}")
            print(f"🔗 MLflow Run ID: {run_id}")
            
            return model, run_id
    
    def register_model(self, run_id, stage="Staging"):
        """Register model in MLflow Model Registry"""
        try:
            client = mlflow.MlflowClient()
            
            # Get the latest version
            latest_versions = client.get_latest_versions(
                name=self.model_name,
                stages=["None"]
            )
            
            if latest_versions:
                version = latest_versions[0].version
                
                # Transition to specified stage
                client.transition_model_version_stage(
                    name=self.model_name,
                    version=version,
                    stage=stage
                )
                
                self.model_version = version
                print(f"✅ Model {self.model_name} version {version} registered in {stage}")
                return version
            else:
                print("❌ No model versions found")
                return None
                
        except Exception as e:
            print(f"❌ Error registering model: {e}")
            return None
    
    def load_model_from_registry(self, stage="Production"):
        """Load model from MLflow Model Registry"""
        try:
            model_uri = f"models:/{self.model_name}/{stage}"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Load artifacts
            if os.path.exists("top_cities_list.pkl"):
                with open("top_cities_list.pkl", "rb") as f:
                    self.top_cities = pickle.load(f)
            
            if os.path.exists("feature_names.pkl"):
                with open("feature_names.pkl", "rb") as f:
                    self.feature_names = pickle.load(f)
            
            print(f"✅ Model loaded from registry: {model_uri}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model from registry: {e}")
            return False
    
    def predict(self, input_data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train the model first.")
        
        # Ensure input_data is a DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Apply city simplification if needed
        if 'City' in input_data.columns and self.top_cities:
            input_data['City_Simplified'] = input_data['City'].apply(
                lambda x: x if x in self.top_cities else 'Other'
            )
        
        # Select only the features used in training
        if self.feature_names:
            available_features = [col for col in self.feature_names if col in input_data.columns]
            input_data = input_data[available_features]
        
        try:
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
        
        try:
            # Get the regressor from the pipeline
            regressor = self.model.named_steps['regressor']
            
            # Get feature names after preprocessing
            preprocessor = self.model.named_steps['preprocessor']
            
            # This is simplified - in practice you'd need to handle the preprocessing
            feature_importance = regressor.feature_importances_
            
            return feature_importance
        except Exception as e:
            print(f"❌ Error getting feature importance: {e}")
            return None

def main():
    """Main function to train and register the model"""
    # Initialize predictor
    predictor = TCAPredictor()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("EduSpend_TCA_Prediction")
    
    # Load data
    df = predictor.load_data()
    if df is None:
        return
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train model
    model, run_id = predictor.train_model(X, y)
    
    # Register model
    version = predictor.register_model(run_id, stage="Production")
    
    if version:
        print(f"🎉 Model successfully registered and ready for deployment!")
        print(f"📦 Model: {predictor.model_name}")
        print(f"🔢 Version: {version}")
        print(f"🚀 Stage: Production")
    
    # Test prediction
    sample_input = {
        'Country': 'USA',
        'Program': 'Computer Science',
        'Level': 'Masters',
        'City': 'New York',
        'Duration_Years': 2.0,
        'Living_Cost_Index': 120,
        'Rent_USD': 2000,
        'Visa_Fee_USD': 500,
        'Insurance_USD': 1500
    }
    
    prediction = predictor.predict(sample_input)
    if prediction is not None:
        print(f"\n🎯 Sample Prediction:")
        print(f"Input: {sample_input}")
        print(f"Predicted TCA: ${prediction[0]:,.0f}")

if __name__ == "__main__":
    main()
