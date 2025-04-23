#!/usr/bin/env python3
"""
Simplified modeling module for wildfire prediction project.
"""
import os
import pandas as pd
import numpy as np
import pickle
import argparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def build_model(features_df):
    """Build XGBoost regression model to predict fire intensity (FRP)."""
    if features_df is None or features_df.empty:
        print("No feature data available for modeling")
        return None, None, None
    
    print(f"Building model with {len(features_df)} wildfire events")
    
    # Define features to use - simplified list
    feature_columns = [
        'avg_temp_30d', 'max_temp_30d', 'avg_max_temp_30d',
        'total_precip_30d', 'days_without_rain',
        'avg_wind_speed', 'max_wind_speed',
        'avg_dewpoint', 'temp_precip_ratio',
        'fire_month'
    ]
    
    # Handle any missing features
    valid_features = [col for col in feature_columns if col in features_df.columns]
    
    print(f"Using {len(valid_features)} features: {valid_features}")
    
    # Prepare data
    X = features_df[valid_features]
    y = features_df['fire_frp']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Build model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"Model Results:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"R² Score: {test_r2:.2f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': valid_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop Features:")
    print(importance_df.head())
    
    return model, scaler, importance_df

def load_features(year):
    """Load feature data for modeling."""
    features_path = f'../data/features/{year}/wildfire_features_{year}.csv'
    try:
        features = pd.read_csv(features_path)
        print(f"Loaded feature data with {len(features)} records")
        return features
    except FileNotFoundError:
        print(f"Feature data not found at {features_path}")
        return None

def save_model(model, scaler, importance_df, year):
    """Save model and related data."""
    output_dir = f'../models/{year}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    with open(os.path.join(output_dir, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save importance
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    print(f"Saved model and related data to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build XGBoost model for wildfire prediction')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2019)')
    args = parser.parse_args()
    
    year = args.year
    
    # Load feature data
    features = load_features(year)
    
    if features is not None:
        # Build model
        model, scaler, importance_df = build_model(features)
        
        # Save model and results
        if model is not None:
            save_model(model, scaler, importance_df, year)
    
    print("Modeling complete!")