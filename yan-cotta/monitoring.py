"""
Model Monitoring and Drift Detection for EduSpend TCA Prediction
Monitors model performance and detects data drift in production
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime, timedelta
import os

class ModelMonitor:
    """
    Model monitoring system for TCA prediction model
    Tracks performance metrics, data drift, and model health
    """
    
    def __init__(self, model_name="EduSpend-TCA-Predictor", reference_data_path='data/International_Education_Costs.csv'):
        self.model_name = model_name
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.monitoring_data = []
        self.drift_threshold = 0.1  # 10% change threshold
        self.performance_threshold = {'mae_increase': 1000, 'r2_decrease': 0.05}
        
    def load_reference_data(self):
        """Load reference dataset for drift comparison"""
        try:
            self.reference_data = pd.read_csv(self.reference_data_path)
            print(f"✅ Reference data loaded: {self.reference_data.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading reference data: {e}")
            return False
    
    def calculate_distribution_stats(self, data, column):
        """Calculate distribution statistics for a column"""
        if column not in data.columns:
            return None
        
        if data[column].dtype in ['object']:
            # Categorical column
            value_counts = data[column].value_counts(normalize=True)
            return {
                'type': 'categorical',
                'distribution': value_counts.to_dict(),
                'unique_count': data[column].nunique(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None
            }
        else:
            # Numerical column
            return {
                'type': 'numerical',
                'mean': data[column].mean(),
                'std': data[column].std(),
                'median': data[column].median(),
                'min': data[column].min(),
                'max': data[column].max(),
                'q25': data[column].quantile(0.25),
                'q75': data[column].quantile(0.75)
            }
    
    def detect_data_drift(self, new_data, feature_columns=None):
        """Detect data drift between reference and new data"""
        if self.reference_data is None:
            if not self.load_reference_data():
                return None
        
        if feature_columns is None:
            feature_columns = ['Country', 'Program', 'Level', 'Living_Cost_Index', 
                             'Rent_USD', 'Visa_Fee_USD', 'Insurance_USD']
        
        drift_results = {}
        
        for column in feature_columns:
            if column not in new_data.columns or column not in self.reference_data.columns:
                continue
                
            ref_stats = self.calculate_distribution_stats(self.reference_data, column)
            new_stats = self.calculate_distribution_stats(new_data, column)
            
            if ref_stats['type'] == 'numerical':
                # Use Kolmogorov-Smirnov test for numerical features
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    new_data[column].dropna()
                )
                
                # Calculate percentage change in mean
                mean_change = abs(new_stats['mean'] - ref_stats['mean']) / ref_stats['mean']
                
                drift_detected = p_value < 0.05 or mean_change > self.drift_threshold
                
                drift_results[column] = {
                    'drift_detected': drift_detected,
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'mean_change_pct': mean_change * 100,
                    'reference_mean': ref_stats['mean'],
                    'new_mean': new_stats['mean'],
                    'reference_std': ref_stats['std'],
                    'new_std': new_stats['std']
                }
            
            else:
                # Chi-square test for categorical features
                ref_dist = ref_stats['distribution']
                new_dist = new_stats['distribution']
                
                # Align distributions
                all_categories = set(ref_dist.keys()) | set(new_dist.keys())
                ref_counts = [ref_dist.get(cat, 0) for cat in all_categories]
                new_counts = [new_dist.get(cat, 0) for cat in all_categories]
                
                # Convert to counts (approximate)
                ref_counts = np.array(ref_counts) * len(self.reference_data)
                new_counts = np.array(new_counts) * len(new_data)
                
                # Chi-square test (if we have enough data)
                if min(ref_counts) > 5 and min(new_counts) > 5:
                    chi2_stat, p_value = stats.chisquare(new_counts, ref_counts)
                    drift_detected = p_value < 0.05
                else:
                    chi2_stat, p_value = 0, 1
                    drift_detected = False
                
                drift_results[column] = {
                    'drift_detected': drift_detected,
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'reference_distribution': ref_dist,
                    'new_distribution': new_dist
                }
        
        return drift_results
    
    def evaluate_model_performance(self, model, X_test, y_test):
        """Evaluate model performance on test data"""
        try:
            y_pred = model.predict(X_test)
            
            performance = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                'predictions_count': len(y_pred),
                'timestamp': datetime.now().isoformat()
            }
            
            return performance
        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            return None
    
    def check_performance_degradation(self, current_performance, baseline_performance):
        """Check if model performance has degraded"""
        degradation_alerts = []
        
        if current_performance['mae'] > baseline_performance['mae'] + self.performance_threshold['mae_increase']:
            degradation_alerts.append({
                'metric': 'MAE',
                'current': current_performance['mae'],
                'baseline': baseline_performance['mae'],
                'increase': current_performance['mae'] - baseline_performance['mae'],
                'severity': 'high' if current_performance['mae'] > baseline_performance['mae'] * 1.2 else 'medium'
            })
        
        if current_performance['r2'] < baseline_performance['r2'] - self.performance_threshold['r2_decrease']:
            degradation_alerts.append({
                'metric': 'R2',
                'current': current_performance['r2'],
                'baseline': baseline_performance['r2'],
                'decrease': baseline_performance['r2'] - current_performance['r2'],
                'severity': 'high' if current_performance['r2'] < baseline_performance['r2'] * 0.9 else 'medium'
            })
        
        return degradation_alerts
    
    def log_monitoring_data(self, performance_data, drift_data=None):
        """Log monitoring data to MLflow and local storage"""
        try:
            with mlflow.start_run(run_name="Model-Monitoring") as run:
                # Log performance metrics
                for metric, value in performance_data.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"monitor_{metric}", value)
                
                # Log drift detection results
                if drift_data:
                    drift_count = sum(1 for result in drift_data.values() if result.get('drift_detected', False))
                    mlflow.log_metric("drift_features_count", drift_count)
                    mlflow.log_metric("total_features_monitored", len(drift_data))
                
                # Save detailed results as artifacts
                monitoring_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'performance': performance_data,
                    'drift_detection': drift_data,
                    'run_id': run.info.run_id
                }
                
                with open("monitoring_summary.json", "w") as f:
                    json.dump(monitoring_summary, f, indent=2, default=str)
                
                mlflow.log_artifact("monitoring_summary.json")
                
                print(f"✅ Monitoring data logged to MLflow: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            print(f"❌ Error logging monitoring data: {e}")
            return None
    
    def create_monitoring_dashboard(self, monitoring_history):
        """Create monitoring dashboard with visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Over Time', 'Error Distribution', 
                          'Data Drift Detection', 'Prediction Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance over time
        timestamps = [entry['timestamp'] for entry in monitoring_history]
        mae_values = [entry['performance']['mae'] for entry in monitoring_history]
        r2_values = [entry['performance']['r2'] for entry in monitoring_history]
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=mae_values, name='MAE', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=r2_values, name='R² Score', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Add baseline thresholds
        baseline_mae = monitoring_history[0]['performance']['mae'] if monitoring_history else 2447
        baseline_r2 = monitoring_history[0]['performance']['r2'] if monitoring_history else 0.9644
        
        fig.add_hline(y=baseline_mae + self.performance_threshold['mae_increase'], 
                     line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=baseline_r2 - self.performance_threshold['r2_decrease'], 
                     line_dash="dash", line_color="blue", row=1, col=2)
        
        # Prediction volume
        volumes = [entry['performance']['predictions_count'] for entry in monitoring_history]
        fig.add_trace(
            go.Bar(x=timestamps, y=volumes, name='Predictions Count'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Model Monitoring Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_monitoring_report(self, performance_data, drift_data=None, alerts=None):
        """Generate a comprehensive monitoring report"""
        report = {
            "monitoring_timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "performance_summary": {
                "current_mae": performance_data.get('mae', 'N/A'),
                "current_r2": performance_data.get('r2', 'N/A'),
                "current_rmse": performance_data.get('rmse', 'N/A'),
                "predictions_processed": performance_data.get('predictions_count', 'N/A')
            }
        }
        
        if drift_data:
            drift_summary = {
                "total_features_monitored": len(drift_data),
                "features_with_drift": [k for k, v in drift_data.items() if v.get('drift_detected', False)],
                "drift_details": drift_data
            }
            report["drift_detection"] = drift_summary
        
        if alerts:
            report["performance_alerts"] = alerts
        
        # Health status
        if alerts:
            report["health_status"] = "DEGRADED" if any(alert['severity'] == 'high' for alert in alerts) else "WARNING"
        elif drift_data and any(result.get('drift_detected', False) for result in drift_data.values()):
            report["health_status"] = "DRIFT_DETECTED"
        else:
            report["health_status"] = "HEALTHY"
        
        return report

def run_monitoring_pipeline():
    """Run the complete monitoring pipeline"""
    monitor = ModelMonitor()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("EduSpend_Model_Monitoring")
    
    # Load reference data
    if not monitor.load_reference_data():
        print("❌ Cannot proceed without reference data")
        return
    
    # Simulate new data (in production, this would come from live predictions)
    # For demo purposes, we'll use a subset of the reference data with some modifications
    new_data = monitor.reference_data.sample(100).copy()
    
    # Simulate some drift by modifying the data slightly
    new_data['Rent_USD'] = new_data['Rent_USD'] * 1.15  # 15% increase in rent
    new_data['Living_Cost_Index'] = new_data['Living_Cost_Index'] + 10  # Slight increase
    
    print("🔍 Starting monitoring pipeline...")
    
    # Detect data drift
    drift_results = monitor.detect_data_drift(new_data)
    if drift_results:
        drift_count = sum(1 for result in drift_results.values() if result.get('drift_detected', False))
        print(f"📊 Drift detection completed: {drift_count} features showing drift")
        
        for feature, result in drift_results.items():
            if result.get('drift_detected', False):
                print(f"⚠️  Drift detected in {feature}")
    
    # Simulate performance evaluation (in production, you'd use actual model predictions)
    performance_data = {
        'mae': 2650,  # Slightly higher than baseline
        'rmse': 4100,
        'r2': 0.9580,  # Slightly lower than baseline
        'mape': 12.5,
        'predictions_count': len(new_data),
        'timestamp': datetime.now().isoformat()
    }
    
    # Check for performance degradation
    baseline_performance = {'mae': 2447, 'r2': 0.9644, 'rmse': 3930}
    alerts = monitor.check_performance_degradation(performance_data, baseline_performance)
    
    if alerts:
        print(f"🚨 Performance alerts: {len(alerts)} issues detected")
        for alert in alerts:
            print(f"  - {alert['metric']}: {alert['severity']} severity")
    
    # Log monitoring data
    run_id = monitor.log_monitoring_data(performance_data, drift_results)
    
    # Generate comprehensive report
    report = monitor.generate_monitoring_report(performance_data, drift_results, alerts)
    
    print(f"\n📋 Monitoring Report:")
    print(f"Status: {report['health_status']}")
    print(f"Current MAE: ${report['performance_summary']['current_mae']:,.0f}")
    print(f"Current R²: {report['performance_summary']['current_r2']:.4f}")
    
    if drift_results:
        drift_features = report['drift_detection']['features_with_drift']
        if drift_features:
            print(f"Features with drift: {', '.join(drift_features)}")
    
    # Save report
    with open("monitoring_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Monitoring pipeline completed. Report saved to monitoring_report.json")
    if run_id:
        print(f"📊 MLflow run: {run_id}")

if __name__ == "__main__":
    run_monitoring_pipeline()
