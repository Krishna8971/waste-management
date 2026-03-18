"""
Model Comparison Module
=======================
Compares Exponential Growth, Random Forest, and SVR models
for waste prediction accuracy.

Uses Time Series Split Cross-Validation (forward chaining) for proper temporal validation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelComparison:
    """Compare different prediction models for waste generation forecasting"""
    
    def __init__(self, data_path=None):
        """Initialize with path to monthly bengaluru_msw_monthly_2018_2025_clean.csv"""
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            data_path = os.path.join(os.path.dirname(parent_dir), "Data", "bengaluru_msw_monthly_2018_2025_clean.csv")
        
        self.data_path = data_path
        self.data = None
        self.X = None
        self.X_enhanced = None
        self.y = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        # Model registry
        self.baseline_models = ['Naive Baseline', 'Avg Growth Baseline']
        self.ml_models = [
            'Random Forest',
            'Gradient Boosting',
            'Linear Regression',
            'Ridge Regression',
            'SVR'
        ]
        self.all_models = self.baseline_models + self.ml_models
        self.trained_models = {}
        self.model_scalers = {}
        
        # Model instances
        self.rf_model = None
        self.svr_model = None
        
        # Results storage
        self.results = {}
        
        # CV predictions storage for visualization
        self.cv_predictions = {}
        
    def load_data(self):
        """Load monthly CSV and build engineered feature matrix."""
        df = pd.read_csv(self.data_path)

        # Parse/sort temporal fields to ensure strict time order for time-series validation.
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['monthly_msw_tonnes'] = df['monthly_msw_tonnes'].astype(float)
        df = df.sort_values(['year', 'month']).reset_index(drop=True)
        df['time_index'] = np.arange(len(df), dtype=int)

        # ------------------------------
        # PRD Feature Engineering
        # ------------------------------
        # 1) Time-based features
        df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        df['is_festival_season'] = df['month'].isin([10, 11, 12]).astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 2) Lag features
        df['lag_1'] = df['monthly_msw_tonnes'].shift(1)
        df['lag_3'] = df['monthly_msw_tonnes'].shift(3)
        df['lag_6'] = df['monthly_msw_tonnes'].shift(6)

        # 3) Rolling statistics (use only past data via shift(1) to avoid leakage)
        df['roll_mean_3'] = df['monthly_msw_tonnes'].shift(1).rolling(3).mean()
        df['roll_mean_6'] = df['monthly_msw_tonnes'].shift(1).rolling(6).mean()
        df['roll_std_3'] = df['monthly_msw_tonnes'].shift(1).rolling(3).std()
        df['roll_std_6'] = df['monthly_msw_tonnes'].shift(1).rolling(6).std()

        # 4) / 5) / 6) Efficiency, growth and ratio features from lagged information.
        df['monthly_growth_rate'] = df['lag_1'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['processing_growth'] = df['monthly_growth_rate']

        # 7) Waste utilization from tonnage-only signals (no percentage columns).
        df['organic_estimate_tonnes'] = df['roll_mean_3']
        df['recyclable_potential_tonnes'] = np.maximum(0.0, df['roll_mean_6'] - df['roll_mean_3'])
        df['biogas_potential_m3'] = df['organic_estimate_tonnes'] * 100.0

        processed_proxy = df['organic_estimate_tonnes'] + df['recyclable_potential_tonnes']
        collected_proxy = df['lag_1']
        df['processing_efficiency'] = (processed_proxy / collected_proxy).replace([np.inf, -np.inf], np.nan)
        df['waste_gap_tonnes'] = collected_proxy - processed_proxy
        df['processed_to_collected_ratio'] = df['processing_efficiency']
        df['unprocessed_ratio'] = 1.0 - df['processing_efficiency']

        # 8) Anomaly flags from growth thresholds
        df['spike_flag'] = (df['monthly_growth_rate'] > 0.08).astype(int)
        df['drop_flag'] = (df['monthly_growth_rate'] < -0.08).astype(int)

        # 9) Normalization feature (raw time + minmax normalized time)
        df['time_index_norm'] = self.minmax_scaler.fit_transform(df[['time_index']].to_numpy(dtype=float))

        # Remove rows without full lag/rolling context.
        feature_df = df.dropna().reset_index(drop=True)

        self.feature_columns = [
            'time_index', 'time_index_norm', 'year', 'month', 'quarter',
            'is_monsoon', 'is_festival_season', 'month_sin', 'month_cos',
            'lag_1', 'lag_3', 'lag_6',
            'roll_mean_3', 'roll_mean_6', 'roll_std_3', 'roll_std_6',
            'monthly_growth_rate', 'processing_growth',
            'processing_efficiency', 'waste_gap_tonnes',
            'processed_to_collected_ratio', 'unprocessed_ratio',
            'organic_estimate_tonnes', 'biogas_potential_m3', 'recyclable_potential_tonnes',
            'spike_flag', 'drop_flag'
        ]

        self.X = feature_df[['time_index']].to_numpy(dtype=float)  # Time axis for exponential + plotting
        self.X_enhanced = feature_df[self.feature_columns].to_numpy(dtype=float)
        self.y = feature_df['monthly_msw_tonnes'].to_numpy(dtype=float)

        # Keep a row-wise dict for downstream reporting/plot labels.
        self.data = feature_df.to_dict('records')
        return self.data
    
    def _exponential_predict(self, X_train, y_train, X_test):
        """
        Exponential growth prediction based on training data
        Fits: y = a * (1 + r)^(year - base_year)
        """
        # Calculate growth rate from training data
        years = X_train.flatten()
        values = y_train
        
        # Sort by year
        sorted_idx = np.argsort(years)
        years = years[sorted_idx]
        values = values[sorted_idx]
        
        # Calculate year-over-year growth rates
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                rate = (values[i] - values[i-1]) / values[i-1]
                growth_rates.append(rate)
        
        if not growth_rates:
            avg_rate = 0.05  # Default
        else:
            avg_rate = np.mean(growth_rates)
        
        # Use first year as base
        base_year = years[0]
        base_value = values[0]
        
        # Predict for test years
        predictions = []
        for test_year in X_test.flatten():
            years_diff = test_year - base_year
            pred = base_value * ((1 + avg_rate) ** years_diff)
            predictions.append(pred)
        
        return np.array(predictions), avg_rate
    
    def cross_validate_naive(self):
        """Naive baseline: predict next points as last observed train value."""
        tscv = TimeSeriesSplit(n_splits=4)
        y_true = []
        y_pred = []
        test_time_idx = []

        for train_idx, test_idx in tscv.split(self.X):
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            test_time_idx.extend(self.X[test_idx].flatten())
            last_value = y_train[-1]
            pred = np.full_like(y_test, fill_value=last_value, dtype=float)
            y_true.extend(y_test)
            y_pred.extend(pred)

        return np.array(y_true), np.array(y_pred), np.array(test_time_idx)

    def cross_validate_avg_growth(self):
        """Average growth baseline: recursive monthly growth from training history."""
        tscv = TimeSeriesSplit(n_splits=4)
        y_true = []
        y_pred = []
        test_time_idx = []

        for train_idx, test_idx in tscv.split(self.X):
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            test_time_idx.extend(self.X[test_idx].flatten())

            growth_rates = []
            for i in range(1, len(y_train)):
                prev = y_train[i - 1]
                if prev > 0:
                    growth_rates.append((y_train[i] - prev) / prev)
            avg_growth = float(np.mean(growth_rates)) if growth_rates else 0.0

            pred = []
            last = y_train[-1]
            for _ in range(len(y_test)):
                next_val = max(0.0, last * (1.0 + avg_growth))
                pred.append(next_val)
                last = next_val

            y_true.extend(y_test)
            y_pred.extend(pred)

        return np.array(y_true), np.array(y_pred), np.array(test_time_idx)

    def _create_ml_model(self, model_name):
        """Factory for ML models used in comparison."""
        if model_name == 'Random Forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        if model_name == 'Gradient Boosting':
            return GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        if model_name == 'Linear Regression':
            return LinearRegression()
        if model_name == 'Ridge Regression':
            return Ridge(alpha=1.0, random_state=42)
        if model_name == 'SVR':
            return SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        raise ValueError(f'Unknown model: {model_name}')

    def _cross_validate_ml_model(self, model_name):
        """Time Series CV for ML model on engineered features."""
        tscv = TimeSeriesSplit(n_splits=4)
        y_true = []
        y_pred = []
        test_time_idx = []

        for train_idx, test_idx in tscv.split(self.X_enhanced):
            X_train, X_test = self.X_enhanced[train_idx], self.X_enhanced[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            test_time_idx.extend(self.X[test_idx].flatten())

            model = self._create_ml_model(model_name)

            if model_name == 'SVR':
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                y_scaler = StandardScaler()
                y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                model.fit(X_train, y_train_scaled)
                pred_scaled = model.predict(X_test)
                pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

            y_true.extend(y_test)
            y_pred.extend(pred)

        return np.array(y_true), np.array(y_pred), np.array(test_time_idx)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        # Handle any negative predictions by clipping
        y_pred = np.maximum(y_pred, 0)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def train_final_models(self):
        """Train models on full dataset for future predictions"""
        self.trained_models = {}
        self.model_scalers = {}

        for model_name in self.ml_models:
            model = self._create_ml_model(model_name)

            if model_name == 'SVR':
                x_scaler = StandardScaler()
                y_scaler = StandardScaler()
                X_scaled = x_scaler.fit_transform(self.X_enhanced)
                y_scaled = y_scaler.fit_transform(self.y.reshape(-1, 1)).flatten()
                model.fit(X_scaled, y_scaled)
                self.model_scalers[model_name] = {'x': x_scaler, 'y': y_scaler}
            else:
                model.fit(self.X_enhanced, self.y)

            self.trained_models[model_name] = model

        # Baseline parameters
        self.naive_last_value = float(self.y[-1])
        growth_rates = []
        for i in range(1, len(self.y)):
            prev = self.y[i - 1]
            if prev > 0:
                growth_rates.append((self.y[i] - prev) / prev)
        self.avg_growth_rate = float(np.mean(growth_rates)) if growth_rates else 0.0

    def _build_future_feature_row(self, history_values, future_time_index, future_year, future_month):
        """Build one engineered feature row for recursive future prediction."""
        lag_1 = history_values[-1]
        lag_3 = history_values[-3]
        lag_6 = history_values[-6]

        roll_window_3 = np.array(history_values[-3:], dtype=float)
        roll_window_6 = np.array(history_values[-6:], dtype=float)

        roll_mean_3 = float(np.mean(roll_window_3))
        roll_mean_6 = float(np.mean(roll_window_6))
        roll_std_3 = float(np.std(roll_window_3, ddof=1))
        roll_std_6 = float(np.std(roll_window_6, ddof=1))

        lag_2 = history_values[-2]
        monthly_growth_rate = ((lag_1 - lag_2) / lag_2) if lag_2 > 0 else 0.0
        processing_growth = monthly_growth_rate

        # Tonnage-only utilization features (no percentage inputs).
        organic_estimate_tonnes = roll_mean_3
        recyclable_potential_tonnes = max(0.0, roll_mean_6 - roll_mean_3)
        biogas_potential_m3 = organic_estimate_tonnes * 100.0

        processed_proxy = organic_estimate_tonnes + recyclable_potential_tonnes
        processing_efficiency = (processed_proxy / lag_1) if lag_1 > 0 else 0.0
        waste_gap_tonnes = lag_1 - processed_proxy
        processed_to_collected_ratio = processing_efficiency
        unprocessed_ratio = 1.0 - processing_efficiency

        quarter = ((future_month - 1) // 3) + 1
        is_monsoon = 1 if future_month in [6, 7, 8, 9] else 0
        is_festival_season = 1 if future_month in [10, 11, 12] else 0
        month_sin = np.sin(2 * np.pi * future_month / 12)
        month_cos = np.cos(2 * np.pi * future_month / 12)
        time_index_norm = float(self.minmax_scaler.transform(np.array([[future_time_index]], dtype=float))[0, 0])

        spike_flag = 1 if monthly_growth_rate > 0.08 else 0
        drop_flag = 1 if monthly_growth_rate < -0.08 else 0

        row = [
            float(future_time_index), time_index_norm, float(future_year), float(future_month), float(quarter),
            float(is_monsoon), float(is_festival_season), float(month_sin), float(month_cos),
            float(lag_1), float(lag_3), float(lag_6),
            roll_mean_3, roll_mean_6, roll_std_3, roll_std_6,
            float(monthly_growth_rate), float(processing_growth),
            float(processing_efficiency), float(waste_gap_tonnes),
            float(processed_to_collected_ratio), float(unprocessed_ratio),
            float(organic_estimate_tonnes), float(biogas_potential_m3), float(recyclable_potential_tonnes),
            float(spike_flag), float(drop_flag)
        ]

        return np.array([row], dtype=float)
    
    def calculate_prediction_intervals(self):
        """
        Calculate prediction intervals (ranges) based on CV errors
        Uses 95% confidence interval (±1.96 standard errors)
        """
        intervals = {}
        
        for model_name, metrics in self.results.items():
            rmse = metrics['RMSE']
            # 95% confidence interval = 1.96 * RMSE
            interval = 1.96 * rmse
            intervals[model_name] = interval
        
        self.prediction_intervals = intervals
        return intervals
    
    def predict_future(self, months_ahead_list=None):
        """
        Generate predictions for future months using all models with prediction ranges
        
        Args:
            months_ahead_list: List of months ahead to predict (e.g., [1, 3, 6, 12])
                             Default: [1, 3, 6, 12, 24, 36] (roughly 1 month to 3 years ahead)
            
        Returns:
            dict: Predictions from each model with lower and upper bounds
        """
        if months_ahead_list is None:
            months_ahead_list = [1, 3, 6, 12, 24, 36]
        
        predictions = {model_name: {} for model_name in self.all_models}

        # Get last data point info
        last_time_index = int(self.data[-1]['time_index'])
        last_value = float(self.data[-1]['monthly_msw_tonnes'])
        base_year = int(self.data[-1]['year'])
        base_month = int(self.data[-1]['month'])
        
        # Get intervals if not already calculated
        if not hasattr(self, 'prediction_intervals'):
            self.calculate_prediction_intervals()
        
        # Baseline predictions for requested horizons.
        for months_ahead in months_ahead_list:
            pred_label = f'Month+{months_ahead}'
            naive_pred = max(0, self.naive_last_value)
            avg_growth_pred = max(0, last_value * ((1 + self.avg_growth_rate) ** months_ahead))

            naive_interval = self.prediction_intervals['Naive Baseline']
            growth_interval = self.prediction_intervals['Avg Growth Baseline']

            predictions['Naive Baseline'][pred_label] = {
                'point': naive_pred,
                'lower': max(0, naive_pred - naive_interval),
                'upper': naive_pred + naive_interval,
                'months_ahead': months_ahead
            }
            predictions['Avg Growth Baseline'][pred_label] = {
                'point': avg_growth_pred,
                'lower': max(0, avg_growth_pred - growth_interval),
                'upper': avg_growth_pred + growth_interval,
                'months_ahead': months_ahead
            }

        # Recursive multi-step forecasting for all ML models with engineered features.
        max_horizon = max(months_ahead_list)
        month_set = set(months_ahead_list)

        histories = {model_name: list(self.y.astype(float)) for model_name in self.ml_models}

        for step in range(1, max_horizon + 1):
            future_time_index = last_time_index + step
            abs_month = (base_month - 1 + step)
            future_year = base_year + (abs_month // 12)
            future_month = (abs_month % 12) + 1

            for model_name in self.ml_models:
                row = self._build_future_feature_row(
                    histories[model_name], future_time_index, future_year, future_month
                )
                model = self.trained_models[model_name]

                if model_name == 'SVR':
                    x_scaler = self.model_scalers[model_name]['x']
                    y_scaler = self.model_scalers[model_name]['y']
                    row_scaled = x_scaler.transform(row)
                    pred_scaled = model.predict(row_scaled)
                    pred_val = float(max(0, y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]))
                else:
                    pred_val = float(max(0, model.predict(row)[0]))

                histories[model_name].append(pred_val)

                if step in month_set:
                    pred_label = f'Month+{step}'
                    interval = self.prediction_intervals[model_name]
                    predictions[model_name][pred_label] = {
                        'point': pred_val,
                        'lower': max(0, pred_val - interval),
                        'upper': pred_val + interval,
                        'months_ahead': step
                    }
        
        return predictions
    
    def run_comparison(self):
        """
        Run full model comparison with proper time series validation
        
        Returns:
            dict: Comparison results with metrics and predictions
        """
        if self.data is None:
            self.load_data()
        
        # Cross-validate all models using Time Series Split
        print("   Running Time Series Split Cross-Validation (forward chaining)...")
        
        self.results = {}
        cv_predictions = {}

        y_true_ref = None
        test_time_ref = None

        y_true_naive, y_pred_naive, test_time = self.cross_validate_naive()
        self.results['Naive Baseline'] = self.calculate_metrics(y_true_naive, y_pred_naive)
        cv_predictions['Naive Baseline'] = y_pred_naive
        y_true_ref = y_true_naive
        test_time_ref = test_time

        y_true_growth, y_pred_growth, test_time_growth = self.cross_validate_avg_growth()
        self.results['Avg Growth Baseline'] = self.calculate_metrics(y_true_growth, y_pred_growth)
        cv_predictions['Avg Growth Baseline'] = y_pred_growth

        for model_name in self.ml_models:
            y_true_ml, y_pred_ml, _ = self._cross_validate_ml_model(model_name)
            self.results[model_name] = self.calculate_metrics(y_true_ml, y_pred_ml)
            cv_predictions[model_name] = y_pred_ml

        # Store CV predictions for visualization
        self.cv_predictions = {
            'time_index': test_time_ref,
            'actual': y_true_ref
        }
        self.cv_predictions.update(cv_predictions)
        
        # Calculate prediction intervals based on CV errors
        print("   Calculating prediction intervals (95% CI)...")
        self.calculate_prediction_intervals()
        
        # Train final models for future predictions
        print("   Training final models on full dataset...")
        self.train_final_models()
        
        # Generate future predictions (1, 3, 6, 12, 24, 36 months ahead)
        self.future_predictions = self.predict_future()
        
        return self.results
    
    def get_best_model(self):
        """Determine best model based on R2 score"""
        if not self.results:
            return None
        
        best_model = max(self.results.keys(), key=lambda m: self.results[m]['R2'])
        return best_model

    def _ordered_models(self):
        """Return models in configured comparison order with available results."""
        return [m for m in self.all_models if m in self.results]

    def _top_models(self, n=3):
        """Return top-n models by R2 for dense visualizations."""
        ordered = sorted(self.results.keys(), key=lambda m: self.results[m]['R2'], reverse=True)
        return ordered[:max(1, min(n, len(ordered)))]

    def _model_key(self, models):
        """Create short model codes for compact x-axis labels."""
        return {model: f"M{i + 1}" for i, model in enumerate(models)}
    
    def format_results(self):
        """Format results for display"""
        if not self.results:
            return "No results available. Run comparison first."
        
        lines = []
        lines.append("\nMODEL COMPARISON - TIME SERIES SPLIT CROSS-VALIDATION RESULTS")
        lines.append("   " + "─" * 65)
        lines.append("   Validation Method: Time Series Split (forward chaining)")
        lines.append("   Prediction Intervals: 95% Confidence (±1.96 × RMSE)")
        lines.append("   " + "─" * 65)
        lines.append(f"   {'Model':<20} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'MAPE':>10}")
        lines.append("   " + "─" * 65)
        
        for model_name in self._ordered_models():
            metrics = self.results[model_name]
            lines.append(
                f"   {model_name:<20} {metrics['R2']:>10.4f} "
                f"{metrics['RMSE']:>12,.0f} {metrics['MAE']:>12,.0f} "
                f"{metrics['MAPE']:>9.2f}%"
            )
        
        lines.append("   " + "─" * 65)
        
        best = self.get_best_model()
        lines.append(f"\n   Best Model (by R²): {best}")
        
        # Add future predictions comparison with ranges
        if hasattr(self, 'future_predictions') and self.future_predictions:
            lines.append("\n" + "─" * 65)
            lines.append("FUTURE PREDICTIONS WITH 95% CONFIDENCE INTERVALS (tonnes/month)")
            lines.append("─" * 65)
            
            first_model = self._ordered_models()[0]
            months_labels = sorted(
                self.future_predictions[first_model].keys(),
                key=lambda x: self.future_predictions[first_model][x]['months_ahead']
            )
            
            for model_name in self._ordered_models():
                lines.append(f"\n{model_name}:")
                lines.append("   " + "─" * 62)
                lines.append(f"   {'Forecast':<12} {'Point Estimate':>15} {'Lower Bound':>15} {'Upper Bound':>15}")
                lines.append("   " + "─" * 62)
                
                for month_label in months_labels:
                    pred_data = self.future_predictions[model_name][month_label]
                    point = pred_data['point']
                    lower = pred_data['lower']
                    upper = pred_data['upper']
                    lines.append(
                        f"   {month_label:<12} {point:>15,.0f} {lower:>15,.0f} {upper:>15,.0f}"
                    )
                
                lines.append("   " + "─" * 62)
            
            # Add prediction range analysis
            lines.append("\nPREDICTION RANGE ANALYSIS:")
            lines.append("   " + "─" * 62)
            
            for month_label in months_labels:
                lines.append(f"\n   {month_label}:")
                for model_name in self._ordered_models():
                    pred_data = self.future_predictions[model_name][month_label]
                    point = pred_data['point']
                    lower = pred_data['lower']
                    upper = pred_data['upper']
                    range_width = upper - lower
                    lines.append(
                        f"      {model_name:<20}: {point:>12,.0f} " 
                        f"(Range: {range_width:>12,.0f} tonnes)"
                    )
        
        return "\n".join(lines)
    
    def generate_visualizations(self, output_dir=None):
        """
        Generate all comparison graphs and save to files
        
        Args:
            output_dir: Directory to save graphs (default: waste_analysis/graphs/)
            
        Returns:
            list: Paths to generated graph files
        """
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(script_dir), "graphs")
        
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = {
            'Naive Baseline': '#7f8c8d',
            'Avg Growth Baseline': '#16a085',
            'Random Forest': '#3498db',
            'Gradient Boosting': '#f39c12',
            'Linear Regression': '#8e44ad',
            'Ridge Regression': '#2c3e50',
            'SVR': '#e74c3c'
        }
        
        # 1. Bar Chart - Metrics Comparison
        fig1_path = self._plot_metrics_comparison(output_dir, colors)
        generated_files.append(fig1_path)
        
        # 2. Line Chart - Historical + Future Predictions
        fig2_path = self._plot_predictions_timeline(output_dir, colors)
        generated_files.append(fig2_path)
        
        # 3. Scatter Plot - Actual vs Predicted
        fig3_path = self._plot_actual_vs_predicted(output_dir, colors)
        generated_files.append(fig3_path)
        
        # 4. Residual Analysis
        fig4_path = self._plot_residuals(output_dir, colors)
        generated_files.append(fig4_path)
        
        # 5. Radar Chart - Model Performance
        fig5_path = self._plot_radar_chart(output_dir, colors)
        generated_files.append(fig5_path)
        
        # 6. Error Distribution Box Plot
        fig6_path = self._plot_error_distribution(output_dir, colors)
        generated_files.append(fig6_path)
        
        # 7. Prediction Ranges - New visualization for confidence intervals
        fig7_path = self._plot_prediction_ranges(output_dir, colors)
        generated_files.append(fig7_path)
        
        print(f"\n   Generated {len(generated_files)} visualization graphs in: {output_dir}")
        
        return generated_files
    
    def _plot_metrics_comparison(self, output_dir, colors):
        """Bar chart comparing model metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        model_codes = self._model_key(models)
        x = np.arange(len(models))
        width = 0.6
        
        metrics_config = [
            ('R2', 'R² Score (Higher is Better)', axes[0, 0], True),
            ('RMSE', 'RMSE - Root Mean Square Error\n(Lower is Better)', axes[0, 1], False),
            ('MAE', 'MAE - Mean Absolute Error\n(Lower is Better)', axes[1, 0], False),
            ('MAPE', 'MAPE % - Mean Absolute Percentage Error\n(Lower is Better)', axes[1, 1], False)
        ]
        
        for metric, title, ax, higher_better in metrics_config:
            values = [self.results[m][metric] for m in models]
            bars = ax.bar(x, values, width, color=[colors[m] for m in models], edgecolor='black', linewidth=1.2)
            
            # Highlight best model
            if higher_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([model_codes[m] for m in models], fontsize=10)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if metric == 'MAPE':
                    label = f'{val:.2f}%'
                elif metric == 'R2':
                    label = f'{val:.4f}'
                else:
                    label = f'{val:,.0f}'
                ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        key_handles = [
            mpatches.Patch(facecolor=colors[m], edgecolor='black', label=f"{model_codes[m]}: {m}")
            for m in models
        ]
        fig.legend(handles=key_handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
                   title='Model Key', fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        filepath = os.path.join(output_dir, '1_metrics_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_predictions_timeline(self, output_dir, colors):
        """Line chart showing historical data and future predictions with confidence intervals"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # All historical data
        all_years = self.X.flatten()
        all_actual = self.y
        
        # Cross-validation test data
        cv_years = self.cv_predictions['time_index']
        cv_actual = self.cv_predictions['actual']
        
        # Plot all historical data as a continuous line
        ax.plot(all_years, all_actual, 'k-', linewidth=2, alpha=0.6, label='Historical Data (Full)', zorder=3)
        
        # Plot actual CV test points
        ax.plot(cv_years, cv_actual, 'ko', markersize=10, label='CV Test Points', zorder=5)
        
        # Plot CV predictions for test years
        for model in self._ordered_models():
            pred = self.cv_predictions[model]
            ax.plot(cv_years, pred, 's--', color=colors[model], markersize=8, 
                   linewidth=1.5, alpha=0.7, label=f'{model} (CV)')
        
        # Future predictions with confidence intervals
        if hasattr(self, 'future_predictions') and self.future_predictions:
            first_model = self._ordered_models()[0]
            future_years = sorted(
                self.future_predictions[first_model].keys(),
                key=lambda x: self.future_predictions[first_model][x]['months_ahead']
            )
            
            for idx, model in enumerate(self._ordered_models()):
                # Extract point estimates, lower and upper bounds
                points = [self.future_predictions[model][y]['point'] for y in future_years]
                lower = [self.future_predictions[model][y]['lower'] for y in future_years]
                upper = [self.future_predictions[model][y]['upper'] for y in future_years]
                months_ahead_list = [self.future_predictions[model][y]['months_ahead'] for y in future_years]
                
                # Connect last historical to first future
                last_year = all_years[-1]
                last_actual_val = all_actual[-1]
                connect_years = [last_year] + [last_year + m for m in months_ahead_list]
                connect_points = [last_actual_val] + points
                connect_lower = [last_actual_val] + lower
                connect_upper = [last_actual_val] + upper
                
                # Plot point estimates
                ax.plot(connect_years, connect_points, 'o-', color=colors[model], 
                       markersize=8, linewidth=2.5, label=f'{model} (Future)', zorder=4)
                
                # Add confidence interval band
                ax.fill_between(connect_years, connect_lower, connect_upper, 
                               color=colors[model], alpha=0.15, zorder=2)
        
        # Styling
        ax.set_xlabel('Month (since Jan 2018)', fontsize=12)
        ax.set_ylabel('Monthly MSW (tonnes)', fontsize=12)
        ax.set_title('Waste Generation: Historical Data & Model Predictions (with 95% CI)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
        
        # Add vertical line separating historical and future
        ax.axvline(x=all_years[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(all_years[-1] + 2, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
               'Future →', fontsize=10, color='gray')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(output_dir, '2_predictions_timeline.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_actual_vs_predicted(self, output_dir, colors):
        """Scatter plot of actual vs predicted values"""
        models = self._top_models(n=4)
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
        fig.suptitle('Actual vs Predicted Values (Cross-Validation)', fontsize=14, fontweight='bold')
        if len(models) == 1:
            axes = [axes]
        
        actual = self.cv_predictions['actual']
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            pred = self.cv_predictions[model]
            
            # Scatter plot
            ax.scatter(actual, pred, c=colors[model], s=100, edgecolors='black', 
                      linewidth=1.5, alpha=0.8, zorder=5)
            
            # Perfect prediction line
            min_val = min(min(actual), min(pred)) * 0.95
            max_val = max(max(actual), max(pred)) * 1.05
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
                   label='Perfect Prediction', alpha=0.7)
            
            # Add R² annotation
            r2 = self.results[model]['R2']
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Actual (tonnes)', fontsize=11)
            ax.set_ylabel('Predicted (tonnes)', fontsize=11)
            ax.set_title(model, fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, '3_actual_vs_predicted.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_residuals(self, output_dir, colors):
        """Residual analysis plot"""
        models = self._top_models(n=4)
        fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 10))
        fig.suptitle('Residual Analysis by Model', fontsize=14, fontweight='bold')
        if len(models) == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        
        time_idx = self.cv_predictions['time_index']
        actual = self.cv_predictions['actual']
        
        for idx, model in enumerate(models):
            pred = self.cv_predictions[model]
            residuals = actual - pred
            pct_error = ((actual - pred) / actual) * 100
            
            # Top row: Residuals over time
            ax1 = axes[0, idx]
            ax1.bar(time_idx, residuals, color=colors[model], edgecolor='black', alpha=0.8)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.set_xlabel('Month', fontsize=10)
            ax1.set_ylabel('Residual (tonnes)', fontsize=10)
            ax1.set_title(f'{model}\nResiduals Over Time', fontsize=11)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Bottom row: Percentage error
            ax2 = axes[1, idx]
            bar_colors = ['green' if e >= 0 else 'red' for e in pct_error]
            ax2.bar(time_idx, pct_error, color=bar_colors, edgecolor='black', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('Month', fontsize=10)
            ax2.set_ylabel('Percentage Error (%)', fontsize=10)
            ax2.set_title(f'Percentage Error\nMAPE: {self.results[model]["MAPE"]:.2f}%', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, '4_residual_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_radar_chart(self, output_dir, colors):
        """Radar chart for overall model comparison"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Metrics to compare (normalized for radar chart)
        categories = ['R² Score', 'Accuracy\n(1-MAPE/100)', 'Precision\n(1/RMSE scaled)', 
                     'Consistency\n(1/MAE scaled)']
        N = len(categories)
        
        # Normalize metrics for radar (all should be 0-1, higher is better)
        max_rmse = max(self.results[m]['RMSE'] for m in self.results)
        max_mae = max(self.results[m]['MAE'] for m in self.results)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        for model in self._ordered_models():
            metrics = self.results[model]
            values = [
                metrics['R2'],
                1 - metrics['MAPE'] / 100,
                1 - (metrics['RMSE'] / max_rmse) * 0.5,  # Inverted and scaled
                1 - (metrics['MAE'] / max_mae) * 0.5    # Inverted and scaled
            ]
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[model])
            ax.fill(angles, values, alpha=0.25, color=colors[model])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, '5_radar_chart.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_error_distribution(self, output_dir, colors):
        """Box plot showing error distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Prediction Error Distribution', fontsize=14, fontweight='bold')
        
        actual = self.cv_predictions['actual']
        
        # Prepare data for box plots
        abs_errors = []
        pct_errors = []
        labels = []
        
        for model in self._ordered_models():
            pred = self.cv_predictions[model]
            abs_err = np.abs(actual - pred)
            pct_err = np.abs((actual - pred) / actual) * 100
            abs_errors.append(abs_err)
            pct_errors.append(pct_err)
            labels.append(model)
        
        model_codes = self._model_key(labels)
        short_labels = [model_codes[m] for m in labels]

        # Absolute errors box plot
        bp1 = axes[0].boxplot(abs_errors, labels=short_labels, patch_artist=True)
        for patch, model in zip(bp1['boxes'], labels):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.7)
        axes[0].set_ylabel('Absolute Error (tonnes)', fontsize=11)
        axes[0].set_title('Absolute Error Distribution', fontsize=12)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Percentage errors box plot
        bp2 = axes[1].boxplot(pct_errors, labels=short_labels, patch_artist=True)
        for patch, model in zip(bp2['boxes'], labels):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.7)
        axes[1].set_ylabel('Percentage Error (%)', fontsize=11)
        axes[1].set_title('Percentage Error Distribution', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        key_handles = [
            mpatches.Patch(facecolor=colors[m], edgecolor='black', label=f"{model_codes[m]}: {m}")
            for m in labels
        ]
        fig.legend(handles=key_handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
                   title='Model Key', fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        filepath = os.path.join(output_dir, '6_error_distribution.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_prediction_ranges(self, output_dir, colors):
        """Visualize prediction ranges with 95% confidence intervals"""
        if not hasattr(self, 'future_predictions') or not self.future_predictions:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Future Waste Predictions with 95% Confidence Intervals', 
                    fontsize=16, fontweight='bold')
        
        models = self._top_models(n=3)
        model_codes = self._model_key(models)
        first_model = models[0]
        future_months = sorted(
            self.future_predictions[first_model].keys(),
            key=lambda x: self.future_predictions[first_model][x]['months_ahead']
        )
        
        # Subplot 1-3: Individual model predictions with ranges
        for idx, model in enumerate(models):
            ax = axes.flat[idx]
            
            points = [self.future_predictions[model][m]['point'] for m in future_months]
            lower = [self.future_predictions[model][m]['lower'] for m in future_months]
            upper = [self.future_predictions[model][m]['upper'] for m in future_months]
            errors = [points[i] - lower[i] for i in range(len(points))]
            
            # Bar chart with error bars
            x_pos = np.arange(len(future_months))
            bars = ax.bar(x_pos, points, color=colors[model], alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
            ax.errorbar(x_pos, points, yerr=errors, fmt='none', 
                       ecolor='black', capsize=5, capthick=2, linewidth=2)
            
            # Add value labels
            for i, (month_label, point) in enumerate(zip(future_months, points)):
                ax.text(i, point + errors[i] + 5000, f'{point/1e3:.0f}K', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
                ax.text(i, lower[i] - 5000, f'{lower[i]/1e3:.0f}K', 
                       ha='center', va='top', fontsize=8, alpha=0.7)
            
            ax.set_xlabel('Forecast Horizon', fontsize=11)
            ax.set_ylabel('Waste (tonnes)', fontsize=11)
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(future_months, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(bottom=0)
        
        # Subplot 4: Comparison of all models for 1 month ahead
        ax = axes.flat[3]
        month_1m = 'Month+1'  # 1 month ahead
        x_pos = np.arange(len(models))
        width = 0.6
        
        points_1m = [self.future_predictions[model][month_1m]['point'] for model in models]
        lower_1m = [self.future_predictions[model][month_1m]['lower'] for model in models]
        upper_1m = [self.future_predictions[model][month_1m]['upper'] for model in models]
        errors_1m = [points_1m[i] - lower_1m[i] for i in range(len(points_1m))]
        
        bars = ax.bar(x_pos, points_1m, width, 
                     color=[colors[m] for m in models], alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        ax.errorbar(x_pos, points_1m, yerr=errors_1m, fmt='none', 
                   ecolor='black', capsize=8, capthick=2, linewidth=2)
        
        # Add value labels
        for i, (point, error) in enumerate(zip(points_1m, errors_1m)):
            ax.text(i, point + error + 2000, f'{point/1e3:.0f}K', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Waste (tonnes)', fontsize=11)
        ax.set_title(f'Model Comparison for Month+1 (Point ± 95% CI)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([model_codes[m] for m in models], fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        
        key_handles = [
            mpatches.Patch(facecolor=colors[m], edgecolor='black', label=f"{model_codes[m]}: {m}")
            for m in models
        ]
        fig.legend(handles=key_handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
                   title='Model Key', fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        filepath = os.path.join(output_dir, '7_prediction_ranges.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath


def run_model_comparison():
    """Standalone function to run model comparison"""
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON STUDY")
    print("=" * 70)
    
    comparison = ModelComparison()
    comparison.load_data()
    comparison.run_comparison()
    
    print(comparison.format_results())
    
    return comparison


if __name__ == "__main__":
    run_model_comparison()
