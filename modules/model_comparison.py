"""
Model Comparison Module
=======================
Compares Exponential Growth, Random Forest, and SVR models
for waste prediction accuracy.

Uses Leave-One-Out Cross-Validation (LOOCV) due to small dataset size.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelComparison:
    """Compare different prediction models for waste generation forecasting"""
    
    def __init__(self, data_path=None):
        """Initialize with path to ml_data.csv"""
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            data_path = os.path.join(os.path.dirname(parent_dir), "Data", "ml_data.csv")
        
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
        # Model instances
        self.rf_model = None
        self.svr_model = None
        
        # Results storage
        self.results = {}
        
        # CV predictions storage for visualization
        self.cv_predictions = {}
        
    def load_data(self):
        """Load and prepare data from CSV"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'year': int(row['year']),
                    'annual_msw_tonnes': float(row['annual_msw_tonnes']),
                    'organic_pct': float(row['organic_pct']),
                    'plastic_pct': float(row['plastic_pct']),
                    'wet_waste_pct': float(row['wet_waste_pct']),
                })
        
        self.data = data
        
        # Prepare features and target
        self.X = np.array([[d['year']] for d in data])
        self.y = np.array([d['annual_msw_tonnes'] for d in data])
        
        return data
    
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
    
    def cross_validate_exponential(self):
        """LOOCV for exponential growth model"""
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            pred, _ = self._exponential_predict(X_train, y_train, X_test)
            y_true.append(y_test[0])
            y_pred.append(pred[0])
        
        return np.array(y_true), np.array(y_pred)
    
    def cross_validate_random_forest(self):
        """LOOCV for Random Forest model"""
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # RF with small-data hyperparameters
            rf = RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            
            y_true.append(y_test[0])
            y_pred.append(pred[0])
        
        return np.array(y_true), np.array(y_pred)
    
    def cross_validate_svr(self):
        """LOOCV for SVR model"""
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Scale features for SVR
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Scale target as well for better SVR performance
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # SVR with RBF kernel
            svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
            svr.fit(X_train_scaled, y_train_scaled)
            pred_scaled = svr.predict(X_test_scaled)
            
            # Inverse transform prediction
            pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            y_true.append(y_test[0])
            y_pred.append(pred[0])
        
        return np.array(y_true), np.array(y_pred)
    
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
        # Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.rf_model.fit(self.X, self.y)
        
        # SVR (with scaling)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(self.y.reshape(-1, 1)).flatten()
        
        self.svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        self.svr_model.fit(X_scaled, y_scaled)
        
        # Calculate exponential growth rate on full data
        _, self.exp_growth_rate = self._exponential_predict(self.X, self.y, self.X)
    
    def predict_future(self, years):
        """
        Generate predictions for future years using all models
        
        Args:
            years: List of years to predict (e.g., [2026, 2028, 2030, 2035])
            
        Returns:
            dict: Predictions from each model
        """
        predictions = {'Exponential Growth': {}, 'Random Forest': {}, 'SVR': {}}
        
        # Get base year info for exponential
        base_year = self.data[-1]['year']
        base_value = self.data[-1]['annual_msw_tonnes']
        
        for year in years:
            X_future = np.array([[year]])
            
            # Exponential Growth
            years_ahead = year - base_year
            exp_pred = base_value * ((1 + self.exp_growth_rate) ** years_ahead)
            predictions['Exponential Growth'][year] = max(0, exp_pred)
            
            # Random Forest
            rf_pred = self.rf_model.predict(X_future)[0]
            predictions['Random Forest'][year] = max(0, rf_pred)
            
            # SVR
            X_scaled = self.scaler.transform(X_future)
            svr_pred_scaled = self.svr_model.predict(X_scaled)
            svr_pred = self.y_scaler.inverse_transform(svr_pred_scaled.reshape(-1, 1))[0, 0]
            predictions['SVR'][year] = max(0, svr_pred)
        
        return predictions
    
    def run_comparison(self):
        """
        Run full model comparison
        
        Returns:
            dict: Comparison results with metrics and predictions
        """
        if self.data is None:
            self.load_data()
        
        # Cross-validate all models
        print("   Running Leave-One-Out Cross-Validation...")
        
        # Exponential Growth
        y_true_exp, y_pred_exp = self.cross_validate_exponential()
        metrics_exp = self.calculate_metrics(y_true_exp, y_pred_exp)
        
        # Random Forest
        y_true_rf, y_pred_rf = self.cross_validate_random_forest()
        metrics_rf = self.calculate_metrics(y_true_rf, y_pred_rf)
        
        # SVR
        y_true_svr, y_pred_svr = self.cross_validate_svr()
        metrics_svr = self.calculate_metrics(y_true_svr, y_pred_svr)
        
        # Store results
        self.results = {
            'Exponential Growth': metrics_exp,
            'Random Forest': metrics_rf,
            'SVR': metrics_svr
        }
        
        # Store CV predictions for visualization
        self.cv_predictions = {
            'years': self.X.flatten(),
            'actual': y_true_exp,  # Same for all models
            'Exponential Growth': y_pred_exp,
            'Random Forest': y_pred_rf,
            'SVR': y_pred_svr
        }
        
        # Train final models for future predictions
        print("   Training final models on full dataset...")
        self.train_final_models()
        
        # Generate future predictions
        future_years = [2026, 2028, 2030, 2035]
        self.future_predictions = self.predict_future(future_years)
        
        return self.results
    
    def get_best_model(self):
        """Determine best model based on R2 score"""
        if not self.results:
            return None
        
        best_model = max(self.results.keys(), key=lambda m: self.results[m]['R2'])
        return best_model
    
    def format_results(self):
        """Format results for display"""
        if not self.results:
            return "No results available. Run comparison first."
        
        lines = []
        lines.append("\nMODEL COMPARISON - CROSS-VALIDATION RESULTS")
        lines.append("   " + "─" * 61)
        lines.append(f"   {'Model':<20} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'MAPE':>10}")
        lines.append("   " + "─" * 61)
        
        for model_name, metrics in self.results.items():
            lines.append(
                f"   {model_name:<20} {metrics['R2']:>10.4f} "
                f"{metrics['RMSE']:>12,.0f} {metrics['MAE']:>12,.0f} "
                f"{metrics['MAPE']:>9.2f}%"
            )
        
        lines.append("   " + "─" * 61)
        
        best = self.get_best_model()
        lines.append(f"\n   Best Model (by R²): {best}")
        
        # Add future predictions comparison
        if hasattr(self, 'future_predictions') and self.future_predictions:
            lines.append("\nFUTURE PREDICTIONS COMPARISON (tonnes/year)")
            lines.append("   " + "─" * 61)
            lines.append(f"   {'Year':<8} {'Exponential':>14} {'Random Forest':>14} {'SVR':>14}")
            lines.append("   " + "─" * 61)
            
            years = sorted(self.future_predictions['Exponential Growth'].keys())
            for year in years:
                exp_val = self.future_predictions['Exponential Growth'][year]
                rf_val = self.future_predictions['Random Forest'][year]
                svr_val = self.future_predictions['SVR'][year]
                lines.append(f"   {year:<8} {exp_val:>14,.0f} {rf_val:>14,.0f} {svr_val:>14,.0f}")
            
            lines.append("   " + "─" * 61)
            
            # Add prediction variance analysis
            lines.append("\n   Prediction Analysis:")
            for year in years:
                preds = [
                    self.future_predictions['Exponential Growth'][year],
                    self.future_predictions['Random Forest'][year],
                    self.future_predictions['SVR'][year]
                ]
                avg = np.mean(preds)
                std = np.std(preds)
                cv = (std / avg) * 100  # Coefficient of variation
                lines.append(f"      {year}: Mean = {avg:,.0f} ± {std:,.0f} tonnes (CV: {cv:.1f}%)")
        
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
        colors = {'Exponential Growth': '#2ecc71', 'Random Forest': '#3498db', 'SVR': '#e74c3c'}
        
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
        
        print(f"\n   Generated {len(generated_files)} visualization graphs in: {output_dir}")
        
        return generated_files
    
    def _plot_metrics_comparison(self, output_dir, colors):
        """Bar chart comparing model metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
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
            ax.set_xticklabels(models, fontsize=10)
            
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
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, '1_metrics_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_predictions_timeline(self, output_dir, colors):
        """Line chart showing historical data and future predictions"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Historical data
        years = self.cv_predictions['years']
        actual = self.cv_predictions['actual']
        
        # Plot actual data
        ax.plot(years, actual, 'ko-', markersize=10, linewidth=2.5, label='Actual Data', zorder=5)
        
        # Plot CV predictions for historical years
        for model in ['Exponential Growth', 'Random Forest', 'SVR']:
            pred = self.cv_predictions[model]
            ax.plot(years, pred, 's--', color=colors[model], markersize=8, 
                   linewidth=1.5, alpha=0.7, label=f'{model} (CV)')
        
        # Future predictions
        if hasattr(self, 'future_predictions') and self.future_predictions:
            future_years = sorted(self.future_predictions['Exponential Growth'].keys())
            
            for model in ['Exponential Growth', 'Random Forest', 'SVR']:
                future_vals = [self.future_predictions[model][y] for y in future_years]
                # Connect last actual to first future
                all_years = list(years) + future_years
                last_actual = actual[-1]
                all_vals = [last_actual] + future_vals
                
                ax.plot([years[-1]] + future_years, all_vals, 'o-', color=colors[model], 
                       markersize=8, linewidth=2, label=f'{model} (Future)')
        
        # Styling
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Annual MSW (tonnes)', fontsize=12)
        ax.set_title('Waste Generation: Historical Data & Model Predictions', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
        
        # Add vertical line separating historical and future
        ax.axvline(x=years[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(years[-1] + 0.3, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
               'Future →', fontsize=10, color='gray')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(output_dir, '2_predictions_timeline.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def _plot_actual_vs_predicted(self, output_dir, colors):
        """Scatter plot of actual vs predicted values"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Actual vs Predicted Values (Cross-Validation)', fontsize=14, fontweight='bold')
        
        actual = self.cv_predictions['actual']
        
        for idx, model in enumerate(['Exponential Growth', 'Random Forest', 'SVR']):
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
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Residual Analysis by Model', fontsize=14, fontweight='bold')
        
        years = self.cv_predictions['years']
        actual = self.cv_predictions['actual']
        
        for idx, model in enumerate(['Exponential Growth', 'Random Forest', 'SVR']):
            pred = self.cv_predictions[model]
            residuals = actual - pred
            pct_error = ((actual - pred) / actual) * 100
            
            # Top row: Residuals over time
            ax1 = axes[0, idx]
            ax1.bar(years, residuals, color=colors[model], edgecolor='black', alpha=0.8)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.set_xlabel('Year', fontsize=10)
            ax1.set_ylabel('Residual (tonnes)', fontsize=10)
            ax1.set_title(f'{model}\nResiduals Over Time', fontsize=11)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Bottom row: Percentage error
            ax2 = axes[1, idx]
            bar_colors = ['green' if e >= 0 else 'red' for e in pct_error]
            ax2.bar(years, pct_error, color=bar_colors, edgecolor='black', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('Year', fontsize=10)
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
        
        for model in ['Exponential Growth', 'Random Forest', 'SVR']:
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
        
        for model in ['Exponential Growth', 'Random Forest', 'SVR']:
            pred = self.cv_predictions[model]
            abs_err = np.abs(actual - pred)
            pct_err = np.abs((actual - pred) / actual) * 100
            abs_errors.append(abs_err)
            pct_errors.append(pct_err)
            labels.append(model)
        
        # Absolute errors box plot
        bp1 = axes[0].boxplot(abs_errors, labels=labels, patch_artist=True)
        for patch, model in zip(bp1['boxes'], labels):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.7)
        axes[0].set_ylabel('Absolute Error (tonnes)', fontsize=11)
        axes[0].set_title('Absolute Error Distribution', fontsize=12)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Percentage errors box plot
        bp2 = axes[1].boxplot(pct_errors, labels=labels, patch_artist=True)
        for patch, model in zip(bp2['boxes'], labels):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.7)
        axes[1].set_ylabel('Percentage Error (%)', fontsize=11)
        axes[1].set_title('Percentage Error Distribution', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, '6_error_distribution.png')
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
