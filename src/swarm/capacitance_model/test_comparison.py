"""
Test script comparing Bayesian vs Kriging updaters for variable capacitance scenarios.

This script simulates a model outputting variable capacitance values at different voltage
coordinates and compares the accuracy of both update methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

from BayesianUpdater import CapacitancePredictor as BayesianUpdater
from KrigingUpdater import InterpolatedCapacitancePredictor as KrigingUpdater


class CapacitanceSimulator:
    """Simulates ground truth capacitance that varies with voltage coordinates."""
    
    def __init__(self, n_dots: int = 5, base_capacitance: float = 0.25):
        self.n_dots = n_dots
        self.base_capacitance = base_capacitance
        
        # Create ground truth spatial variation patterns
        self.spatial_patterns = {}
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_dots):
            for j in range(i, n_dots):
                if i == j:
                    # Self-capacitance has less spatial variation
                    amplitude = 0.05
                    freq_x, freq_y = np.random.uniform(0.5, 2.0, 2)
                    phase_x, phase_y = np.random.uniform(0, 2*np.pi, 2)
                    base_val = 0.5
                else:
                    # Inter-dot capacitance varies more spatially
                    amplitude = 0.1 if abs(i-j) == 1 else 0.05
                    freq_x, freq_y = np.random.uniform(0.3, 1.5, 2)
                    phase_x, phase_y = np.random.uniform(0, 2*np.pi, 2)
                    base_val = base_capacitance
                
                self.spatial_patterns[(i,j)] = {
                    'base': base_val,
                    'amplitude': amplitude,
                    'freq_x': freq_x, 'freq_y': freq_y,
                    'phase_x': phase_x, 'phase_y': phase_y
                }
    
    def get_true_capacitance(self, i: int, j: int, voltage_coords: Tuple[float, float]) -> float:
        """Get the true capacitance value at given voltage coordinates."""
        key = (min(i,j), max(i,j))  # Ensure symmetry
        pattern = self.spatial_patterns[key]
        
        vx, vy = voltage_coords
        spatial_variation = (
            pattern['amplitude'] * 
            np.sin(pattern['freq_x'] * vx + pattern['phase_x']) *
            np.cos(pattern['freq_y'] * vy + pattern['phase_y'])
        )
        
        return pattern['base'] + spatial_variation
    
    def generate_ml_output(self, dot_pair: Tuple[int, int], voltage_coords: Tuple[float, float], 
                          noise_level: float = 0.05, bias: float = 0.0) -> List[Tuple[float, float]]:
        """
        Simulate ML model output with noise and potential bias.
        Returns [(capacitance_estimate, log_variance), ...] for 3 measurements.
        """
        i, j = dot_pair
        
        # Generate the 3 capacitance values as per scan protocol
        dot_pairs = [(i, j), (i, j+1), (i-1, j)]
        ml_outputs = []
        
        for pair_i, pair_j in dot_pairs:
            if pair_i < 0 or pair_j >= self.n_dots or pair_i >= self.n_dots or pair_j < 0:
                # Handle edge cases - use a default value
                true_val = 0.1
            else:
                true_val = self.get_true_capacitance(pair_i, pair_j, voltage_coords)
            
            # Add noise and bias to simulate ML model imperfection
            noise = np.random.normal(0, noise_level)
            estimate = true_val + noise + bias
            
            # Log variance represents model confidence (lower = more confident)
            # More noise in estimate should correlate with higher uncertainty
            log_variance = np.log(noise_level**2 + 0.001)  # Add small constant for numerical stability
            
            ml_outputs.append((estimate, log_variance))
        
        return ml_outputs


def create_prior_config(n_dots: int) -> Callable:
    """Create prior configuration function."""
    def distance_prior(i: int, j: int) -> Tuple[float, float]:
        if i == j:
            return (0.5, 0.05)  # Self-capacitance
        elif abs(i-j) == 1:
            return (0.25, 0.1)  # Adjacent dots
        elif abs(i-j) == 2:
            return (0.15, 0.08)  # Next-adjacent
        else:
            return (0.05, 0.1)  # Distant pairs
    return distance_prior


def run_comparison_test(n_dots: int = 5, n_test_points: int = 50, n_measurements_per_point: int = 3):
    """
    Run comparison test between Bayesian and Kriging updaters.
    
    Args:
        n_dots: Number of quantum dots
        n_test_points: Number of different voltage coordinate points to test
        n_measurements_per_point: Number of measurements at each point
    """
    
    print(f"Running comparison with {n_dots} dots, {n_test_points} test points, {n_measurements_per_point} measurements per point")
    
    # Initialize simulator and updaters
    simulator = CapacitanceSimulator(n_dots)
    prior_config = create_prior_config(n_dots)
    
    bayesian_updater = BayesianUpdater(n_dots, prior_config)
    kriging_updater = KrigingUpdater(n_dots, prior_config)
    
    # Generate test voltage coordinates
    np.random.seed(123)
    voltage_coords = []
    for _ in range(n_test_points):
        vx = np.random.uniform(-2.0, 2.0)
        vy = np.random.uniform(-2.0, 2.0)
        voltage_coords.append((vx, vy))
    
    # Track errors over time
    bayesian_errors = []
    kriging_errors = []
    test_points = []
    
    # Run sequential updates
    for point_idx, coords in enumerate(voltage_coords):
        for measurement_idx in range(n_measurements_per_point):
            # Choose which dot pair to scan (cycle through available pairs)
            scan_options = [(i, i+1) for i in range(n_dots-1)]
            dot_pair = scan_options[measurement_idx % len(scan_options)]
            
            # Add some variability in noise and bias to simulate realistic conditions
            noise_level = np.random.uniform(0.02, 0.08)
            bias = np.random.uniform(-0.02, 0.02)
            
            # Generate ML outputs
            ml_outputs = simulator.generate_ml_output(dot_pair, coords, noise_level, bias)
            
            # Update both predictors
            bayesian_updater.update_from_scan(dot_pair, ml_outputs)
            kriging_updater.update_from_scan(dot_pair, coords, ml_outputs)
            
            # Calculate current errors against ground truth
            bayesian_error = calculate_prediction_error(bayesian_updater, simulator, coords, n_dots)
            kriging_error = calculate_prediction_error(kriging_updater, simulator, coords, n_dots)
            
            bayesian_errors.append(bayesian_error)
            kriging_errors.append(kriging_error)
            test_points.append(point_idx * n_measurements_per_point + measurement_idx)
    
    return test_points, bayesian_errors, kriging_errors, bayesian_updater, kriging_updater, simulator


def calculate_prediction_error(updater, simulator, coords: Tuple[float, float], n_dots: int) -> float:
    """Calculate RMS error between updater predictions and ground truth."""
    errors = []
    
    for i in range(n_dots):
        for j in range(i, n_dots):  # Only upper triangle to avoid double counting
            pred_mean, pred_var = updater.get_capacitance_stats(i, j)
            true_val = simulator.get_true_capacitance(i, j, coords)
            error = (pred_mean - true_val) ** 2
            errors.append(error)
    
    return np.sqrt(np.mean(errors))


def plot_comparison_results(test_points, bayesian_errors, kriging_errors):
    """Plot comparison results."""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Error over time
    plt.subplot(2, 2, 1)
    plt.plot(test_points, bayesian_errors, 'b-', alpha=0.7, label='Bayesian Updater')
    plt.plot(test_points, kriging_errors, 'r-', alpha=0.7, label='Kriging Updater')
    plt.xlabel('Measurement Number')
    plt.ylabel('RMS Error')
    plt.title('Prediction Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Moving average errors
    window = 10
    plt.subplot(2, 2, 2)
    bayesian_smooth = np.convolve(bayesian_errors, np.ones(window)/window, mode='valid')
    kriging_smooth = np.convolve(kriging_errors, np.ones(window)/window, mode='valid')
    
    plt.plot(test_points[window-1:], bayesian_smooth, 'b-', linewidth=2, label='Bayesian (smoothed)')
    plt.plot(test_points[window-1:], kriging_smooth, 'r-', linewidth=2, label='Kriging (smoothed)')
    plt.xlabel('Measurement Number')
    plt.ylabel('RMS Error (Moving Average)')
    plt.title(f'Smoothed Error Trends (window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(2, 2, 3)
    plt.hist(bayesian_errors, bins=30, alpha=0.6, label='Bayesian', color='blue', density=True)
    plt.hist(kriging_errors, bins=30, alpha=0.6, label='Kriging', color='red', density=True)
    plt.xlabel('RMS Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final comparison stats
    plt.subplot(2, 2, 4)
    final_window = 20  # Last N measurements
    final_bayesian = np.mean(bayesian_errors[-final_window:])
    final_kriging = np.mean(kriging_errors[-final_window:])
    
    methods = ['Bayesian', 'Kriging']
    final_errors = [final_bayesian, final_kriging]
    colors = ['blue', 'red']
    
    bars = plt.bar(methods, final_errors, color=colors, alpha=0.7)
    plt.ylabel('Average RMS Error')
    plt.title(f'Final Performance (last {final_window} measurements)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('capacitance_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved as 'capacitance_comparison_results.png'")


def print_summary_statistics(bayesian_errors, kriging_errors):
    """Print summary statistics of the comparison."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    final_window = 20
    
    print(f"\nOverall Performance:")
    print(f"  Bayesian Mean Error: {np.mean(bayesian_errors):.6f}")
    print(f"  Kriging Mean Error:  {np.mean(kriging_errors):.6f}")
    print(f"  Bayesian Std Error:  {np.std(bayesian_errors):.6f}")
    print(f"  Kriging Std Error:   {np.std(kriging_errors):.6f}")
    
    print(f"\nFinal Performance (last {final_window} measurements):")
    final_bayesian = np.mean(bayesian_errors[-final_window:])
    final_kriging = np.mean(kriging_errors[-final_window:])
    print(f"  Bayesian Final Error: {final_bayesian:.6f}")
    print(f"  Kriging Final Error:  {final_kriging:.6f}")
    
    improvement = (final_bayesian - final_kriging) / final_bayesian * 100
    print(f"  Kriging Improvement: {improvement:.2f}%")
    
    # Statistical significance test (simple t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(bayesian_errors, kriging_errors)
    print(f"\nStatistical Test (paired t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    if p_value < 0.05:
        better_method = "Kriging" if np.mean(kriging_errors) < np.mean(bayesian_errors) else "Bayesian"
        print(f"  Result: {better_method} is significantly better (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")


if __name__ == "__main__":
    print("Capacitance Updater Comparison Test")
    print("=" * 50)
    
    # Run the main comparison
    test_points, bayesian_errors, kriging_errors, bayesian_updater, kriging_updater, simulator = run_comparison_test(
        n_dots=5, 
        n_test_points=30, 
        n_measurements_per_point=4
    )
    
    # Display results
    print_summary_statistics(bayesian_errors, kriging_errors)
    
    # Plot results
    plot_comparison_results(test_points, bayesian_errors, kriging_errors)
    
    print("\n" + "="*60)
    print("Test completed. Check the generated plots for visual comparison.")