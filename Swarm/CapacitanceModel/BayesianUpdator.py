import numpy as np
from typing import Dict, Tuple, List, Union, Callable
import warnings


class CapacitancePredictor:
    """
    Bayesian parameter estimation for fixed capacitance values in a quantum dot array.
    
    This class maintains a symmetric N×N capacitance matrix where each element represents
    the capacitance between two quantum dots. Each matrix element stores both the mean
    and variance of a Gaussian posterior distribution, which is updated using conjugate
    Bayesian inference as new measurements arrive from ML model predictions.
    
    Attributes:
        n_dots (int): Number of quantum dots in the array
        means (np.ndarray): N×N matrix of posterior means
        variances (np.ndarray): N×N matrix of posterior variances
        prior_config (Dict or Callable): Configuration for prior distributions
    """
    
    def __init__(self, n_dots: int, prior_config: Union[Dict[Tuple[int, int], Tuple[float, float]], Callable]):
        """
        Initialize the capacitance predictor with prior distributions.
        
        Args:
            n_dots (int): Number of quantum dots in the array
            prior_config (Dict or Callable): Either a dictionary mapping (i,j) pairs to 
                (prior_mean, prior_variance) tuples, or a callable that takes (i,j) and 
                returns (prior_mean, prior_variance)
        
        Example:
            # Dictionary-based prior
            priors = {(i,j): (0.25, 0.1) for i in range(5) for j in range(5)}
            predictor = CapacitancePredictor(5, priors)
            
            # Function-based prior (distance-dependent)
            def distance_prior(i, j):
                if i == j:
                    return (0.5, 0.05)  # Self-capacitance
                elif abs(i-j) == 1:
                    return (0.25, 0.1)  # Nearest neighbors
                else:
                    return (0.1, 0.2)   # Distant pairs
            predictor = CapacitancePredictor(5, distance_prior)
        """
        self.n_dots = n_dots
        self.prior_config = prior_config
        
        # Initialize symmetric matrices for means and variances
        self.means = np.zeros((n_dots, n_dots))
        self.variances = np.zeros((n_dots, n_dots))
        
        # Set up prior distributions
        self._initialize_priors()
        
        # Validation
        self._validate_initialization()
    
    def _initialize_priors(self):
        """Initialize the prior distributions for all matrix elements."""
        for i in range(self.n_dots):
            for j in range(self.n_dots):
                if callable(self.prior_config):
                    prior_mean, prior_var = self.prior_config(i, j)
                else:
                    prior_mean, prior_var = self.prior_config.get((i, j), (0.25, 0.1))
                
                self.means[i, j] = prior_mean
                self.variances[i, j] = prior_var
    
    def _validate_initialization(self):
        """Validate the initialization of the matrices."""
        # Check symmetry
        if not np.allclose(self.means, self.means.T):
            raise ValueError("Mean matrix is not symmetric")
        if not np.allclose(self.variances, self.variances.T):
            raise ValueError("Variance matrix is not symmetric")
        
        # Check positive variances
        if np.any(self.variances <= 0):
            raise ValueError("All variances must be positive")
    
    def bayesian_update(self, i: int, j: int, ml_estimate: float, ml_variance: float):
        """
        Perform conjugate Bayesian update for a single capacitance element.
        
        Uses the conjugate prior property of Gaussian distributions to update
        the posterior mean and variance given a new measurement.
        
        Args:
            i (int): First dot index
            j (int): Second dot index
            ml_estimate (float): ML model's capacitance estimate
            ml_variance (float): Variance/uncertainty of the ML estimate
        
        Raises:
            ValueError: If indices are invalid or variances are non-positive
        """
        # Validate inputs
        if not (0 <= i < self.n_dots and 0 <= j < self.n_dots):
            raise ValueError(f"Invalid indices: ({i}, {j}). Must be in range [0, {self.n_dots})")
        if ml_variance <= 0:
            raise ValueError("ML variance must be positive")
        
        # Get current posterior for element (i,j)
        current_mean = self.means[i, j]
        current_var = self.variances[i, j]
        
        # Conjugate update formulas
        precision_prior = 1 / current_var
        precision_ml = 1 / ml_variance
        precision_post = precision_prior + precision_ml
        
        new_mean = (current_mean * precision_prior + ml_estimate * precision_ml) / precision_post
        new_var = 1 / precision_post
        
        # Sanity check: posterior variance should not increase
        if new_var > current_var:
            warnings.warn(f"Posterior variance increased for element ({i},{j}). "
                         f"Old: {current_var:.6f}, New: {new_var:.6f}")
        
        # Update both (i,j) and (j,i) to maintain symmetry
        self.means[i, j] = new_mean
        self.means[j, i] = new_mean
        self.variances[i, j] = new_var
        self.variances[j, i] = new_var
    
    def convert_logprobs_to_variance(self, confidence: float) -> float:
        """
        Convert ML model confidence (log probabilities) to variance estimate.
        
        This is a heuristic conversion that maps confidence scores to measurement
        uncertainty. Higher confidence (closer to 0) results in lower variance.
        
        Args:
            confidence (float): Confidence score from ML model (typically negative log prob)
        
        Returns:
            float: Estimated variance for the measurement
        
        Note:
            This conversion may need to be calibrated based on your specific ML model.
            The current implementation uses an exponential mapping.
        """
        # Convert negative log probability to variance
        # Higher confidence (less negative) -> lower variance
        # This is a heuristic that may need calibration
        base_variance = 0.01
        scaling_factor = 10.0
        
        # Ensure confidence is treated as negative log probability
        if confidence > 0:
            confidence = -confidence
        
        variance = base_variance * np.exp(-confidence / scaling_factor)
        return max(variance, 1e-6)  # Minimum variance threshold
    
    def update_from_scan(self, dot_pair: Tuple[int, int], ml_outputs: List[Tuple[float, float]]):
        """
        Process ML model output for a dot pair scan and update relevant capacitances.
        
        Args:
            dot_pair (Tuple[int, int]): The measured dot pair (i, j)
            ml_outputs (List[Tuple[float, float]]): List of 3 tuples containing
                (capacitance_estimate, confidence) for:
                - C_ij: Capacitance between the measured dots
                - C_ik: Capacitance between dot i and neighboring dot k
                - C_jk: Capacitance between dot j and neighboring dot k
        
        Example:
            predictor.update_from_scan(
                dot_pair=(2, 3),
                ml_outputs=[(0.23, 0.1), (0.18, 0.15), (0.31, 0.08)]
            )
        """
        if len(ml_outputs) != 3:
            raise ValueError("ml_outputs must contain exactly 3 measurements")
        
        i, j = dot_pair
        
        # First measurement: direct capacitance C_ij
        estimate_ij, confidence_ij = ml_outputs[0]
        variance_ij = self.convert_logprobs_to_variance(confidence_ij)
        self.bayesian_update(i, j, estimate_ij, variance_ij)
        
        # Second and third measurements: neighboring capacitances
        # We need to determine which neighboring dots these correspond to
        # This is a simplified approach - in practice, you might need more
        # sophisticated logic to identify the specific neighboring dots
        
        for idx, (estimate, confidence) in enumerate(ml_outputs[1:], 1):
            variance = self.convert_logprobs_to_variance(confidence)
            
            # Find appropriate neighboring dot pairs
            # This is a heuristic - you may need to modify based on your specific setup
            if idx == 1:  # C_ik - find neighbor of i
                k = self._find_neighbor(i, exclude=[j])
                if k is not None:
                    self.bayesian_update(i, k, estimate, variance)
            elif idx == 2:  # C_jk - find neighbor of j
                k = self._find_neighbor(j, exclude=[i])
                if k is not None:
                    self.bayesian_update(j, k, estimate, variance)
    
    def _find_neighbor(self, dot_idx: int, exclude: List[int] = None) -> Union[int, None]:
        """
        Find a neighboring dot for the given dot index.
        
        This is a simplified neighbor-finding heuristic. In practice, you might
        have a more sophisticated graph structure or physical layout.
        
        Args:
            dot_idx (int): Index of the dot to find a neighbor for
            exclude (List[int]): List of dot indices to exclude from consideration
        
        Returns:
            int or None: Index of a neighboring dot, or None if none found
        """
        if exclude is None:
            exclude = []
        
        # Simple linear arrangement assumption - neighbors are ±1 index
        candidates = [dot_idx - 1, dot_idx + 1]
        
        for neighbor in candidates:
            if (0 <= neighbor < self.n_dots and 
                neighbor not in exclude and 
                neighbor != dot_idx):
                return neighbor
        
        return None
    
    def get_capacitance_stats(self, i: int, j: int) -> Tuple[float, float]:
        """
        Return current mean and variance estimates for a specific capacitance.
        
        Args:
            i (int): First dot index
            j (int): Second dot index
        
        Returns:
            Tuple[float, float]: (mean, variance) of the posterior distribution
        
        Raises:
            ValueError: If indices are invalid
        """
        if not (0 <= i < self.n_dots and 0 <= j < self.n_dots):
            raise ValueError(f"Invalid indices: ({i}, {j}). Must be in range [0, {self.n_dots})")
        
        return self.means[i, j], self.variances[i, j]
    
    def get_full_matrix(self, return_variance: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return the full symmetric matrix of current estimates.
        
        Args:
            return_variance (bool): If True, return both mean and variance matrices
        
        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: 
                If return_variance=False: matrix of means
                If return_variance=True: (means_matrix, variances_matrix)
        """
        if return_variance:
            return self.means.copy(), self.variances.copy()
        else:
            return self.means.copy()
    
    def get_confidence_interval(self, i: int, j: int, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for a specific capacitance estimate.
        
        Args:
            i (int): First dot index
            j (int): Second dot index
            confidence_level (float): Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound) of confidence interval
        """
        mean, variance = self.get_capacitance_stats(i, j)
        std = np.sqrt(variance)
        
        # For Gaussian distribution
        from scipy.stats import norm
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha/2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return lower, upper
    
    def reset_element(self, i: int, j: int):
        """
        Reset a specific matrix element to its prior distribution.
        
        Args:
            i (int): First dot index
            j (int): Second dot index
        """
        if callable(self.prior_config):
            prior_mean, prior_var = self.prior_config(i, j)
        else:
            prior_mean, prior_var = self.prior_config.get((i, j), (0.25, 0.1))
        
        self.means[i, j] = prior_mean
        self.means[j, i] = prior_mean
        self.variances[i, j] = prior_var
        self.variances[j, i] = prior_var
    
    def get_matrix_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of the current capacitance matrix.
        
        Returns:
            Dict[str, float]: Dictionary containing summary statistics
        """
        # Only consider upper triangular (excluding diagonal) to avoid double counting
        upper_tri_indices = np.triu_indices(self.n_dots, k=1)
        off_diagonal_means = self.means[upper_tri_indices]
        off_diagonal_vars = self.variances[upper_tri_indices]
        
        # Diagonal elements (self-capacitance)
        diagonal_means = np.diag(self.means)
        diagonal_vars = np.diag(self.variances)
        
        return {
            'off_diagonal_mean_avg': np.mean(off_diagonal_means),
            'off_diagonal_mean_std': np.std(off_diagonal_means),
            'off_diagonal_var_avg': np.mean(off_diagonal_vars),
            'diagonal_mean_avg': np.mean(diagonal_means),
            'diagonal_var_avg': np.mean(diagonal_vars),
            'total_uncertainty': np.sum(off_diagonal_vars) + np.sum(diagonal_vars)
        }


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Dictionary-based prior configuration
    n_dots = 5
    prior_dict = {}
    
    for i in range(n_dots):
        for j in range(n_dots):
            if i == j:
                prior_dict[(i, j)] = (0.5, 0.05)  # Self-capacitance
            elif abs(i - j) == 1:
                prior_dict[(i, j)] = (0.25, 0.1)  # Nearest neighbors
            else:
                prior_dict[(i, j)] = (0.1, 0.2)   # Distant pairs
    
    predictor = CapacitancePredictor(n_dots, prior_dict)
    
    # Example scan update
    predictor.update_from_scan(
        dot_pair=(2, 3),
        ml_outputs=[(0.23, 0.1), (0.18, 0.15), (0.31, 0.08)]
    )
    
    # Get specific capacitance stats
    mean, var = predictor.get_capacitance_stats(2, 3)
    print(f"C(2,3): mean={mean:.4f}, variance={var:.6f}")
    
    # Get confidence interval
    lower, upper = predictor.get_confidence_interval(2, 3)
    print(f"95% CI for C(2,3): [{lower:.4f}, {upper:.4f}]")
    
    # Get full matrix
    full_matrix = predictor.get_full_matrix()
    print(f"Full capacitance matrix:\n{full_matrix}")
    
    # Get summary statistics
    summary = predictor.get_matrix_summary()
    print(f"Matrix summary: {summary}")
