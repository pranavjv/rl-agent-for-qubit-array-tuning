import numpy as np
from typing import Callable, Dict, Optional, Union


class GlobalCapacitanceModel:
    """
    Class to handle globally varying capacitances
    At the ground truth coordinates, the capacitance matrices are unmodified
    in order to match the ground truth coordinate calculation
    """

    def __init__(self, base_capacitances: Dict[str, Union[np.ndarray, list]], ground_truth_coords: Dict[str, Union[np.ndarray, list]], 
                 update_func: Callable, alpha: float = 0.01, beta: float = 0.01):
        """
        Initialize global capacitance model.
        
        Args:
            base_capacitances: Dictionary with 'Cgd' and 'Cdd' base matrices
            ground_truth_coords: Dictionary with 'vg' and optionally 'vb' ground truth coordinates
            update_func: Function to update capacitances based on coordinate differences
            alpha: Scaling factor for gate voltage influence
            beta: Scaling factor for barrier voltage influence
        """
        self.Cgd_base = np.array(base_capacitances['Cgd'])
        self.Cdd_base = np.array(base_capacitances['Cdd'])

        self.vg_ground_truth = np.array(ground_truth_coords['vg'])
        vb = ground_truth_coords['vb']
        self.vb_ground_truth = np.array(vb) if vb is not None else None

        self.update_func = update_func
        self.alpha = alpha  # Gate voltage scaling factor
        self.beta = beta    # Barrier voltage scaling factor


    def update(self, gate_voltages: Union[np.ndarray, list], barrier_voltages: Union[np.ndarray, list] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Get capacitance matrices updated based on current voltage coordinates.
        
        Args:
            voltage_coords: Dictionary with 'vg' and optionally 'vb' current coordinates
            
        Returns:
            Tuple of (Cgd, Cdd) updated capacitance matrices
        """
        vg_current = np.array(gate_voltages)
        vg_diff = vg_current - self.vg_ground_truth

        if self.vb_ground_truth is not None and barrier_voltages is not None:
            vb_current = np.array(barrier_voltages)
            vb_diff = vb_current - self.vb_ground_truth
        else:
            vb_diff = None

        # Apply updates using the specified function
        Cgd_updated = self.update_func(self.Cgd_base, vg_diff, vb_diff, self.alpha, self.beta)
        Cdd_updated = self.update_func(self.Cdd_base, vg_diff, vb_diff, self.alpha, self.beta)

        return Cgd_updated, Cdd_updated


def linear_dependent_update(base_matrix: np.ndarray, vg_diff: np.ndarray, vb_diff: Optional[np.ndarray], 
                          alpha: float, beta: float, tol: float = 1e-6) -> np.ndarray:
    """
    Update capacitance matrix linearly based on voltage differences from ground truth.
    
    When vg_diff and vb_diff are zero (at ground truth coordinates), the matrix remains unmodified.
    Linear variations are applied proportionally to the distance from ground truth coordinates.

    Args:
        base_matrix: The base capacitance matrix to be modified
        vg_diff: Difference between current and ground truth gate voltages
        vb_diff: Difference between current and ground truth barrier voltages (can be None)
        alpha: Scaling factor for gate voltage influence
        beta: Scaling factor for barrier voltage influence

    Returns:
        updated_matrix: The modified capacitance matrix
    """
    updated_matrix = base_matrix.copy()
    raise NotImplementedError("TODO linear dependent update")
    
    # If we're at ground truth coordinates (all differences are zero), return unmodified matrix
    if np.allclose(vg_diff, 0, atol=tol) and (vb_diff is None or np.allclose(vb_diff, 0, atol=tol)):
        return updated_matrix

    # update diagonal elements based on the correct ranges in gt diff
    # update neareset neighbours based on gt diff

    return updated_matrix


    
    # Calculate the total voltage deviation magnitude for scaling
    vg_magnitude = np.linalg.norm(vg_diff) if len(vg_diff) > 0 else 0.0
    vb_magnitude = np.linalg.norm(vb_diff) if vb_diff is not None and len(vb_diff) > 0 else 0.0
    
    # Linear scaling based on voltage deviations
    # This affects the entire matrix proportionally to distance from ground truth
    gate_scaling = alpha * vg_magnitude
    barrier_scaling = beta * vb_magnitude if vb_diff is not None else 0.0
    
    # Apply scaling to the matrix elements
    # You can customize this logic based on how you want capacitances to vary
    total_scaling = gate_scaling + barrier_scaling
    
    # Example: Scale all matrix elements by a small fraction based on total deviation
    # This ensures smooth variation while keeping the matrix well-conditioned
    scaling_factor = 1.0 + total_scaling
    updated_matrix = updated_matrix * scaling_factor
    
    # Alternative approach: Add position-dependent variations to diagonal elements
    # for i in range(min(len(vg_diff), updated_matrix.shape[0])):
    #     updated_matrix[i, i] += alpha * vg_diff[i]
    
    return updated_matrix

