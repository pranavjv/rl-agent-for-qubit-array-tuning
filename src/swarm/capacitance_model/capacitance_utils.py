import numpy as np
from typing import List, Tuple


def get_channel_targets(channel_idx: int, cgd_matrix: np.ndarray, num_dots: int, has_sensor: bool = True) -> np.ndarray:
    """
    Get the target CGD values for a specific channel.
    
    For each channel (1-indexed as specified):
    - Channel 0 (image 1): [0, Cgd[1,2], Cgd[1,3]] (Cgd[0,2] doesn't exist, so use 0)
    - Channel 1 (image 2): [Cgd[2,3], Cgd[2,4], Cgd[1,3]]
    - Channel 2 (image 3): [Cgd[3,4], Cgd[3,5], Cgd[2,4]]
    - etc.
    
    Args:
        channel_idx: Index of the leftmost dot in the pair being swept (0-indexed)
        cgd_matrix: CGD matrix of shape (num_dots, num_dots+1)
        num_dots: Number of dots in the system
        
    Returns:
        List of 3 target values for this channel
    """

    assert channel_idx in list(range(num_dots-1)), f"Out-of-bounds channel index given for {num_dots} dots."

    if has_sensor:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] - 1 == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots+1})"
    else:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots})"

    # Extract the pairs of dots to consider for the channel index (0-indexed)
    left_pair = (channel_idx-1, channel_idx+1) # out of bounds for channel 0
    middle_pair = (channel_idx, channel_idx+1)
    right_pair = (channel_idx, channel_idx+2) # out of bounds for last channel

    # # Convert to 1-indexed for the logic as specified
    # channel_1indexed = channel_idx + 1
    
    # # Calculate the three target indices for this channel (1-indexed)
    # target1_row = channel_1indexed - 1  # Will be -1 for channel 0, which doesn't exist
    # target1_col = channel_1indexed + 1
    
    # target2_row = channel_1indexed
    # target2_col = channel_1indexed + 1
    
    # target3_row = channel_1indexed - 1  # Will be -1 for channel 0  - ? should be not minus 1??
    # target3_col = channel_1indexed + 2
    
    targets = []

    if left_pair[0] < 0:
        targets.append(0.0)
    else:
        targets.append(float(cgd_matrix[left_pair[0], left_pair[1]]))

    targets.append(float(cgd_matrix[middle_pair[0], middle_pair[1]]))

    if right_pair[1] > num_dots - 1:
        targets.append(0.0)
    else:
        targets.append(float(cgd_matrix[right_pair[0], right_pair[1]]))

    # # Target 1: Check if indices are valid (convert back to 0-indexed for array access)
    # if target1_row < 0 or target1_row >= num_dots or target1_col >= cgd_matrix.shape[1]:
    #     targets.append(0.0)  # Pad with zero if doesn't exist
    # else:
    #     targets.append(float(cgd_matrix[target1_row, target1_col]))
    
    # # Target 2: Check if indices are valid
    # if target2_row >= num_dots or target2_col >= cgd_matrix.shape[1]:
    #     targets.append(0.0)
    # else:
    #     targets.append(float(cgd_matrix[target2_row, target2_col]))
    
    # # Target 3: Check if indices are valid
    # if target3_row < 0 or target3_row >= num_dots or target3_col >= cgd_matrix.shape[1]:
    #     targets.append(0.0)
    # else:
    #     targets.append(float(cgd_matrix[target3_row, target3_col]))

    # reorder to match expected model output: l, m, r -> m, r, l
    targets = np.array(targets, dtype=np.float32)[[1, 2, 0]]
    
    return targets