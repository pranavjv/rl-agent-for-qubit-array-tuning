import torch
import torch.utils.data as data
import numpy as np
import os
import glob
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


def get_channel_targets(channel_idx: int, cgd_matrix: np.ndarray, num_dots: int) -> List[float]:
    """
    Get the target CGD values for a specific channel.
    
    For each channel (1-indexed as specified):
    - Channel 0 (image 1): [0, Cgd[1,2], Cgd[1,3]] (Cgd[0,2] doesn't exist, so use 0)
    - Channel 1 (image 2): [Cgd[2,3], Cgd[2,4], Cgd[1,3]]
    - Channel 2 (image 3): [Cgd[3,4], Cgd[3,5], Cgd[2,4]]
    - etc.
    
    Args:
        channel_idx: Channel index (0-indexed)
        cgd_matrix: CGD matrix of shape (num_dots, num_dots+1)
        num_dots: Number of dots in the system
        
    Returns:
        List of 3 target values for this channel
    """
    # Convert to 1-indexed for the logic as specified
    channel_1indexed = channel_idx + 1
    
    # Calculate the three target indices for this channel (1-indexed)
    target1_row = channel_1indexed - 1  # Will be -1 for channel 0, which doesn't exist
    target1_col = channel_1indexed + 1
    
    target2_row = channel_1indexed
    target2_col = channel_1indexed + 1
    
    target3_row = channel_1indexed - 1  # Will be -1 for channel 0
    target3_col = channel_1indexed + 2
    
    targets = []
    
    # Target 1: Check if indices are valid (convert back to 0-indexed for array access)
    if target1_row < 0 or target1_row >= num_dots or target1_col >= cgd_matrix.shape[1]:
        targets.append(0.0)  # Pad with zero if doesn't exist
    else:
        targets.append(float(cgd_matrix[target1_row, target1_col]))
    
    # Target 2: Check if indices are valid
    if target2_row >= num_dots or target2_col >= cgd_matrix.shape[1]:
        targets.append(0.0)
    else:
        targets.append(float(cgd_matrix[target2_row, target2_col]))
    
    # Target 3: Check if indices are valid
    if target3_row < 0 or target3_row >= num_dots or target3_col >= cgd_matrix.shape[1]:
        targets.append(0.0)
    else:
        targets.append(float(cgd_matrix[target3_row, target3_col]))
    
    return targets


class CapacitanceDataset(data.Dataset):
    """
    PyTorch dataset for loading multi-channel capacitance images and Cgd targets.
    
    Loads data from batched NPY files containing multi-channel images and CGD matrices.
    Each image channel corresponds to different charge sensor measurements with different target CGD values.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        transform: Optional[callable] = None,
        load_to_memory: bool = False,
        validate_data: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory containing images/ and parameters/
            transform: Optional image transforms
            load_to_memory: If True, load all data to memory (requires ~11.5GB)
            validate_data: If True, validate data consistency during initialization
        """
        self.data_dir = data_dir
        self.transform = transform
        self.load_to_memory = load_to_memory
        
        # Get all batch files
        self.image_files = sorted(glob.glob(os.path.join(data_dir, "images", "batch_*.npy")))
        self.cgd_files = sorted(glob.glob(os.path.join(data_dir, "cgd_matrices", "batch_*.npy")))
        
        assert len(self.image_files) == len(self.cgd_files), \
            f"Mismatch: {len(self.image_files)} image files vs {len(self.cgd_files)} cgd files"
        
        print(f"Found {len(self.image_files)} batch files")
        
        # Build index mapping global_idx -> (file_idx, local_idx, channel_idx)
        self._build_index()
        
        if validate_data:
            self._validate_data()
        
        if load_to_memory:
            print("Loading all data to memory...")
            self._load_to_memory()
        
    def _build_index(self):
        """Build mapping from global index to (batch_file_idx, local_idx, channel_idx)"""
        self.index_map = []
        self.batch_sizes = []
        
        for file_idx, (img_file, cgd_file) in enumerate(zip(self.image_files, self.cgd_files)):
            # Load images to get batch size and number of channels
            images = np.load(img_file, allow_pickle=True)
            batch_size = images.shape[0]
            num_channels = images.shape[-1] if len(images.shape) == 4 else 1  # (batch, H, W, channels)
            
            self.batch_sizes.append(batch_size)
            
            # Create index entries for each sample and each channel
            for local_idx in range(batch_size):
                for channel_idx in range(num_channels):
                    self.index_map.append((file_idx, local_idx, channel_idx))
        
        self.total_samples = len(self.index_map)
        print(f"Total samples: {self.total_samples} (including all channels)")
        
    def _validate_data(self):
        """Validate data consistency across a few sample batches"""
        print("Validating data consistency...")
        
        # Check first, middle, and last batch
        check_indices = [0, len(self.image_files) // 2, len(self.image_files) - 1]
        
        for idx in check_indices:
            img_file = self.image_files[idx]
            cgd_file = self.cgd_files[idx]
            
            images = np.load(img_file, allow_pickle=True)
            cgd_matrices = np.load(cgd_file, allow_pickle=True)
            
            # Check shapes and structure
            assert len(images) == len(cgd_matrices), f"Batch {idx}: image/cgd count mismatch"
            assert images.ndim == 4, f"Batch {idx}: expected 4D images (batch, H, W, channels), got {images.ndim}D"
            assert images.shape[1] == images.shape[2], f"Batch {idx}: images not square"
            
            # Check CGD matrix structure
            sample_cgd = cgd_matrices[0]
            assert sample_cgd.ndim == 2, f"Batch {idx}: expected 2D CGD matrix, got {sample_cgd.ndim}D"
            assert sample_cgd.shape[0] >= 3, f"Batch {idx}: CGD matrix too small {sample_cgd.shape}"
            assert sample_cgd.shape[1] >= 3, f"Batch {idx}: CGD matrix too small {sample_cgd.shape}"
            
        print("Data validation passed!")
        
    def _load_to_memory(self):
        """Load all data to memory for faster access"""
        self.memory_images = []
        self.memory_cgd_matrices = []
        
        for file_idx in range(len(self.image_files)):
            images = np.load(self.image_files[file_idx], allow_pickle=True)
            cgd_matrices = np.load(self.cgd_files[file_idx], allow_pickle=True)
            
            self.memory_images.append(images.astype(np.float32))
            self.memory_cgd_matrices.append(cgd_matrices.astype(np.float32))
            
        print("Data loaded to memory!")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: (1, H, W) tensor
            targets: (3,) tensor containing [Cgd_1_2, Cgd_1_3, Cgd_0_2]
        """
        file_idx, local_idx, channel_idx = self.index_map[idx]
        
        if self.load_to_memory:
            # Load from memory
            full_image = self.memory_images[file_idx][local_idx]  # Shape: (H, W, channels)
            cgd_matrix = self.memory_cgd_matrices[file_idx][local_idx]  # Shape: (num_dots, num_dots+1)
        else:
            # Load from disk
            images = np.load(self.image_files[file_idx], allow_pickle=True)
            cgd_matrices = np.load(self.cgd_files[file_idx], allow_pickle=True)
            
            full_image = images[local_idx].astype(np.float32)  # Shape: (H, W, channels)
            cgd_matrix = cgd_matrices[local_idx].astype(np.float32)  # Shape: (num_dots, num_dots+1)
        
        # Extract single channel from the multi-channel image
        image = full_image[:, :, channel_idx]  # Shape: (H, W)
        
        # Get targets for this specific channel
        num_dots = cgd_matrix.shape[0]
        targets = get_channel_targets(channel_idx, cgd_matrix, num_dots)
        targets = np.array(targets, dtype=np.float32)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension (1, H, W)
        targets = torch.from_numpy(targets)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            
        return image, targets


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    load_to_memory: bool = False,
    transform: Optional[callable] = None,
    random_state: int = 42
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        load_to_memory: Whether to load all data to memory
        transform: Optional image transforms
        random_state: Random seed for train/val split
        
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = CapacitanceDataset(
        data_dir=data_dir,
        transform=transform,
        load_to_memory=load_to_memory
    )
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Image transforms for data augmentation and normalization
def get_transforms(normalize: bool = True):
    """Get image transforms for preprocessing"""
    import torchvision.transforms as transforms
    
    transform_list = []
    
    if normalize:
        # Normalize based on observed range
        # Convert to approximately zero mean, unit variance
        mean = 0.34  # Approximate middle of range
        std = 0.14   # Approximate std based on range
        transform_list.append(transforms.Normalize(mean=[mean], std=[std]))
    
    return transforms.Compose(transform_list) if transform_list else None


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/home/rahul/Summer2025/rl-agent-for-qubit-array-tuning/Swarm/CapacitanceModel/example_dataset"
    
    print("Testing dataset loading...")
    dataset = CapacitanceDataset(data_dir, load_to_memory=False)
    
    # Test first few samples
    for i in range(3):
        image, targets = dataset[i]
        print(f"Sample {i}: image shape {image.shape}, targets {targets}")
    
    print("\nTesting data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir, 
        batch_size=4, 
        load_to_memory=False
    )
    
    # Test one batch
    for batch_imgs, batch_targets in train_loader:
        print(f"Batch: images {batch_imgs.shape}, targets {batch_targets.shape}")
        print(f"Target range: {batch_targets.min():.3f} to {batch_targets.max():.3f}")
        break 