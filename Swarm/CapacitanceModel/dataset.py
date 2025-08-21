import torch
import torch.utils.data as data
import numpy as np
import os
import glob
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


class CapacitanceDataset(data.Dataset):
    """
    PyTorch dataset for loading capacitance images and Cgd targets.
    
    Loads data from batched NPY files containing images and parameters.
    Extracts Cgd[1,2], Cgd[1,3], and Cgd[0,2] as targets.
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
        self.param_files = sorted(glob.glob(os.path.join(data_dir, "parameters", "batch_*.npy")))
        
        assert len(self.image_files) == len(self.param_files), \
            f"Mismatch: {len(self.image_files)} image files vs {len(self.param_files)} param files"
        
        print(f"Found {len(self.image_files)} batch files")
        
        # Build index mapping global_idx -> (file_idx, local_idx)
        self._build_index()
        
        if validate_data:
            self._validate_data()
        
        if load_to_memory:
            print("Loading all data to memory...")
            self._load_to_memory()
        
    def _build_index(self):
        """Build mapping from global index to (batch_file_idx, local_idx)"""
        self.index_map = []
        self.batch_sizes = []
        
        for file_idx, (img_file, param_file) in enumerate(zip(self.image_files, self.param_files)):
            # Load to get batch size
            params = np.load(param_file, allow_pickle=True)
            batch_size = len(params)
            self.batch_sizes.append(batch_size)
            
            for local_idx in range(batch_size):
                self.index_map.append((file_idx, local_idx))
        
        self.total_samples = len(self.index_map)
        print(f"Total samples: {self.total_samples}")
        
    def _validate_data(self):
        """Validate data consistency across a few sample batches"""
        print("Validating data consistency...")
        
        # Check first, middle, and last batch
        check_indices = [0, len(self.image_files) // 2, len(self.image_files) - 1]
        
        for idx in check_indices:
            img_file = self.image_files[idx]
            param_file = self.param_files[idx]
            
            images = np.load(img_file, allow_pickle=True)
            params = np.load(param_file, allow_pickle=True)
            
            # Check shapes and structure
            assert len(images) == len(params), f"Batch {idx}: image/param count mismatch"
            assert images.ndim == 3, f"Batch {idx}: expected 3D images, got {images.ndim}D"
            assert images.shape[1] == images.shape[2], f"Batch {idx}: images not square"
            
            # Check Cgd structure in first sample
            sample_params = params[0]
            assert 'model_params' in sample_params, f"Batch {idx}: missing model_params"
            assert 'Cgd' in sample_params['model_params'], f"Batch {idx}: missing Cgd"
            
            cgd = np.array(sample_params['model_params']['Cgd'])
            assert cgd.shape[0] >= 4 and cgd.shape[1] >= 4, f"Batch {idx}: Cgd too small {cgd.shape}"
            
        print("Data validation passed!")
        
    def _load_to_memory(self):
        """Load all data to memory for faster access"""
        self.memory_images = []
        self.memory_targets = []
        
        for file_idx in range(len(self.image_files)):
            images = np.load(self.image_files[file_idx], allow_pickle=True)
            params = np.load(self.param_files[file_idx], allow_pickle=True)
            
            # Process targets for this batch
            batch_targets = []
            for param in params:
                cgd = np.array(param['model_params']['Cgd'])
                targets = [cgd[1,2], cgd[1,3], cgd[0,2]]  # Cgd 1,2, Cgd 1,3, Cgd 0,2
                batch_targets.append(targets)
            
            self.memory_images.append(images.astype(np.float32))
            self.memory_targets.append(np.array(batch_targets, dtype=np.float32))
            
        print("Data loaded to memory!")
    
    def _extract_targets(self, param_dict) -> List[float]:
        """Extract Cgd[1,2], Cgd[1,3], Cgd[0,2] from parameter dictionary"""
        cgd = np.array(param_dict['model_params']['Cgd'])
        return [cgd[1,2], cgd[1,3], cgd[0,2]]
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: (1, H, W) tensor
            targets: (3,) tensor containing [Cgd_1_2, Cgd_1_3, Cgd_0_2]
        """
        file_idx, local_idx = self.index_map[idx]
        
        if self.load_to_memory:
            # Load from memory
            image = self.memory_images[file_idx][local_idx]
            targets = self.memory_targets[file_idx][local_idx]
        else:
            # Load from disk
            images = np.load(self.image_files[file_idx], allow_pickle=True)
            params = np.load(self.param_files[file_idx], allow_pickle=True)
            
            image = images[local_idx].astype(np.float32)
            targets = self._extract_targets(params[local_idx])
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
        # Normalize based on observed range [0.25, 1.14]
        # Convert to approximately zero mean, unit variance
        mean = 0.695  # Approximate middle of range
        std = 0.297   # Approximate std based on range
        transform_list.append(transforms.Normalize(mean=[mean], std=[std]))
    
    return transforms.Compose(transform_list) if transform_list else None


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/home/rahul/Summer2025/rl-agent-for-qubit-array-tuning/Swarm/Qarray/example_dataset"
    
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