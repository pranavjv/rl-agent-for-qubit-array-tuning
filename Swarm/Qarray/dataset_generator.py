#!/usr/bin/env python3
"""
Quantum Device Dataset Generator

This script generates large datasets of quantum device charge sensor images
with corresponding ground truth parameters for machine learning applications.

Usage:
    python dataset_generator.py --total_samples 10000 --batch_size 1000 --workers 4 --output_dir ./dataset
"""

import os
import sys
import json
import yaml
import argparse
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict

# Add the current directory to path to import qarray components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
except ImportError:
    print("Error: qarray package not found. Please ensure it's installed and available.")
    sys.exit(1)

# Set matplotlib backend to prevent GUI issues in multiprocessing
import matplotlib
matplotlib.use('Agg')


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""
    total_samples: int
    batch_size: int
    workers: int
    output_dir: str
    config_path: str
    voltage_range: Tuple[float, float] = (-2.0, 2.0)
    resolution: int = 128
    seed: int = 42
    resume: bool = True


class ParameterSampler:
    """Handles sampling of quantum device parameters from configuration ranges"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.param_ranges = self._extract_parameter_ranges()
        
    def _extract_parameter_ranges(self) -> Dict[str, Any]:
        """Extract parameter ranges from the configuration"""
        model_config = self.config['simulator']['model']
        measurement_config = self.config['simulator']['measurement']
        
        ranges = {
            'Cdd': model_config['Cdd'],
            'Cgd': model_config['Cgd'],
            'Cds': model_config['Cds'],
            'Cgs': model_config['Cgs'],
            'white_noise_amplitude': model_config['white_noise_amplitude'],
            'telegraph_noise_parameters': model_config['telegraph_noise_parameters'],
            'latching_model_parameters': model_config['latching_model_parameters'],
            'T': model_config['T'],
            'coulomb_peak_width': model_config['coulomb_peak_width'],
            'algorithm': model_config['algorithm'],
            'implementation': model_config['implementation'],
            'max_charge_carriers': model_config['max_charge_carriers'],
            'fixed_gate_voltages': measurement_config['fixed_gate_voltages'],
            'sensor_gate_voltage': measurement_config['sensor_gate_voltage'],
            'swept_gates': measurement_config['swept_gates'],
            'fixed_gates': measurement_config['fixed_gates'],
            'optimal_VG_center': measurement_config['optimal_VG_center']  # Add this for optimal voltage calculation
        }
        
        return ranges
    
    def sample_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Sample random parameters using the provided random generator"""
        rb = self.param_ranges
        
        latching = True
        
        # Generate 4x4 Cdd matrix based on linear array structure
        cdd_diag = rb["Cdd"]["diagonal"]
        cdd_nn = rng.uniform(rb["Cdd"]["nearest_neighbor"]["min"], 
                            rb["Cdd"]["nearest_neighbor"]["max"])
        cdd_next = rng.uniform(rb["Cdd"]["next_nearest"]["min"], 
                              rb["Cdd"]["next_nearest"]["max"])  
        cdd_far = rng.uniform(rb["Cdd"]["furthest"]["min"], 
                             rb["Cdd"]["furthest"]["max"])
        
        Cdd = [
            [cdd_diag, cdd_nn, cdd_next, cdd_far],
            [cdd_nn, cdd_diag, cdd_nn, cdd_next],
            [cdd_next, cdd_nn, cdd_diag, cdd_nn],
            [cdd_far, cdd_next, cdd_nn, cdd_diag]
        ]
        
        # Generate 4x5 Cgd matrix (4 dots x 5 gates) with proper symmetry
        # For plunger gates: Cgd[dot_i][gate_j] should equal Cgd[dot_j][gate_i] (when both exist)
        Cgd = [[0.0 for _ in range(5)] for _ in range(4)]
        
        # First, fill the diagonal (primary couplings)
        for i in range(4):
            Cgd[i][i] = rng.uniform(rb["Cgd"][i][i]["min"], rb["Cgd"][i][i]["max"])
        
        # Fill symmetric cross-couplings for plunger gates (gates 0-3)
        for i in range(4):
            for j in range(4):
                if i < j:  # Only fill upper triangle, then mirror
                    coupling = rng.uniform(rb["Cgd"][i][j]["min"], rb["Cgd"][i][j]["max"])
                    Cgd[i][j] = coupling
                    Cgd[j][i] = coupling  # Ensure symmetry
        
        # Fill sensor gate couplings (gate 4) - these are independent for each dot
        for i in range(4):
            Cgd[i][4] = rng.uniform(rb["Cgd"][i][4]["min"], rb["Cgd"][i][4]["max"])
        
        # Generate arrays for Cds and Cgs
        Cds = [[rng.uniform(rb["Cds"][i]["min"], rb["Cds"][i]["max"]) for i in range(4)]]  # 1x4 (sensor x 4 dots)
        Cgs = [[rng.uniform(rb["Cgs"][i]["min"], rb["Cgs"][i]["max"]) for i in range(5)]]  # 1x5 (sensor x 5 gates)
        
        # Generate 4x4 p_inter matrix for latching model (must be symmetric)
        p_inter = [[0.0 for _ in range(4)] for _ in range(4)]  # Initialize with zeros
        
        # Fill upper triangle and mirror to lower triangle for symmetry
        for i in range(4):
            for j in range(i+1, 4):  # Only fill upper triangle
                coupling = rng.uniform(rb["latching_model_parameters"]["p_inter"]["min"],
                                     rb["latching_model_parameters"]["p_inter"]["max"])
                p_inter[i][j] = coupling
                p_inter[j][i] = coupling  # Ensure symmetry
        # Diagonal elements remain 0.0 (no self-interaction)
        
        # Generate 4-element p_leads array
        p_leads = [rng.uniform(rb["latching_model_parameters"]["p_leads"]["min"],
                              rb["latching_model_parameters"]["p_leads"]["max"]) 
                   for _ in range(4)]
        
        p01 = rng.uniform(rb["telegraph_noise_parameters"]["p01"]["min"], 
                         rb["telegraph_noise_parameters"]["p01"]["max"])
        
        # Generate fixed gate voltages for outer gates (these will be replaced by optimal voltages later)
        fixed_gate_voltages = {}
        for gate_idx in rb["fixed_gates"]:
            if gate_idx == 4:  # Sensor gate (will be replaced with optimal)
                fixed_gate_voltages[gate_idx] = rng.uniform(
                    rb["sensor_gate_voltage"]["min"], 
                    rb["sensor_gate_voltage"]["max"]
                )
            else:  # Plunger gates (will be replaced with optimal)
                fixed_gate_voltages[gate_idx] = rng.uniform(
                    rb["fixed_gate_voltages"]["min"], 
                    rb["fixed_gate_voltages"]["max"]
                )
        
        model_params = {
            "Cdd": Cdd,
            "Cgd": Cgd,
            "Cds": Cds,
            "Cgs": Cgs,
            "white_noise_amplitude": rng.uniform(rb["white_noise_amplitude"]["min"], 
                                               rb["white_noise_amplitude"]["max"]),
            "telegraph_noise_parameters": {
                "p01": p01,
                "p10": rng.uniform(rb["telegraph_noise_parameters"]["p10_factor"]["min"], 
                                 rb["telegraph_noise_parameters"]["p10_factor"]["max"]) * p01,
                "amplitude": rng.uniform(rb["telegraph_noise_parameters"]["amplitude"]["min"], 
                                       rb["telegraph_noise_parameters"]["amplitude"]["max"]),
            },
            "latching_model_parameters": {
                "Exists": latching,
                "n_dots": 4,
                "p_leads": p_leads,
                "p_inter": p_inter,
            },
            "T": rng.uniform(rb["T"]["min"], rb["T"]["max"]),
            "coulomb_peak_width": rng.uniform(rb["coulomb_peak_width"]["min"], 
                                            rb["coulomb_peak_width"]["max"]),
            "algorithm": rb["algorithm"],
            "implementation": rb["implementation"],
            "max_charge_carriers": rb["max_charge_carriers"],
            "fixed_gate_voltages": fixed_gate_voltages,
            "swept_gates": rb["swept_gates"],
            "fixed_gates": rb["fixed_gates"],
            "optimal_VG_center": rb["optimal_VG_center"]
        }
        
        return model_params


class ModelBuilder:
    """Builds ChargeSensedDotArray models from parameters"""
    
    @staticmethod
    def build_model(model_params: Dict[str, Any]) -> ChargeSensedDotArray:
        """Build a ChargeSensedDotArray model from parameters"""
        # Create noise models
        white_noise = WhiteNoise(amplitude=model_params['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**model_params['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        
        # Create latching model if specified
        latching_params = model_params['latching_model_parameters']
        latching_model = LatchingModel(
            **{k: v for k, v in latching_params.items() if k != "Exists"}
        ) if latching_params["Exists"] else None
        
        # Create the main model
        model = ChargeSensedDotArray(
            Cdd=model_params['Cdd'],
            Cgd=model_params['Cgd'],
            Cds=model_params['Cds'],
            Cgs=model_params['Cgs'],
            coulomb_peak_width=model_params['coulomb_peak_width'],
            T=model_params['T'],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=model_params['algorithm'],
            implementation=model_params['implementation'],
            max_charge_carriers=model_params['max_charge_carriers'],
        )
        
        # Set 5x5 virtual gate matrix for 5-gate system (4 plunger + 1 sensor)
        model.gate_voltage_composer.virtual_gate_matrix = np.array([
            [1, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0], 
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        
        # Calculate optimal voltages for target charge occupancy and update fixed gate voltages
        optimal_voltages = model.optimal_Vg(model_params['optimal_VG_center'])
        
        # Update fixed gate voltages with optimal values
        updated_fixed_voltages = {}
        for gate_idx in model_params['fixed_gates']:
            updated_fixed_voltages[gate_idx] = optimal_voltages[gate_idx]
        
        # Update the model_params with optimal voltages
        model_params['fixed_gate_voltages'] = updated_fixed_voltages
        return model


class ImageGenerator:
    """Generates charge sensor images from quantum device models"""
    
    def __init__(self, voltage_range: Tuple[float, float], resolution: int):
        self.voltage_range = voltage_range
        self.resolution = resolution
        self.v_min, self.v_max = voltage_range
    
    def generate_voltage_grid(self, center: Tuple[float, float], 
                            rng: np.random.Generator, 
                            fixed_gate_voltages: Dict[int, float]) -> np.ndarray:
        """Generate a voltage grid for 5-gate system with central two-gate sweep"""
        # Create voltage grid for 5-gate system (4 plunger + 1 sensor)
        # Only sweep central gates (1 and 2), fix outer gates (0, 3) and sensor gate (4)
        x_center, y_center = center
        
        # Create 2D voltage grid for central gates only
        x_voltages = np.linspace(self.v_min, self.v_max, self.resolution)
        y_voltages = np.linspace(self.v_min, self.v_max, self.resolution)
        
        X, Y = np.meshgrid(x_voltages + x_center, y_voltages + y_center)
        
        # Create 5D voltage array (height, width, 5 gates)
        voltage_grid = np.zeros((self.resolution, self.resolution, 5))
        
        # Fixed voltage for gate 0 (leftmost plunger)
        voltage_grid[:, :, 0] = fixed_gate_voltages[0]
        
        # Swept voltages for central gates 1 and 2
        voltage_grid[:, :, 1] = X  # Central gate 1
        voltage_grid[:, :, 2] = Y  # Central gate 2
        
        # Fixed voltage for gate 3 (rightmost plunger)  
        voltage_grid[:, :, 3] = fixed_gate_voltages[3]
        
        # Fixed voltage for gate 4 (sensor gate)
        voltage_grid[:, :, 4] = fixed_gate_voltages[4]
        
        return voltage_grid
    
    def generate_image(self, model: ChargeSensedDotArray, 
                      voltage_grid: np.ndarray) -> np.ndarray:
        """Generate charge sensor image from model and voltage grid"""
        try:
            z, _ = model.charge_sensor_open(voltage_grid)
            # Return the first channel only
            return z[:, :, 0]
        except Exception as e:
            # If simulation fails, return a zero image
            logging.warning(f"Image generation failed: {e}")
            return np.zeros((self.resolution, self.resolution))


class BatchProcessor:
    """Handles batch processing and saving of generated data"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.parameters_dir = self.output_dir / 'parameters'
        self.metadata_dir = self.output_dir / 'metadata'
        
        # Create directories
        for dir_path in [self.images_dir, self.parameters_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_batch(self, batch_id: int, images: np.ndarray, 
                   parameters: List[Dict[str, Any]]) -> bool:
        """Save a batch of images and parameters"""
        try:
            # Save images
            image_path = self.images_dir / f'batch_{batch_id:06d}.npy'
            np.save(image_path, images)
            
            # Save parameters
            param_path = self.parameters_dir / f'batch_{batch_id:06d}.npy'
            np.save(param_path, parameters)
            
            return True
        except Exception as e:
            logging.error(f"Failed to save batch {batch_id}: {e}")
            return False
    
    def load_progress(self) -> Dict[str, Any]:
        """Load generation progress from file"""
        progress_path = self.metadata_dir / 'progress.json'
        if progress_path.exists():
            try:
                with open(progress_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load progress: {e}")
        
        return {
            'total_samples': 0,
            'completed_samples': 0,
            'completed_batches': 0,
            'batch_size': 0,
            'last_batch_timestamp': None,
            'failed_batches': []
        }
    
    def save_progress(self, progress: Dict[str, Any]) -> None:
        """Save generation progress to file"""
        progress_path = self.metadata_dir / 'progress.json'
        progress['last_batch_timestamp'] = datetime.now().isoformat()
        
        try:
            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save progress: {e}")


def generate_single_sample(model_params: Dict[str, Any], 
                          voltage_config: Dict[str, Any],
                          sample_seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a single image and its ground truth parameters"""
    try:
        # Set up random generator for this sample
        rng = np.random.default_rng(sample_seed)
        
        # Build model
        model = ModelBuilder.build_model(model_params)
        
        # Generate image
        image_gen = ImageGenerator(
            voltage_range=(voltage_config['v_min'], voltage_config['v_max']),
            resolution=voltage_config['resolution']
        )
        
        # Random center for voltage sweep
        center_x = rng.uniform(-1.0, 1.0)  # Adjust range as needed
        center_y = rng.uniform(-1.0, 1.0)
        
        voltage_grid = image_gen.generate_voltage_grid(
            (center_x, center_y), rng, model_params['fixed_gate_voltages']
        )
        image = image_gen.generate_image(model, voltage_grid)
        
        # Prepare ground truth data
        ground_truth = {
            'model_params': model_params,
            'voltage_center': [center_x, center_y],  # Centers for swept gates 1,2
            'fixed_gate_voltages': model_params['fixed_gate_voltages'],
            'swept_gates': model_params['swept_gates'],
            'fixed_gates': model_params['fixed_gates'],
            'voltage_range': voltage_config,
            'seed': sample_seed
        }
        
        return image, ground_truth
        
    except Exception as e:
        logging.error(f"Failed to generate sample with seed {sample_seed}: {e}")
        # Return zero image and empty parameters on failure
        resolution = voltage_config.get('resolution', 128)
        return np.zeros((resolution, resolution)), {}


def worker_process(work_queue: mp.Queue, result_queue: mp.Queue, 
                  config: Dict[str, Any], worker_id: int) -> None:
    """Worker process for parallel sample generation"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f'Worker-{worker_id}')
    
    try:
        param_sampler = ParameterSampler(config['qarray_config'])
        voltage_config = config['voltage_config']
        logger.info(f"Worker {worker_id} initialized successfully")
    except Exception as e:
        logger.error(f"Worker {worker_id} initialization failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    
    samples_processed = 0
    
    while True:
        try:
            # Get work from queue
            work_item = work_queue.get(timeout=5.0)
            if work_item is None:  # Shutdown signal
                break
                
            batch_start, batch_size, base_seed = work_item
            
            # Generate batch of samples
            batch_images = []
            batch_params = []
            
            logger.info(f"Worker {worker_id} starting batch of {batch_size} samples")
            batch_start_time = time.time()
            
            for i in range(batch_size):
                sample_seed = base_seed + i
                rng = np.random.default_rng(sample_seed)
                
                # Sample parameters
                model_params = param_sampler.sample_parameters(rng)
                
                # Generate sample
                image, ground_truth = generate_single_sample(
                    model_params, voltage_config, sample_seed
                )
                
                batch_images.append(image)
                batch_params.append(ground_truth)
                samples_processed += 1
                
                # Progress reporting for large batches
                if batch_size >= 1000 and (i + 1) % max(1, batch_size // 10) == 0:
                    elapsed = time.time() - batch_start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (batch_size - i - 1) / rate if rate > 0 else 0
                    logger.info(f"Worker {worker_id}: {i+1}/{batch_size} samples ({rate:.1f}/sec, ETA: {eta/60:.1f}min)")
            
            # Convert to numpy arrays and free memory
            batch_images = np.array(batch_images, dtype=np.float32)  # Use float32 to save memory
            
            batch_time = time.time() - batch_start_time
            memory_mb = batch_images.nbytes / (1024 * 1024)
            batch_rate = batch_size / batch_time if batch_time > 0 else 0
            logger.info(f"Worker {worker_id} completed batch: {batch_size} samples in {batch_time/60:.1f}min ({batch_rate:.1f} samples/sec, {memory_mb:.1f}MB)")
            
            # Send results back
            result_queue.put((batch_start // batch_size, batch_images, batch_params))
            
            # Clear memory
            del batch_images
            del batch_params
            
            logger.info(f"Completed batch starting at {batch_start} ({samples_processed} total)")
            
        except Exception as e:
            if "timeout" not in str(e).lower():
                logger.error(f"Worker {worker_id} error: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            break
    
    logger.info(f"Worker {worker_id} processed {samples_processed} samples")


class DatasetGenerator:
    """Main dataset generator coordinating the entire process"""
    
    def __init__(self, gen_config: GenerationConfig):
        self.config = gen_config
        self.setup_logging()
        self.load_qarray_config()
        
        # Resolution is already set correctly from main() function
        self.logger.info(f"Using resolution: {self.config.resolution}")
        
        self.batch_processor = BatchProcessor(gen_config.output_dir)
        
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = Path(self.config.output_dir) / 'metadata'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'generation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_qarray_config(self) -> None:
        """Load qarray configuration from YAML file"""
        config_path = Path(self.config.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.qarray_config = yaml.safe_load(f)
    
    def save_dataset_metadata(self) -> None:
        """Save dataset metadata and configuration"""
        metadata = {
            'generation_config': asdict(self.config),
            'qarray_config': self.qarray_config,
            'generation_timestamp': datetime.now().isoformat(),
            'total_samples': self.config.total_samples,
            'batch_size': self.config.batch_size,
            'image_shape': [self.config.resolution, self.config.resolution],
            'voltage_range': self.config.voltage_range
        }
        
        metadata_path = Path(self.config.output_dir) / 'metadata' / 'dataset_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a copy of the config used
        config_copy_path = Path(self.config.output_dir) / 'metadata' / 'config_used.yaml'
        with open(config_copy_path, 'w') as f:
            yaml.dump(self.qarray_config, f)
    
    def generate_dataset(self) -> None:
        """Generate the complete dataset"""
        self.logger.info("Starting dataset generation...")
        self.logger.info(f"Configuration: {self.config}")
        
        # Check for suboptimal batch sizes and warn
        if self.config.batch_size > 5000:
            self.logger.warning(f"Large batch size ({self.config.batch_size}) may cause memory issues and timeouts")
            self.logger.warning("Recommended batch sizes: 500-2000 for most systems")
        
        # Estimate memory usage
        image_size_mb = (self.config.resolution ** 2) * 8 / (1024 * 1024)  # float64
        batch_memory_mb = image_size_mb * self.config.batch_size
        self.logger.info(f"Estimated memory per batch: {batch_memory_mb:.1f} MB")
        
        if batch_memory_mb > 2000:  # > 2GB
            self.logger.warning("High memory usage expected - consider reducing batch_size")
        
        # Save metadata
        self.save_dataset_metadata()
        
        # Load or initialize progress
        progress = self.batch_processor.load_progress()
        
        if self.config.resume and progress['completed_samples'] > 0:
            self.logger.info(f"Resuming from {progress['completed_samples']} completed samples")
            start_sample = progress['completed_samples']
            # Keep existing progress structure but ensure all fields exist
            progress.setdefault('total_samples', self.config.total_samples)
            progress.setdefault('completed_batches', 0)
            progress.setdefault('batch_size', self.config.batch_size)
            progress.setdefault('last_batch_timestamp', None)
            progress.setdefault('failed_batches', [])
        else:
            start_sample = 0
            progress = {
                'total_samples': self.config.total_samples,
                'completed_samples': 0,
                'completed_batches': 0,
                'batch_size': self.config.batch_size,
                'last_batch_timestamp': None,
                'failed_batches': []
            }
        
        # Calculate batches to process
        remaining_samples = self.config.total_samples - start_sample
        total_batches = (remaining_samples + self.config.batch_size - 1) // self.config.batch_size
        
        self.logger.info(f"Processing {total_batches} batches with {self.config.workers} workers")
        if start_sample > 0:
            self.logger.info(f"Starting from sample {start_sample}, generating {remaining_samples} new samples")
        
        # Setup multiprocessing
        work_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Create worker config
        worker_config = {
            'qarray_config': self.qarray_config,
            'voltage_config': {
                'v_min': self.config.voltage_range[0],
                'v_max': self.config.voltage_range[1],
                'resolution': self.config.resolution
            }
        }
        
        # Start worker processes
        workers = []
        for i in range(self.config.workers):
            worker = mp.Process(
                target=worker_process,
                args=(work_queue, result_queue, worker_config, i)
            )
            worker.start()
            workers.append(worker)
        
        # Submit work
        for batch_idx in range(total_batches):
            batch_start = start_sample + batch_idx * self.config.batch_size
            batch_size = min(self.config.batch_size, 
                           self.config.total_samples - batch_start)
            base_seed = self.config.seed + batch_start
            
            work_queue.put((batch_start, batch_size, base_seed))
        
        # Add shutdown signals
        for _ in range(self.config.workers):
            work_queue.put(None)
        
        # Collect results - track session vs overall progress separately
        completed_batches = 0  # Batches completed in this session
        session_start_time = time.time()
        session_samples_generated = 0  # Samples generated in this session only
        
        # Dynamic timeout based on batch size (at least 5 minutes, up to 30 minutes for large batches)
        base_timeout = 300  # 5 minutes base
        timeout_per_sample = 2.0  # 2 seconds per sample
        dynamic_timeout = max(base_timeout, self.config.batch_size * timeout_per_sample)
        self.logger.info(f"Using timeout of {dynamic_timeout/60:.1f} minutes per batch")
        
        for _ in range(total_batches):
            try:
                batch_id, batch_images, batch_params = result_queue.get(timeout=dynamic_timeout)
                
                # Save batch
                if self.batch_processor.save_batch(batch_id, batch_images, batch_params):
                    completed_batches += 1
                    
                    # Calculate actual samples in this batch (handle partial last batch)
                    actual_batch_samples = batch_images.shape[0]
                    session_samples_generated += actual_batch_samples
                    
                    # Calculate overall progress (total samples including previous sessions)
                    total_samples_completed = start_sample + session_samples_generated
                    
                    # Update progress file
                    progress['completed_samples'] = min(total_samples_completed, self.config.total_samples)
                    progress['completed_batches'] = progress.get('completed_batches', 0) + 1  # Increment by 1 for this batch
                    self.batch_processor.save_progress(progress)
                    
                    # Calculate session-specific rates (only for this session)
                    session_elapsed = time.time() - session_start_time
                    session_samples_per_sec = session_samples_generated / session_elapsed if session_elapsed > 0 else 0
                    
                    # Calculate ETA based on session rate and remaining samples
                    remaining_samples = self.config.total_samples - total_samples_completed
                    eta_seconds = remaining_samples / session_samples_per_sec if session_samples_per_sec > 0 else 0
                    
                    # Log progress with both session and overall context
                    progress_msg = (
                        f"Progress: {total_samples_completed}/{self.config.total_samples} samples "
                        f"({total_samples_completed/self.config.total_samples*100:.1f}%) | "
                        f"Session: {session_samples_generated} samples, {session_samples_per_sec:.1f}/sec | "
                        f"ETA: {eta_seconds/60:.1f} min"
                    )
                    
                    # Add batch info every 5 batches or for large batches
                    if completed_batches % 5 == 0 or self.config.batch_size >= 1000:
                        progress_msg += f" | Batch {completed_batches}/{total_batches} ({actual_batch_samples} samples)"
                    
                    self.logger.info(progress_msg)
                else:
                    progress['failed_batches'].append(batch_id)
                    
            except Exception as e:
                self.logger.error(f"Failed to process result: {e}")
        
        # Cleanup workers
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        
        self.logger.info("Dataset generation completed!")
        
        # Final statistics - session-specific only
        session_total_time = time.time() - session_start_time
        session_avg_rate = session_samples_generated / session_total_time if session_total_time > 0 else 0
        
        self.logger.info(f"Session summary:")
        self.logger.info(f"  Generated {session_samples_generated} samples in {session_total_time/60:.1f} minutes")
        self.logger.info(f"  Session average rate: {session_avg_rate:.1f} samples/second")
        
        # Overall statistics
        final_total = start_sample + session_samples_generated
        self.logger.info(f"Overall dataset now contains {final_total} samples total")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate quantum device dataset')
    parser.add_argument('--total_samples', type=int, default=10000,
                       help='Total number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of samples per batch')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(),
                       help='Number of worker processes')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                       help='Output directory for dataset')
    parser.add_argument('--config_path', type=str, default='qarray_config.yaml',
                       help='Path to qarray configuration file')
    parser.add_argument('--voltage_range', nargs=2, type=float, default=[-2.0, 2.0],
                       help='Voltage range for measurements')
    parser.add_argument('--resolution', type=int, default=None,
                       help='Image resolution (pixels). If not specified, uses value from config file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_resume', action='store_true',
                       help='Start fresh (do not resume from existing progress)')
    
    args = parser.parse_args()
    
    # Load config first to get resolution if not specified
    if args.resolution is None:
        try:
            import yaml
            with open(args.config_path, 'r') as f:
                config = yaml.safe_load(f)
            config_resolution = config.get('simulator', {}).get('measurement', {}).get('resolution', 128)
            print(f"Using resolution from {args.config_path}: {config_resolution}")
        except Exception as e:
            config_resolution = 128
            print(f"Could not load resolution from config, using default: {config_resolution}")
    else:
        config_resolution = args.resolution
        print(f"Using command-line specified resolution: {config_resolution}")
    
    # Create generation config
    gen_config = GenerationConfig(
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        output_dir=args.output_dir,
        config_path=args.config_path,
        voltage_range=tuple(args.voltage_range),
        resolution=config_resolution,
        seed=args.seed,
        resume=not args.no_resume
    )
    
    # Generate dataset
    try:
        generator = DatasetGenerator(gen_config)
        generator.generate_dataset()
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Generation failed: {e}")
        raise


if __name__ == '__main__':
    main() 