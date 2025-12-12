"""
=====================================
File: config.py
=====================================
Configuration and constants for ABIT Transform Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class TransformConfig:
    """
    Configuration for transform analysis.
    Centralized configuration makes experiments reproducible and manageable.
    """
    
    # Known patterns in the data (ground truth)
    known_frequencies: List[float] = field(default_factory=lambda: [1/12, 1/6, 1/3])
    known_periods: List[float] = field(default_factory=lambda: [12.0, 6.0, 3.0])
    
    # HHT Parameters
    hht_max_imfs: int = 8
    hht_sift_iterations: int = 1000
    hht_stopping_criterion: float = 0.001
    hht_envelope_method: str = 'savgol'  # 'savgol' or 'spline'
    
    # Wavelet Parameters
    wavelet_type: str = 'morl'  # Morlet wavelet for business cycles
    wavelet_scales: Optional[np.ndarray] = None
    wavelet_max_scale: int = 64
    wavelet_min_scale: int = 1
    wavelet_scale_step: int = 1
    dwt_wavelet: str = 'db4'  # Daubechies-4 for DWT
    dwt_level: int = 5
    
    # STFT Parameters
    stft_window: str = 'hann'
    stft_nperseg: int = 24  # Default window size (2 years for monthly data)
    stft_noverlap: Optional[int] = None  # Will be set to 75% of nperseg if None
    stft_fs: float = 1.0  # Sampling frequency (1 = per month)
    stft_adaptive_window: bool = True
    
    # Analysis Parameters
    minority_amplification: float = 10.0
    pattern_confidence_threshold: float = 0.5
    frequency_tolerance: float = 0.02  # Tolerance for frequency matching
    period_tolerance: float = 1.0  # Tolerance for period matching (months)
    min_pattern_energy: float = 0.01  # Minimum energy to consider a pattern
    
    # Cascade Parameters
    cascade_validation_threshold: float = 0.5
    cascade_combination_method: str = 'average'  # 'average', 'weighted', 'max'
    enable_cross_validation: bool = True
    
    # Evaluation Parameters
    detection_weight: float = 0.6
    noise_rejection_weight: float = 0.4
    confidence_weight: float = 0.5
    amplitude_weight: float = 0.5
    
    # Business Relevance Parameters
    business_periods: List[float] = field(default_factory=lambda: [12.0, 6.0, 3.0, 1.0])
    business_period_names: Dict[float, str] = field(default_factory=lambda: {
        12.0: 'Annual',
        6.0: 'Semi-Annual',
        3.0: 'Quarterly',
        1.0: 'Monthly'
    })
    business_relevance_boost: float = 1.5  # Boost for business-relevant patterns
    
    # Visualization Parameters
    figure_size: Tuple[int, int] = (16, 10)
    color_scheme: str = 'husl'
    save_figures: bool = False
    figure_dpi: int = 150
    
    # Performance Parameters
    enable_caching: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    verbose: bool = True
    
    # Data Parameters
    signal_normalization: bool = True
    detrend_before_analysis: bool = False
    handle_missing_values: str = 'interpolate'  # 'interpolate', 'drop', 'fill'
    
    def __post_init__(self):
        """Initialize computed fields and validate configuration"""
        
        # Set default wavelet scales if not provided
        if self.wavelet_scales is None:
            self.wavelet_scales = np.arange(
                self.wavelet_min_scale, 
                self.wavelet_max_scale, 
                self.wavelet_scale_step
            )
        
        # Set STFT overlap to 75% if not specified
        if self.stft_noverlap is None:
            self.stft_noverlap = int(self.stft_nperseg * 0.75)
        
        # Validate weights sum to reasonable values
        total_eval_weight = self.detection_weight + self.noise_rejection_weight
        if abs(total_eval_weight - 1.0) > 0.01:
            # Normalize weights
            self.detection_weight /= total_eval_weight
            self.noise_rejection_weight /= total_eval_weight
        
        # Ensure known periods and frequencies match
        if self.known_frequencies and not self.known_periods:
            self.known_periods = [1/f for f in self.known_frequencies]
        elif self.known_periods and not self.known_frequencies:
            self.known_frequencies = [1/p for p in self.known_periods]
    
    def get_config_summary(self) -> str:
        """Return a summary of current configuration"""
        summary = "ABIT Transform Configuration\n"
        summary += "="*50 + "\n"
        summary += f"Known Patterns: {len(self.known_frequencies)} frequencies\n"
        summary += f"  Periods: {self.known_periods} months\n"
        summary += f"\nTransform Settings:\n"
        summary += f"  HHT: {self.hht_max_imfs} max IMFs\n"
        summary += f"  Wavelet: {self.wavelet_type}, {len(self.wavelet_scales)} scales\n"
        summary += f"  STFT: {self.stft_window} window, size {self.stft_nperseg}\n"
        summary += f"\nAnalysis Settings:\n"
        summary += f"  Minority Amplification: {self.minority_amplification}x\n"
        summary += f"  Confidence Threshold: {self.pattern_confidence_threshold}\n"
        summary += f"  Frequency Tolerance: {self.frequency_tolerance}\n"
        summary += f"\nPerformance:\n"
        summary += f"  Caching: {'Enabled' if self.enable_caching else 'Disabled'}\n"
        summary += f"  Parallel: {'Enabled' if self.parallel_processing else 'Disabled'}\n"
        return summary
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TransformConfig':
        """Create configuration from dictionary"""
        # Convert lists back to numpy arrays where needed
        if 'wavelet_scales' in config_dict and config_dict['wavelet_scales'] is not None:
            config_dict['wavelet_scales'] = np.array(config_dict['wavelet_scales'])
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TransformConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Create default configuration instance
DEFAULT_CONFIG = TransformConfig()


# Utility functions for configuration management
def create_config(**kwargs) -> TransformConfig:
    """
    Create a configuration with custom parameters.
    
    Example:
        config = create_config(minority_amplification=15.0, hht_max_imfs=10)
    """
    return TransformConfig(**kwargs)


def get_predefined_config(config_type: str) -> TransformConfig:
    """
    Get a predefined configuration for common scenarios.
    
    Args:
        config_type: One of 'default', 'real_time', 'minority', 'high_accuracy', 'business'
    
    Returns:
        TransformConfig instance
    """
    configs = {
        'default': TransformConfig(),
        
        'real_time': TransformConfig(
            hht_max_imfs=5,
            wavelet_max_scale=32,
            stft_nperseg=12,
            pattern_confidence_threshold=0.6,
            enable_caching=True,
            verbose=False
        ),
        
        'minority': TransformConfig(
            minority_amplification=20.0,
            pattern_confidence_threshold=0.3,
            frequency_tolerance=0.03,
            min_pattern_energy=0.005,
            hht_max_imfs=10,
            detection_weight=0.7,
            noise_rejection_weight=0.3
        ),
        
        'high_accuracy': TransformConfig(
            hht_max_imfs=12,
            hht_sift_iterations=2000,
            wavelet_max_scale=128,
            stft_nperseg=36,
            pattern_confidence_threshold=0.7,
            frequency_tolerance=0.01,
            cascade_validation_threshold=0.6
        ),
        
        'business': TransformConfig(
            known_frequencies=[1/12, 1/6, 1/3, 1/1],
            business_relevance_boost=2.0,
            pattern_confidence_threshold=0.4,
            wavelet_scales=np.array([1, 3, 6, 12, 24, 36, 48]),
            stft_nperseg=24
        )
    }
    
    if config_type not in configs:
        raise ValueError(f"Unknown config type: {config_type}. Choose from: {list(configs.keys())}")
    
    return configs[config_type]


# Export all necessary items
__all__ = [
    'TransformConfig',
    'DEFAULT_CONFIG',
    'create_config',
    'get_predefined_config'
]