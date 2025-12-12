
import numpy as np
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Pattern, TransformCascade
from transforms import HHTTransform, WaveletTransform, STFTTransform
from config import TransformConfig

class SingleTransformCascade(TransformCascade):
    """
    Cascade with only a single transform (baseline for comparison).
    This is essentially a wrapper to make single transforms compatible
    with the cascade interface.
    """
    
    def __init__(self, transform_type: str, config: TransformConfig = None):
        """
        Initialize with a single transform.
        
        Args:
            transform_type: 'hht', 'wavelet', or 'stft'
            config: Configuration object
        """
        # Map string to transform class
        transform_map = {
            'hht': HHTTransform,
            'wavelet': WaveletTransform,
            'stft': STFTTransform
        }
        
        if transform_type.lower() not in transform_map:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Create transform instance
        transform = transform_map[transform_type.lower()](config)
        
        # Initialize parent with single transform
        super().__init__([transform], config)
        
        self.transform_type = transform_type.lower()
    
    def analyze(self, signal: np.ndarray, signal_type: str = 'unknown') -> List[Pattern]:
        """
        Run single transform analysis.
        
        Args:
            signal: Input time series
            signal_type: Type of signal ('majority', 'minority', 'unknown')
            
        Returns:
            List of discovered patterns
        """
        self.log_execution(f"Starting single transform: {self.transform_type}")
        
        # Run the single transform
        if self.transform_type == 'hht':
            # HHT needs signal_type for minority amplification
            patterns, artifacts = self.transforms[0].analyze(signal, signal_type=signal_type)
        else:
            # Other transforms don't use signal_type in the same way
            patterns, artifacts = self.transforms[0].analyze(signal)
        
        self.log_execution(f"Transform complete", {
            'patterns_found': len(patterns),
            'signal_length': len(signal),
            'signal_type': signal_type
        })
        
        # Apply validation
        validated_patterns = self.validate_patterns(patterns)
        
        self.log_execution(f"Validation complete", {
            'patterns_validated': len(validated_patterns),
            'patterns_rejected': len(patterns) - len(validated_patterns)
        })
        
        return validated_patterns
    
    def get_name(self) -> str:
        """Get cascade name"""
        return self.transform_type.upper()
