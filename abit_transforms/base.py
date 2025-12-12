
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
import numpy as np
from datetime import datetime
import json

# Forward declaration to avoid circular import
if TYPE_CHECKING:
    from config import TransformConfig

@dataclass
class Pattern:
    """
    Represents a discovered pattern in the signal.
    This is the fundamental data structure that all transforms produce.
    """
    frequency: float  # Frequency in cycles per time unit
    period: float  # Period in time units (typically months)
    amplitude: float = 0.0  # Pattern amplitude/strength
    confidence: float = 0.0  # Confidence score (0-1)
    source_method: str = ""  # Which transform/cascade discovered this
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional information
    
    def __post_init__(self):
        """Validate and compute derived fields"""
        # Ensure metadata is a dict
        if self.metadata is None:
            self.metadata = {}
        
        # Compute period from frequency if needed
        if self.frequency > 0 and self.period == 0:
            self.period = 1 / self.frequency
        elif self.period > 0 and self.frequency == 0:
            self.frequency = 1 / self.period
        
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def matches_frequency(self, target_freq: float, tolerance: float = 0.02) -> bool:
        """Check if pattern matches target frequency within tolerance"""
        if self.frequency == 0 or target_freq == 0:
            return self.frequency == target_freq
        return abs(self.frequency - target_freq) < tolerance
    
    def matches_period(self, target_period: float, tolerance: float = 1.0) -> bool:
        """Check if pattern matches target period within tolerance"""
        if self.period == np.inf or target_period == np.inf:
            return self.period == target_period
        return abs(self.period - target_period) < tolerance
    
    def is_business_relevant(self, business_periods: List[float] = None) -> bool:
        """Check if pattern matches standard business periods"""
        if business_periods is None:
            business_periods = [12.0, 6.0, 3.0, 1.0]  # Annual, semi, quarterly, monthly
        
        for bp in business_periods:
            if self.matches_period(bp, tolerance=1.0):
                return True
        return False
    
    def to_dict(self) -> Dict:
        """Convert pattern to dictionary for serialization"""
        return {
            'frequency': float(self.frequency),
            'period': float(self.period),
            'amplitude': float(self.amplitude),
            'confidence': float(self.confidence),
            'source_method': self.source_method,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Pattern':
        """Create pattern from dictionary"""
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation"""
        if self.period < np.inf:
            return f"Pattern(period={self.period:.1f}, confidence={self.confidence:.2f}, source={self.source_method})"
        else:
            return f"Pattern(trend, confidence={self.confidence:.2f}, source={self.source_method})"
    
    def get_business_label(self) -> str:
        """Get business-friendly label for the pattern"""
        period_labels = {
            12.0: "Annual Pattern",
            6.0: "Semi-Annual Pattern",
            3.0: "Quarterly Pattern",
            1.0: "Monthly Pattern",
            0.5: "Bi-Weekly Pattern",
            0.25: "Weekly Pattern"
        }
        
        # Find closest business period
        if self.period < np.inf:
            closest_period = min(period_labels.keys(), 
                               key=lambda x: abs(x - self.period))
            if abs(closest_period - self.period) < 1.0:
                return period_labels[closest_period]
            else:
                return f"{self.period:.1f}-Month Pattern"
        else:
            return "Long-term Trend"


class Transform(ABC):
    """
    Abstract base class for all transforms.
    Every transform must implement the analyze method.
    """
    
    def __init__(self, config: 'TransformConfig' = None):
        """
        Initialize transform with configuration.
        
        Args:
            config: Configuration object with transform parameters
        """
        from config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG
        self._cache = {}  # Cache for expensive computations
        
    @abstractmethod
    def analyze(self, signal: np.ndarray, **kwargs) -> Tuple[List[Pattern], Dict]:
        """
        Analyze signal and return discovered patterns.
        
        Args:
            signal: Input time series data
            **kwargs: Additional transform-specific parameters
            
        Returns:
            patterns: List of discovered Pattern objects
            artifacts: Dict containing transform-specific outputs
                      (e.g., IMFs for HHT, coefficients for wavelets)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return transform name for identification"""
        pass
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Common preprocessing steps for all transforms.
        
        Args:
            signal: Raw input signal
            
        Returns:
            Preprocessed signal
        """
        processed = signal.copy()
        
        # Handle missing values
        if self.config.handle_missing_values == 'interpolate':
            if np.any(np.isnan(processed)):
                valid_indices = ~np.isnan(processed)
                processed = np.interp(
                    np.arange(len(processed)),
                    np.arange(len(processed))[valid_indices],
                    processed[valid_indices]
                )
        
        # Normalize if configured
        if self.config.signal_normalization:
            mean = np.mean(processed)
            std = np.std(processed)
            if std > 0:
                processed = (processed - mean) / std
        
        # Detrend if configured
        if self.config.detrend_before_analysis:
            from scipy import signal
            processed = signal.detrend(processed)
        
        return processed
    
    def clear_cache(self):
        """Clear cached computations"""
        self._cache.clear()
    
    def get_cache_key(self, signal: np.ndarray, **kwargs) -> str:
        """Generate cache key for given inputs"""
        import hashlib
        signal_hash = hashlib.md5(signal.tobytes()).hexdigest()[:8]
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]
        return f"{self.get_name()}_{signal_hash}_{kwargs_hash}"


class TransformCascade(ABC):
    """
    Abstract base class for transform combinations.
    Cascades combine multiple transforms in specific orders.
    """
    
    def __init__(self, transforms: List[Transform], config: 'TransformConfig' = None):
        """
        Initialize cascade with list of transforms.
        
        Args:
            transforms: Ordered list of Transform objects
            config: Configuration object
        """
        from config import DEFAULT_CONFIG
        self.transforms = transforms
        self.config = config or DEFAULT_CONFIG
        self.execution_history = []  # Track execution for debugging
        
    @abstractmethod
    def analyze(self, signal: np.ndarray, signal_type: str = 'unknown') -> List[Pattern]:
        """
        Run cascade analysis on signal.
        
        Args:
            signal: Input time series
            signal_type: Type of signal ('majority', 'minority', 'unknown')
            
        Returns:
            List of discovered and validated patterns
        """
        pass
    
    def get_name(self) -> str:
        """Get cascade name from component transforms"""
        return "→".join([t.get_name() for t in self.transforms])
    
    def validate_patterns(self, patterns: List[Pattern], 
                         validation_data: Dict = None) -> List[Pattern]:
        """
        Common validation logic for patterns.
        
        Args:
            patterns: List of patterns to validate
            validation_data: Additional data for validation
            
        Returns:
            Validated patterns
        """
        validated = []
        
        for pattern in patterns:
            # Check minimum confidence
            if pattern.confidence < self.config.pattern_confidence_threshold:
                continue
            
            # Check minimum energy if specified
            if hasattr(self.config, 'min_pattern_energy'):
                if pattern.amplitude < self.config.min_pattern_energy:
                    continue
            
            # Apply business relevance boost if applicable
            if pattern.is_business_relevant() and hasattr(self.config, 'business_relevance_boost'):
                pattern.confidence *= self.config.business_relevance_boost
                pattern.confidence = min(1.0, pattern.confidence)
            
            validated.append(pattern)
        
        return validated
    
    def combine_patterns(self, patterns_list: List[List[Pattern]]) -> List[Pattern]:
        """
        Combine patterns from multiple transforms.
        
        Args:
            patterns_list: List of pattern lists from different transforms
            
        Returns:
            Combined and deduplicated patterns
        """
        if not patterns_list:
            return []
        
        # Flatten all patterns
        all_patterns = []
        for patterns in patterns_list:
            all_patterns.extend(patterns)
        
        # Deduplicate based on frequency
        unique_patterns = []
        seen_frequencies = set()
        
        for pattern in all_patterns:
            freq_key = round(pattern.frequency, 3)  # Round for comparison
            if freq_key not in seen_frequencies:
                unique_patterns.append(pattern)
                seen_frequencies.add(freq_key)
            else:
                # If duplicate, keep the one with higher confidence
                for i, unique_p in enumerate(unique_patterns):
                    if abs(unique_p.frequency - pattern.frequency) < 0.01:
                        if pattern.confidence > unique_p.confidence:
                            unique_patterns[i] = pattern
                        break
        
        return unique_patterns
    
    def log_execution(self, step: str, details: Dict = None):
        """Log execution step for debugging"""
        if self.config.verbose:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'cascade': self.get_name(),
                'step': step,
                'details': details or {}
            }
            self.execution_history.append(entry)
            print(f"[{self.get_name()}] {step}")
    
    def get_execution_summary(self) -> str:
        """Get summary of cascade execution"""
        summary = f"Cascade: {self.get_name()}\n"
        summary += f"Steps executed: {len(self.execution_history)}\n"
        
        for entry in self.execution_history:
            summary += f"  - {entry['step']}\n"
            if entry['details']:
                for key, value in entry['details'].items():
                    summary += f"    {key}: {value}\n"
        
        return summary


class SignalProcessor:
    """
    Utility class for common signal processing operations.
    Used by transforms and cascades.
    """
    
    @staticmethod
    def estimate_snr(signal: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        from scipy import signal as scipy_signal
        
        # Detrend to remove slow variations
        detrended = scipy_signal.detrend(signal)
        
        # Use high-frequency components as noise estimate
        b, a = scipy_signal.butter(4, 0.1, 'high')
        noise = scipy_signal.filtfilt(b, a, detrended)
        
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        else:
            return float('inf')
    
    @staticmethod
    def detect_signal_type(signal: np.ndarray) -> str:
        """
        Detect if signal is majority or minority based on characteristics.
        
        Returns:
            'majority', 'minority', or 'unknown'
        """
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        cv = std_val / (mean_val + 1e-10)  # Coefficient of variation
        
        # Simple heuristic based on scale
        if mean_val > 5000:
            return 'majority'
        elif mean_val < 1000:
            return 'minority'
        else:
            # Use variability as additional indicator
            if cv < 0.3:
                return 'majority'
            elif cv > 0.5:
                return 'minority'
            else:
                return 'unknown'
    
    @staticmethod
    def compute_seasonality_strength(signal: np.ndarray, period: int = 12) -> float:
        """
        Compute strength of seasonality at given period.
        
        Args:
            signal: Input signal
            period: Seasonal period to test
            
        Returns:
            Seasonality strength (0-1)
        """
        if len(signal) < 2 * period:
            return 0.0
        
        # Reshape into seasons
        n_complete = len(signal) // period
        seasonal_matrix = signal[:n_complete * period].reshape(n_complete, period)
        
        # Compute seasonal pattern
        seasonal_pattern = np.mean(seasonal_matrix, axis=0)
        
        # Compute strength as ratio of seasonal variance to total variance
        seasonal_var = np.var(seasonal_pattern)
        total_var = np.var(signal)
        
        if total_var > 0:
            return min(seasonal_var / total_var, 1.0)
        else:
            return 0.0

