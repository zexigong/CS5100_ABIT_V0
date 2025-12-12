
import numpy as np
from typing import List, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Pattern, TransformCascade
from transforms import HHTTransform, WaveletTransform, STFTTransform
from config import TransformConfig

class DualTransformCascade(TransformCascade):
    """
    Cascade combining two transforms.
    The order matters - first transform discovers, second validates/enhances.
    
    Supported combinations:
    - HHT → Wavelet: Natural discovery → Multi-scale validation
    - Wavelet → HHT: Scale discovery → Natural mode validation
    - HHT → STFT: Natural discovery → Frequency quantification
    - STFT → HHT: Fixed window discovery → Natural validation
    - Wavelet → STFT: Multi-scale discovery → Precise quantification
    - STFT → Wavelet: Fixed window discovery → Scale validation
    """
    
    def __init__(self, first: str, second: str, config: TransformConfig = None):
        """
        Initialize dual transform cascade.
        
        Args:
            first: First transform type ('hht', 'wavelet', 'stft')
            second: Second transform type
            config: Configuration object
        """
        # Map strings to transform classes
        transform_map = {
            'hht': HHTTransform,
            'wavelet': WaveletTransform,
            'stft': STFTTransform
        }
        
        # Normalize to lowercase
        first = first.lower()
        second = second.lower()
        
        # Validate inputs
        if first not in transform_map or second not in transform_map:
            raise ValueError(f"Unknown transform type: {first} or {second}")
        
        if first == second:
            raise ValueError("Dual cascade should use different transforms")
        
        # Create transform instances
        transforms = [
            transform_map[first](config),
            transform_map[second](config)
        ]
        
        # Initialize parent
        super().__init__(transforms, config)
        
        # Store order for method routing
        self.first_type = first
        self.second_type = second
    
    def analyze(self, signal: np.ndarray, signal_type: str = 'unknown') -> List[Pattern]:
        """
        Run dual transform cascade.
        Routes to specific implementation based on transform combination.
        
        Args:
            signal: Input time series
            signal_type: Type of signal ('majority', 'minority', 'unknown')
            
        Returns:
            List of validated patterns
        """
        self.log_execution(f"Starting dual cascade: {self.get_name()}")
        
        # Route to specific method based on combination
        method_map = {
            ('hht', 'wavelet'): self._hht_wavelet,
            ('wavelet', 'hht'): self._wavelet_hht,
            ('hht', 'stft'): self._hht_stft,
            ('stft', 'hht'): self._stft_hht,
            ('wavelet', 'stft'): self._wavelet_stft,
            ('stft', 'wavelet'): self._stft_wavelet
        }
        
        key = (self.first_type, self.second_type)
        
        if key in method_map:
            patterns = method_map[key](signal, signal_type)
        else:
            # This shouldn't happen with validation, but fallback just in case
            self.log_execution("Warning: Using generic sequential processing")
            patterns = self._sequential_processing(signal, signal_type)
        
        self.log_execution(f"Dual cascade complete", {
            'patterns_found': len(patterns),
            'method_used': self.get_name()
        })
        
        return patterns
    
    # ============================================================
    # HHT-based combinations
    # ============================================================
    
    def _hht_wavelet(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        HHT → Wavelet: Natural mode discovery → Multi-scale validation
        
        Best for: Finding natural oscillations and validating them across scales
        Particularly good for: Non-stationary signals, minority products
        """
        self.log_execution("HHT→Wavelet: Starting HHT discovery phase")
        
        # Step 1: HHT discovers natural patterns without assumptions
        hht_patterns, hht_artifacts = self.transforms[0].analyze(signal, signal_type=signal_type)
        
        self.log_execution("HHT→Wavelet: HHT complete", {
            'patterns_discovered': len(hht_patterns),
            'n_imfs': hht_artifacts.get('n_imfs', 0)
        })
        
        if not hht_patterns:
            return []
        
        # Step 2: Wavelet validates discovered patterns across scales
        wavelet_patterns, wavelet_artifacts = self.transforms[1].analyze(signal)
        power_spectrum = wavelet_artifacts['power_spectrum']
        frequencies = wavelet_artifacts['frequencies']
        scales = wavelet_artifacts['scales']
        
        # Validate each HHT pattern with wavelet
        validated_patterns = []
        
        for pattern in hht_patterns:
            if pattern.frequency > 0:  # Skip trends
                # Find closest frequency in wavelet analysis
                freq_idx = np.argmin(np.abs(frequencies - pattern.frequency))
                wavelet_power = power_spectrum[freq_idx]
                avg_power = np.mean(power_spectrum)
                
                # Calculate validation score
                validation_score = wavelet_power / (avg_power + 1e-10)
                
                if validation_score > self.config.cascade_validation_threshold:
                    # Pattern is validated - update with wavelet information
                    pattern.confidence = (pattern.confidence + validation_score) / 2
                    pattern.metadata['wavelet_validated'] = True
                    pattern.metadata['wavelet_power'] = float(wavelet_power)
                    pattern.metadata['wavelet_scale'] = float(scales[freq_idx])
                    pattern.metadata['validation_score'] = float(validation_score)
                    pattern.source_method = "HHT→Wavelet"
                    validated_patterns.append(pattern)
                else:
                    self.log_execution(f"Pattern at {pattern.frequency:.4f} Hz rejected", {
                        'validation_score': validation_score
                    })
            else:
                # Trend patterns pass through without validation
                pattern.source_method = "HHT→Wavelet"
                validated_patterns.append(pattern)
        
        self.log_execution("HHT→Wavelet: Validation complete", {
            'patterns_validated': len(validated_patterns),
            'validation_rate': len(validated_patterns) / len(hht_patterns) if hht_patterns else 0
        })
        
        return validated_patterns
    
    def _wavelet_hht(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        Wavelet → HHT: Multi-scale discovery → Natural mode validation
        
        Best for: Finding patterns at specific scales and confirming they're natural modes
        Good for: Stable patterns that need natural validation
        """
        self.log_execution("Wavelet→HHT: Starting Wavelet discovery phase")
        
        # Step 1: Wavelet discovers patterns at multiple scales
        wavelet_patterns, wavelet_artifacts = self.transforms[0].analyze(signal)
        
        self.log_execution("Wavelet→HHT: Wavelet complete", {
            'patterns_discovered': len(wavelet_patterns)
        })
        
        if not wavelet_patterns:
            return []
        
        # Step 2: HHT validates that these are natural modes
        hht_patterns, hht_artifacts = self.transforms[1].analyze(signal, signal_type=signal_type)
        
        # Validate wavelet patterns against HHT's natural modes
        validated_patterns = []
        
        for wav_pattern in wavelet_patterns:
            validated = False
            
            # Check if any HHT IMF matches this frequency
            for hht_pattern in hht_patterns:
                if hht_pattern.matches_frequency(wav_pattern.frequency, self.config.frequency_tolerance):
                    # Pattern confirmed as natural mode
                    wav_pattern.metadata['hht_validated'] = True
                    wav_pattern.metadata['is_natural_mode'] = True
                    wav_pattern.metadata['imf_index'] = hht_pattern.metadata.get('imf_index', -1)
                    wav_pattern.confidence = (wav_pattern.confidence + hht_pattern.confidence) / 2
                    wav_pattern.source_method = "Wavelet→HHT"
                    validated_patterns.append(wav_pattern)
                    validated = True
                    break
            
            if not validated:
                self.log_execution(f"Pattern at scale {wav_pattern.metadata.get('scale', 0)} not natural")
        
        self.log_execution("Wavelet→HHT: Validation complete", {
            'patterns_validated': len(validated_patterns)
        })
        
        return validated_patterns
    
    def _hht_stft(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        HHT → STFT: Natural mode discovery → Precise frequency quantification
        
        Best for: Finding natural patterns and measuring their exact frequencies
        Good for: Quantifying time-varying patterns discovered by HHT
        """
        self.log_execution("HHT→STFT: Starting HHT discovery phase")
        
        # Step 1: HHT discovers natural modes
        hht_patterns, hht_artifacts = self.transforms[0].analyze(signal, signal_type=signal_type)
        
        self.log_execution("HHT→STFT: HHT complete", {
            'patterns_discovered': len(hht_patterns)
        })
        
        if not hht_patterns:
            return []
        
        # Optimize STFT window based on HHT findings
        periods = [p.period for p in hht_patterns if 0 < p.period < np.inf]
        if periods:
            optimal_window = int(np.median(periods) * 2)
            optimal_window = np.clip(optimal_window, 12, 48)
        else:
            optimal_window = self.config.stft_nperseg
        
        self.log_execution("HHT→STFT: Optimized STFT window", {
            'window_size': optimal_window
        })
        
        # Step 2: STFT quantifies the discovered patterns
        stft_patterns, stft_artifacts = self.transforms[1].analyze(signal, window_size=optimal_window)
        magnitude = stft_artifacts['magnitude']
        frequencies = stft_artifacts['frequencies']
        
        # Quantify each HHT pattern with STFT
        quantified_patterns = []
        
        for hht_pattern in hht_patterns:
            if hht_pattern.frequency > 0 and hht_pattern.frequency < 0.5:
                # Find corresponding frequency in STFT
                freq_idx = np.argmin(np.abs(frequencies - hht_pattern.frequency))
                
                if freq_idx < len(magnitude):
                    freq_magnitude = magnitude[freq_idx, :]
                    
                    # Update pattern with STFT measurements
                    hht_pattern.amplitude = float(np.mean(freq_magnitude))
                    hht_pattern.metadata['stft_amplitude'] = hht_pattern.amplitude
                    hht_pattern.metadata['stft_std'] = float(np.std(freq_magnitude))
                    hht_pattern.metadata['stft_max'] = float(np.max(freq_magnitude))
                    hht_pattern.metadata['time_varying'] = np.std(freq_magnitude) > 0.3 * np.mean(freq_magnitude)
                    hht_pattern.metadata['stft_quantified'] = True
                    hht_pattern.source_method = "HHT→STFT"
                    quantified_patterns.append(hht_pattern)
            else:
                # Keep trends and DC components
                hht_pattern.source_method = "HHT→STFT"
                quantified_patterns.append(hht_pattern)
        
        self.log_execution("HHT→STFT: Quantification complete", {
            'patterns_quantified': len(quantified_patterns)
        })
        
        return quantified_patterns
    
    def _stft_hht(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        STFT → HHT: Fixed window discovery → Natural mode validation
        
        Best for: Finding stable frequencies and checking if they're natural
        Good for: Signals with known frequency content needing validation
        """
        self.log_execution("STFT→HHT: Starting STFT discovery phase")
        
        # Step 1: STFT discovers patterns with fixed window
        stft_patterns, stft_artifacts = self.transforms[0].analyze(signal)
        
        self.log_execution("STFT→HHT: STFT complete", {
            'patterns_discovered': len(stft_patterns)
        })
        
        if not stft_patterns:
            return []
        
        # Step 2: HHT validates as natural modes
        hht_patterns, hht_artifacts = self.transforms[1].analyze(signal, signal_type=signal_type)
        
        # Validate STFT patterns against HHT's natural modes
        validated_patterns = []
        
        for stft_pattern in stft_patterns:
            validated = False
            
            for hht_pattern in hht_patterns:
                if hht_pattern.matches_frequency(stft_pattern.frequency, self.config.frequency_tolerance):
                    # Pattern confirmed as natural mode
                    stft_pattern.metadata['hht_validated'] = True
                    stft_pattern.metadata['is_natural_mode'] = True
                    stft_pattern.confidence = (stft_pattern.confidence + hht_pattern.confidence) / 2
                    stft_pattern.source_method = "STFT→HHT"
                    validated_patterns.append(stft_pattern)
                    validated = True
                    break
            
            if not validated:
                self.log_execution(f"STFT frequency {stft_pattern.frequency:.4f} Hz not natural")
        
        self.log_execution("STFT→HHT: Validation complete", {
            'patterns_validated': len(validated_patterns)
        })
        
        return validated_patterns
    
    # ============================================================
    # Wavelet-STFT combinations
    # ============================================================
    
    def _wavelet_stft(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        Wavelet → STFT: Multi-scale discovery → Precise quantification
        
        Best for: Finding patterns at multiple scales and measuring precisely
        Good for: Comprehensive analysis of stable patterns
        """
        self.log_execution("Wavelet→STFT: Starting Wavelet discovery phase")
        
        # Step 1: Wavelet discovers patterns at multiple scales
        wavelet_patterns, wavelet_artifacts = self.transforms[0].analyze(signal)
        
        self.log_execution("Wavelet→STFT: Wavelet complete", {
            'patterns_discovered': len(wavelet_patterns)
        })
        
        if not wavelet_patterns:
            return []
        
        # Optimize STFT window based on wavelet findings
        periods = [p.period for p in wavelet_patterns if p.period < np.inf]
        if periods:
            optimal_window = int(np.median(periods) * 2)
            optimal_window = np.clip(optimal_window, 12, 48)
        else:
            optimal_window = self.config.stft_nperseg
        
        # Step 2: STFT quantifies with optimized parameters
        stft_patterns, stft_artifacts = self.transforms[1].analyze(signal, window_size=optimal_window)
        magnitude = stft_artifacts['magnitude']
        frequencies = stft_artifacts['frequencies']
        
        # Enhance wavelet patterns with STFT quantification
        quantified_patterns = []
        
        for wav_pattern in wavelet_patterns:
            if wav_pattern.frequency > 0 and wav_pattern.frequency < 0.5:
                # Find corresponding STFT frequency
                freq_idx = np.argmin(np.abs(frequencies - wav_pattern.frequency))
                
                if freq_idx < len(magnitude):
                    freq_magnitude = magnitude[freq_idx, :]
                    
                    # Update with precise measurements
                    wav_pattern.amplitude = float(np.mean(freq_magnitude))
                    wav_pattern.metadata['stft_amplitude'] = wav_pattern.amplitude
                    wav_pattern.metadata['stft_std'] = float(np.std(freq_magnitude))
                    wav_pattern.metadata['stft_max'] = float(np.max(freq_magnitude))
                    wav_pattern.metadata['time_varying'] = np.std(freq_magnitude) > 0.3 * np.mean(freq_magnitude)
                    wav_pattern.source_method = "Wavelet→STFT"
                    quantified_patterns.append(wav_pattern)
        
        self.log_execution("Wavelet→STFT: Quantification complete", {
            'patterns_quantified': len(quantified_patterns)
        })
        
        return quantified_patterns
    
    def _stft_wavelet(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        STFT → Wavelet: Fixed window discovery → Multi-scale validation
        
        Best for: Finding specific frequencies and validating across scales
        Good for: Known frequency content needing scale confirmation
        """
        self.log_execution("STFT→Wavelet: Starting STFT discovery phase")
        
        # Step 1: STFT discovers patterns with fixed window
        stft_patterns, stft_artifacts = self.transforms[0].analyze(signal)
        
        self.log_execution("STFT→Wavelet: STFT complete", {
            'patterns_discovered': len(stft_patterns)
        })
        
        if not stft_patterns:
            return []
        
        # Step 2: Wavelet validates across multiple scales
        wavelet_patterns, wavelet_artifacts = self.transforms[1].analyze(signal)
        power_spectrum = wavelet_artifacts['power_spectrum']
        frequencies = wavelet_artifacts['frequencies']
        scales = wavelet_artifacts['scales']
        
        # Validate STFT patterns with wavelet
        validated_patterns = []
        
        for stft_pattern in stft_patterns:
            if stft_pattern.frequency > 0:
                # Find wavelet validation
                freq_idx = np.argmin(np.abs(frequencies - stft_pattern.frequency))
                wavelet_power = power_spectrum[freq_idx]
                
                # Check if significant in wavelet domain
                if wavelet_power > np.mean(power_spectrum) * 0.5:
                    stft_pattern.metadata['wavelet_validated'] = True
                    stft_pattern.metadata['wavelet_power'] = float(wavelet_power)
                    stft_pattern.metadata['wavelet_scale'] = float(scales[freq_idx])
                    stft_pattern.source_method = "STFT→Wavelet"
                    validated_patterns.append(stft_pattern)
        
        self.log_execution("STFT→Wavelet: Validation complete", {
            'patterns_validated': len(validated_patterns)
        })
        
        return validated_patterns
    
    # ============================================================
    # Fallback method
    # ============================================================
    
    def _sequential_processing(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        Fallback: Simple sequential processing without specific optimization.
        This should rarely be used due to input validation.
        """
        self.log_execution("Sequential: Running first transform")
        
        # Run first transform
        if self.first_type == 'hht':
            patterns1, _ = self.transforms[0].analyze(signal, signal_type=signal_type)
        else:
            patterns1, _ = self.transforms[0].analyze(signal)
        
        self.log_execution("Sequential: Running second transform")
        
        # Run second transform
        if self.second_type == 'hht':
            patterns2, _ = self.transforms[1].analyze(signal, signal_type=signal_type)
        else:
            patterns2, _ = self.transforms[1].analyze(signal)
        
        # Combine patterns (avoiding duplicates)
        combined = self.combine_patterns([patterns1, patterns2])
        
        # Update source method
        for pattern in combined:
            pattern.source_method = self.get_name()
        
        return combined