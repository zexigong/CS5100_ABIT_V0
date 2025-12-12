import numpy as np
from typing import List, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Pattern, TransformCascade
from transforms import HHTTransform, WaveletTransform, STFTTransform
from config import TransformConfig

class TripleTransformCascade(TransformCascade):
    """
    Cascade combining three transforms.
    
    Typical flow: Discovery → Validation → Quantification
    
    All 6 permutations implemented:
    1. HHT→Wavelet→STFT: Natural discovery → Scale validation → Precise quantification
    2. HHT→STFT→Wavelet: Natural discovery → Quantification → Scale validation
    3. Wavelet→HHT→STFT: Scale discovery → Natural validation → Quantification
    4. Wavelet→STFT→HHT: Scale discovery → Quantification → Natural validation
    5. STFT→HHT→Wavelet: Fixed discovery → Natural validation → Scale confirmation
    6. STFT→Wavelet→HHT: Fixed discovery → Scale validation → Natural confirmation
    """
    
    def __init__(self, first: str, second: str, third: str, config: TransformConfig = None):
        """
        Initialize triple transform cascade.
        
        Args:
            first: First transform type ('hht', 'wavelet', 'stft')
            second: Second transform type
            third: Third transform type
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
        third = third.lower()
        
        # Validate transform types
        for transform_type in [first, second, third]:
            if transform_type not in transform_map:
                raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Check for diversity (at least two different transforms)
        unique_transforms = set([first, second, third])
        if len(unique_transforms) < 2:
            raise ValueError("Triple cascade should use at least two different transform types")
        
        # Create transform instances
        transforms = [
            transform_map[first](config),
            transform_map[second](config),
            transform_map[third](config)
        ]
        
        # Initialize parent class
        super().__init__(transforms, config)
        
        # Store order for routing
        self.order = (first, second, third)
    
    def analyze(self, signal: np.ndarray, signal_type: str = 'unknown') -> List[Pattern]:
        """
        Run triple transform cascade.
        
        Args:
            signal: Input time series
            signal_type: 'majority', 'minority', or 'unknown'
            
        Returns:
            List of patterns after three-stage processing
        """
        self.log_execution(f"Starting triple cascade: {self.get_name()}", {
            'signal_length': len(signal),
            'signal_type': signal_type
        })
        
        # Route to specific implementation based on order
        if self.order == ('hht', 'wavelet', 'stft'):
            patterns = self._hht_wavelet_stft(signal, signal_type)
        elif self.order == ('hht', 'stft', 'wavelet'):
            patterns = self._hht_stft_wavelet(signal, signal_type)
        elif self.order == ('wavelet', 'hht', 'stft'):
            patterns = self._wavelet_hht_stft(signal, signal_type)
        elif self.order == ('wavelet', 'stft', 'hht'):
            patterns = self._wavelet_stft_hht(signal, signal_type)
        elif self.order == ('stft', 'hht', 'wavelet'):
            patterns = self._stft_hht_wavelet(signal, signal_type)
        elif self.order == ('stft', 'wavelet', 'hht'):
            patterns = self._stft_wavelet_hht(signal, signal_type)
        else:
            # Fallback to generic implementation
            patterns = self._generic_triple(signal, signal_type)
        
        self.log_execution(f"Triple cascade complete", {
            'patterns_final': len(patterns)
        })
        
        return patterns
    
    def _hht_wavelet_stft(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        HHT→Wavelet→STFT: The original ABIT cascade
        Natural discovery → Multi-scale validation → Precise quantification
        
        Best for: Unknown patterns, minority products, comprehensive analysis
        """
        self.log_execution("HHT→Wavelet→STFT: Starting HHT discovery")
        
        # Step 1: HHT discovers natural patterns without assumptions
        hht_patterns, hht_artifacts = self.transforms[0].analyze(signal, signal_type=signal_type)
        
        if not hht_patterns:
            self.log_execution("HHT→Wavelet→STFT: No patterns discovered by HHT")
            return []
        
        self.log_execution("HHT→Wavelet→STFT: HHT complete", {
            'patterns_discovered': len(hht_patterns),
            'n_imfs': hht_artifacts.get('n_imfs', 0)
        })
        
        # Step 2: Wavelet validates discovered patterns across scales
        self.log_execution("HHT→Wavelet→STFT: Starting Wavelet validation")
        
        wavelet_patterns, wavelet_artifacts = self.transforms[1].analyze(signal)
        power_spectrum = wavelet_artifacts['power_spectrum']
        frequencies = wavelet_artifacts['frequencies']
        scales = wavelet_artifacts['scales']
        
        validated_patterns = []
        
        for pattern in hht_patterns:
            if pattern.frequency > 0:
                # Find wavelet validation
                freq_idx = np.argmin(np.abs(frequencies - pattern.frequency))
                wavelet_power = power_spectrum[freq_idx]
                validation_score = wavelet_power / (np.mean(power_spectrum) + 1e-10)
                
                if validation_score > self.config.cascade_validation_threshold:
                    pattern.metadata['wavelet_validated'] = True
                    pattern.metadata['wavelet_score'] = float(validation_score)
                    pattern.metadata['wavelet_scale'] = int(scales[freq_idx])
                    validated_patterns.append(pattern)
            else:
                # Keep trends without frequency validation
                validated_patterns.append(pattern)
        
        if not validated_patterns:
            self.log_execution("HHT→Wavelet→STFT: No patterns validated by Wavelet")
            return []
        
        self.log_execution("HHT→Wavelet→STFT: Wavelet validation complete", {
            'patterns_validated': len(validated_patterns)
        })
        
        # Step 3: STFT quantifies validated patterns with optimal parameters
        self.log_execution("HHT→Wavelet→STFT: Starting STFT quantification")
        
        # Optimize STFT window based on validated patterns
        periods = [p.period for p in validated_patterns if 0 < p.period < np.inf]
        if periods:
            optimal_window = int(np.median(periods) * 2)
            optimal_window = np.clip(optimal_window, 12, 48)
        else:
            optimal_window = 24
        
        stft_patterns, stft_artifacts = self.transforms[2].analyze(signal, window_size=optimal_window)
        magnitude = stft_artifacts['magnitude']
        stft_frequencies = stft_artifacts['frequencies']
        
        # Quantify each validated pattern
        final_patterns = []
        
        for pattern in validated_patterns:
            if 0 < pattern.frequency < 0.5:
                # Find STFT quantification
                freq_idx = np.argmin(np.abs(stft_frequencies - pattern.frequency))
                
                if freq_idx < len(magnitude):
                    freq_magnitude = magnitude[freq_idx, :]
                    
                    # Update pattern with precise measurements
                    pattern.amplitude = float(np.mean(freq_magnitude))
                    pattern.metadata['stft_quantified'] = True
                    pattern.metadata['stft_amplitude'] = pattern.amplitude
                    pattern.metadata['stft_std'] = float(np.std(freq_magnitude))
                    pattern.metadata['stft_max'] = float(np.max(freq_magnitude))
                    pattern.metadata['time_varying'] = np.std(freq_magnitude) > 0.3 * np.mean(freq_magnitude)
                    
                    # Calculate combined confidence from all three transforms
                    hht_conf = pattern.confidence
                    wav_conf = pattern.metadata.get('wavelet_score', 0.5)
                    stft_conf = min(pattern.amplitude / (np.max(magnitude) + 1e-10), 1.0)
                    
                    pattern.confidence = float(np.mean([hht_conf, wav_conf, stft_conf]))
            
            pattern.source_method = "HHT→Wavelet→STFT"
            final_patterns.append(pattern)
        
        self.log_execution("HHT→Wavelet→STFT: Complete", {
            'final_patterns': len(final_patterns),
            'window_used': optimal_window
        })
        
        return final_patterns
    
    def _hht_stft_wavelet(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        HHT→STFT→Wavelet: Natural discovery → Quantification → Scale confirmation
        """
        # Step 1: HHT discovers natural modes
        hht_patterns, _ = self.transforms[0].analyze(signal, signal_type=signal_type)
        
        if not hht_patterns:
            return []
        
        # Step 2: STFT quantifies discovered patterns
        periods = [p.period for p in hht_patterns if 0 < p.period < np.inf]
        optimal_window = int(np.median(periods) * 2) if periods else 24
        optimal_window = np.clip(optimal_window, 12, 48)
        
        stft_patterns, stft_artifacts = self.transforms[1].analyze(signal, window_size=optimal_window)
        magnitude = stft_artifacts['magnitude']
        stft_frequencies = stft_artifacts['frequencies']
        
        quantified_patterns = []
        for pattern in hht_patterns:
            if 0 < pattern.frequency < 0.5:
                freq_idx = np.argmin(np.abs(stft_frequencies - pattern.frequency))
                if freq_idx < len(magnitude):
                    freq_magnitude = magnitude[freq_idx, :]
                    pattern.amplitude = float(np.mean(freq_magnitude))
                    pattern.metadata['stft_amplitude'] = pattern.amplitude
                    pattern.metadata['stft_std'] = float(np.std(freq_magnitude))
                    quantified_patterns.append(pattern)
            else:
                quantified_patterns.append(pattern)
        
        # Step 3: Wavelet confirms at appropriate scales
        wavelet_patterns, wavelet_artifacts = self.transforms[2].analyze(signal)
        power_spectrum = wavelet_artifacts['power_spectrum']
        frequencies = wavelet_artifacts['frequencies']
        scales = wavelet_artifacts['scales']
        
        final_patterns = []
        for pattern in quantified_patterns:
            if pattern.frequency > 0:
                freq_idx = np.argmin(np.abs(frequencies - pattern.frequency))
                pattern.metadata['wavelet_scale'] = int(scales[freq_idx])
                pattern.metadata['wavelet_power'] = float(power_spectrum[freq_idx])
                
                # Combined confidence
                pattern.confidence = float(np.mean([
                    pattern.confidence,
                    pattern.metadata.get('stft_amplitude', 0) / (np.max(magnitude) + 1e-10),
                    power_spectrum[freq_idx] / (np.max(power_spectrum) + 1e-10)
                ]))
            
            pattern.source_method = "HHT→STFT→Wavelet"
            final_patterns.append(pattern)
        
        return final_patterns
    
    def _wavelet_hht_stft(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        Wavelet→HHT→STFT: Multi-scale discovery → Natural validation → Quantification
        """
        # Step 1: Wavelet discovers multi-scale patterns
        wavelet_patterns, _ = self.transforms[0].analyze(signal)
        
        if not wavelet_patterns:
            return []
        
        # Step 2: HHT validates as natural modes
        hht_patterns, _ = self.transforms[1].analyze(signal, signal_type=signal_type)
        
        validated_patterns = []
        for wav_pattern in wavelet_patterns:
            for hht_pattern in hht_patterns:
                if hht_pattern.matches_frequency(wav_pattern.frequency, self.config.frequency_tolerance):
                    wav_pattern.metadata['hht_validated'] = True
                    wav_pattern.metadata['hht_confidence'] = hht_pattern.confidence
                    wav_pattern.metadata['imf_index'] = hht_pattern.metadata.get('imf_index', -1)
                    validated_patterns.append(wav_pattern)
                    break
        
        if not validated_patterns:
            return []
        
        # Step 3: STFT quantifies validated patterns
        periods = [p.period for p in validated_patterns if p.period < np.inf]
        optimal_window = int(np.median(periods) * 2) if periods else 24
        optimal_window = np.clip(optimal_window, 12, 48)
        
        stft_patterns, stft_artifacts = self.transforms[2].analyze(signal, window_size=optimal_window)
        magnitude = stft_artifacts['magnitude']
        frequencies = stft_artifacts['frequencies']
        
        final_patterns = []
        for pattern in validated_patterns:
            if 0 < pattern.frequency < 0.5:
                freq_idx = np.argmin(np.abs(frequencies - pattern.frequency))
                if freq_idx < len(magnitude):
                    freq_magnitude = magnitude[freq_idx, :]
                    pattern.amplitude = float(np.mean(freq_magnitude))
                    pattern.metadata['stft_amplitude'] = pattern.amplitude
                    pattern.metadata['stft_std'] = float(np.std(freq_magnitude))
                    
                    # Combined confidence from all three
                    pattern.confidence = float(np.mean([
                        pattern.confidence,
                        pattern.metadata.get('hht_confidence', 0.5),
                        min(pattern.amplitude / (np.max(magnitude) + 1e-10), 1.0)
                    ]))
            
            pattern.source_method = "Wavelet→HHT→STFT"
            final_patterns.append(pattern)
        
        return final_patterns
    
    def _wavelet_stft_hht(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        Wavelet→STFT→HHT: Multi-scale discovery → Quantification → Natural confirmation
        """
        # Step 1: Wavelet discovers patterns
        wavelet_patterns, _ = self.transforms[0].analyze(signal)
        
        if not wavelet_patterns:
            return []
        
        # Step 2: STFT quantifies
        periods = [p.period for p in wavelet_patterns if p.period < np.inf]
        optimal_window = int(np.median(periods) * 2) if periods else 24
        optimal_window = np.clip(optimal_window, 12, 48)
        
        stft_patterns, stft_artifacts = self.transforms[1].analyze(signal, window_size=optimal_window)
        magnitude = stft_artifacts['magnitude']
        frequencies = stft_artifacts['frequencies']
        
        quantified_patterns = []
        for pattern in wavelet_patterns:
            if 0 < pattern.frequency < 0.5:
                freq_idx = np.argmin(np.abs(frequencies - pattern.frequency))
                if freq_idx < len(magnitude):
                    freq_magnitude = magnitude[freq_idx, :]
                    pattern.amplitude = float(np.mean(freq_magnitude))
                    pattern.metadata['stft_amplitude'] = pattern.amplitude
                    quantified_patterns.append(pattern)
        
        if not quantified_patterns:
            return []
        
        # Step 3: HHT confirms as natural modes
        hht_patterns, _ = self.transforms[2].analyze(signal, signal_type=signal_type)
        
        final_patterns = []
        for pattern in quantified_patterns:
            pattern.metadata['hht_confirmed'] = False
            for hht_pattern in hht_patterns:
                if hht_pattern.matches_frequency(pattern.frequency, self.config.frequency_tolerance):
                    pattern.metadata['hht_confirmed'] = True
                    pattern.metadata['is_natural_mode'] = True
                    pattern.confidence = float(np.mean([
                        pattern.confidence,
                        hht_pattern.confidence,
                        min(pattern.amplitude / (np.max(magnitude) + 1e-10), 1.0)
                    ]))
                    break
            
            pattern.source_method = "Wavelet→STFT→HHT"
            final_patterns.append(pattern)
        
        return final_patterns
    
    def _stft_hht_wavelet(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        STFT→HHT→Wavelet: Fixed-window discovery → Natural validation → Scale confirmation
        """
        # Step 1: STFT discovers with fixed window
        stft_patterns, _ = self.transforms[0].analyze(signal)
        
        if not stft_patterns:
            return []
        
        # Step 2: HHT validates as natural modes
        hht_patterns, _ = self.transforms[1].analyze(signal, signal_type=signal_type)
        
        validated_patterns = []
        for stft_pattern in stft_patterns:
            for hht_pattern in hht_patterns:
                if hht_pattern.matches_frequency(stft_pattern.frequency, self.config.frequency_tolerance):
                    stft_pattern.metadata['hht_validated'] = True
                    stft_pattern.metadata['imf_index'] = hht_pattern.metadata.get('imf_index', -1)
                    validated_patterns.append(stft_pattern)
                    break
        
        if not validated_patterns:
            return []
        
        # Step 3: Wavelet confirms at multiple scales
        wavelet_patterns, wavelet_artifacts = self.transforms[2].analyze(signal)
        power_spectrum = wavelet_artifacts['power_spectrum']
        frequencies = wavelet_artifacts['frequencies']
        scales = wavelet_artifacts['scales']
        
        final_patterns = []
        for pattern in validated_patterns:
            if pattern.frequency > 0:
                freq_idx = np.argmin(np.abs(frequencies - pattern.frequency))
                pattern.metadata['wavelet_power'] = float(power_spectrum[freq_idx])
                pattern.metadata['wavelet_scale'] = int(scales[freq_idx])
                
                # Combined confidence
                wav_conf = power_spectrum[freq_idx] / (np.max(power_spectrum) + 1e-10)
                pattern.confidence = float(np.mean([pattern.confidence, wav_conf]))
            
            pattern.source_method = "STFT→HHT→Wavelet"
            final_patterns.append(pattern)
        
        return final_patterns
    
    def _stft_wavelet_hht(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        STFT→Wavelet→HHT: Fixed-window discovery → Scale validation → Natural confirmation
        """
        # Step 1: STFT discovers patterns
        stft_patterns, _ = self.transforms[0].analyze(signal)
        
        if not stft_patterns:
            return []
        
        # Step 2: Wavelet validates across scales
        wavelet_patterns, wavelet_artifacts = self.transforms[1].analyze(signal)
        power_spectrum = wavelet_artifacts['power_spectrum']
        frequencies = wavelet_artifacts['frequencies']
        
        validated_patterns = []
        for stft_pattern in stft_patterns:
            if stft_pattern.frequency > 0:
                freq_idx = np.argmin(np.abs(frequencies - stft_pattern.frequency))
                wavelet_power = power_spectrum[freq_idx]
                
                if wavelet_power > np.mean(power_spectrum) * 0.5:
                    stft_pattern.metadata['wavelet_validated'] = True
                    stft_pattern.metadata['wavelet_power'] = float(wavelet_power)
                    validated_patterns.append(stft_pattern)
        
        if not validated_patterns:
            return []
        
        # Step 3: HHT confirms as natural modes
        hht_patterns, _ = self.transforms[2].analyze(signal, signal_type=signal_type)
        
        final_patterns = []
        for pattern in validated_patterns:
            pattern.metadata['hht_confirmed'] = False
            for hht_pattern in hht_patterns:
                if hht_pattern.matches_frequency(pattern.frequency, self.config.frequency_tolerance):
                    pattern.metadata['hht_confirmed'] = True
                    pattern.confidence = float((pattern.confidence + hht_pattern.confidence) / 2)
                    break
            
            pattern.source_method = "STFT→Wavelet→HHT"
            final_patterns.append(pattern)
        
        return final_patterns
    
    def _generic_triple(self, signal: np.ndarray, signal_type: str) -> List[Pattern]:
        """
        Generic triple cascade for any order.
        Implements: Discovery → Validation → Enhancement
        """
        self.log_execution("Generic triple: Starting first transform (Discovery)")
        
        # Step 1: First transform discovers patterns
        if self.order[0] == 'hht':
            patterns1, artifacts1 = self.transforms[0].analyze(signal, signal_type=signal_type)
        else:
            patterns1, artifacts1 = self.transforms[0].analyze(signal)
        
        if not patterns1:
            return []
        
        self.log_execution("Generic triple: First transform complete", {
            'patterns_discovered': len(patterns1)
        })
        
        # Step 2: Second transform validates
        self.log_execution("Generic triple: Starting second transform (Validation)")
        
        if self.order[1] == 'hht':
            patterns2, artifacts2 = self.transforms[1].analyze(signal, signal_type=signal_type)
        else:
            patterns2, artifacts2 = self.transforms[1].analyze(signal)
        
        # Validate: Keep patterns found by both transforms
        validated_patterns = []
        for p1 in patterns1:
            for p2 in patterns2:
                if p2.matches_frequency(p1.frequency, self.config.frequency_tolerance):
                    p1.metadata['second_validated'] = True
                    p1.metadata['second_confidence'] = p2.confidence
                    p1.confidence = (p1.confidence + p2.confidence) / 2
                    validated_patterns.append(p1)
                    break
        
        # If no validation, keep high-confidence patterns
        if not validated_patterns:
            validated_patterns = [p for p in patterns1 if p.confidence > 0.7]
        
        if not validated_patterns:
            return []
        
        self.log_execution("Generic triple: Second transform complete", {
            'patterns_validated': len(validated_patterns)
        })
        
        # Step 3: Third transform enhances
        self.log_execution("Generic triple: Starting third transform (Enhancement)")
        
        if self.order[2] == 'hht':
            patterns3, artifacts3 = self.transforms[2].analyze(signal, signal_type=signal_type)
        else:
            patterns3, artifacts3 = self.transforms[2].analyze(signal)
        
        # Enhance validated patterns
        final_patterns = []
        for pattern in validated_patterns:
            enhanced = False
            for p3 in patterns3:
                if p3.matches_frequency(pattern.frequency, self.config.frequency_tolerance):
                    pattern.metadata['third_enhanced'] = True
                    
                    # Update amplitude if available
                    if p3.amplitude > 0 and pattern.amplitude > 0:
                        pattern.amplitude = (pattern.amplitude + p3.amplitude) / 2
                    elif p3.amplitude > 0:
                        pattern.amplitude = p3.amplitude
                    
                    # Update confidence (average of all three)
                    pattern.confidence = float(np.mean([
                        pattern.confidence,
                        pattern.metadata.get('second_confidence', 0.5),
                        p3.confidence
                    ]))
                    enhanced = True
                    break
            
            # Keep pattern even if not enhanced by third transform
            pattern.source_method = self.get_name()
            final_patterns.append(pattern)
        
        self.log_execution("Generic triple: Complete", {
            'final_patterns': len(final_patterns)
        })
        
        return final_patterns