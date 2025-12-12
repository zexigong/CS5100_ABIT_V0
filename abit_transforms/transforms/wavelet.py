import numpy as np
import pywt
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Transform, Pattern
from config import TransformConfig, DEFAULT_CONFIG

class WaveletTransform(Transform):
    """
    Wavelet Transform implementation.
    
    Provides both Continuous Wavelet Transform (CWT) for analysis
    and Discrete Wavelet Transform (DWT) for decomposition.
    """
    
    def get_name(self) -> str:
        return "Wavelet"
    
    def analyze(self, signal: np.ndarray, **kwargs) -> Tuple[List[Pattern], Dict]:
        """
        Perform Wavelet analysis on signal.
        
        Args:
            signal: Input time series
            
        Returns:
            patterns: List of discovered patterns
            artifacts: Dict containing coefficients and scales
        """
        patterns = []
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal)
        
        # Continuous Wavelet Transform for pattern discovery
        cwt_coeffs, frequencies, scales = self._perform_cwt(processed_signal)
        
        # Calculate power spectrum
        power = np.mean(np.abs(cwt_coeffs)**2, axis=1)
        
        # Find peaks in power spectrum
        peaks, properties = find_peaks(
            power,
            height=np.max(power) * 0.15,  # At least 15% of max power
            distance=2  # Minimum distance between peaks
        )
        
        # Create patterns from peaks
        for i, peak_idx in enumerate(peaks):
            freq = frequencies[peak_idx]
            
            if freq > 0 and freq < 0.5:  # Valid frequency range
                # Calculate pattern characteristics
                scale = scales[peak_idx]
                peak_power = power[peak_idx]
                
                # Get peak width as measure of frequency spread
                if 'widths' in properties:
                    width = properties['widths'][i] if i < len(properties['widths']) else 1
                else:
                    width = 1
                
                # Extract time-varying amplitude for this scale
                scale_coeffs = cwt_coeffs[peak_idx, :]
                amplitude = np.mean(np.abs(scale_coeffs))
                amplitude_std = np.std(np.abs(scale_coeffs))
                
                pattern = Pattern(
                    frequency=freq,
                    period=1/freq,
                    amplitude=amplitude,
                    confidence=peak_power / np.max(power),
                    source_method="Wavelet",
                    metadata={
                        'scale': int(scale),
                        'power': float(peak_power),
                        'peak_width': float(width),
                        'amplitude_std': float(amplitude_std),
                        'is_persistent': amplitude_std < 0.5 * amplitude,
                        'wavelet_type': self.config.wavelet_type
                    }
                )
                patterns.append(pattern)
        
        # Also perform DWT for decomposition information
        dwt_coeffs = self._perform_dwt(processed_signal)
        
        # Analyze DWT levels for additional patterns
        dwt_patterns = self._analyze_dwt_levels(dwt_coeffs, processed_signal)
        
        # Combine CWT and DWT patterns (avoiding duplicates)
        for dwt_p in dwt_patterns:
            is_duplicate = False
            for cwt_p in patterns:
                if cwt_p.matches_frequency(dwt_p.frequency, tolerance=0.02):
                    is_duplicate = True
                    # Update confidence if DWT also found it
                    cwt_p.confidence = (cwt_p.confidence + dwt_p.confidence) / 2
                    cwt_p.metadata['dwt_confirmed'] = True
                    break
            
            if not is_duplicate:
                patterns.append(dwt_p)
        
        # Store artifacts
        artifacts = {
            'cwt_coefficients': cwt_coeffs,
            'frequencies': frequencies,
            'scales': scales,
            'power_spectrum': power,
            'dwt_coefficients': dwt_coeffs,
            'peaks': peaks
        }
        
        return patterns, artifacts
    
    def _perform_cwt(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Continuous Wavelet Transform.
        
        Returns:
            coefficients: CWT coefficients (scales x time)
            frequencies: Corresponding frequencies for each scale
            scales: Scale parameters used
        """
        # Use configured scales
        scales = self.config.wavelet_scales
        
        # Perform CWT
        coefficients, frequencies = pywt.cwt(
            signal,
            scales,
            self.config.wavelet_type,
            sampling_period=1  # Assuming monthly data
        )
        
        return coefficients, frequencies, scales
    
    def _perform_dwt(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Perform Discrete Wavelet Transform (multi-level decomposition).
        
        Returns:
            List of coefficients [cA_n, cD_n, cD_n-1, ..., cD_1]
            where cA is approximation and cD are details
        """
        # Determine maximum decomposition level
        max_level = pywt.dwt_max_level(len(signal), self.config.dwt_wavelet)
        level = min(self.config.dwt_level, max_level)
        
        # Perform decomposition
        coeffs = pywt.wavedec(signal, self.config.dwt_wavelet, level=level)
        
        return coeffs
    
    def _analyze_dwt_levels(self, dwt_coeffs: List[np.ndarray], 
                           original_signal: np.ndarray) -> List[Pattern]:
        """
        Analyze DWT decomposition levels to find patterns.
        
        Each level corresponds to different frequency bands.
        """
        patterns = []
        
        # Skip the approximation coefficients (first element)
        # Analyze detail coefficients
        for level, detail_coeffs in enumerate(dwt_coeffs[1:], start=1):
            if len(detail_coeffs) < 3:
                continue
            
            # Estimate frequency band for this level
            # For monthly data: level 1 = 0.25-0.5, level 2 = 0.125-0.25, etc.
            freq_high = 0.5 / (2 ** (level - 1))
            freq_low = 0.5 / (2 ** level)
            freq_center = (freq_high + freq_low) / 2
            
            # Calculate energy in this level
            energy = np.sum(detail_coeffs ** 2) / len(detail_coeffs)
            
            # Only create pattern if significant energy
            total_energy = np.sum(original_signal ** 2) / len(original_signal)
            relative_energy = energy / (total_energy + 1e-10)
            
            if relative_energy > 0.05:  # At least 5% of total energy
                pattern = Pattern(
                    frequency=freq_center,
                    period=1/freq_center,
                    amplitude=np.sqrt(energy),
                    confidence=min(relative_energy * 2, 1.0),  # Scale up for visibility
                    source_method="Wavelet-DWT",
                    metadata={
                        'dwt_level': level,
                        'freq_band': (freq_low, freq_high),
                        'energy': float(energy),
                        'relative_energy': float(relative_energy),
                        'n_coeffs': len(detail_coeffs)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def get_wavelet_families(self) -> Dict[str, List[str]]:
        """Get available wavelet families and their members."""
        families = {}
        for family in pywt.families():
            families[family] = pywt.wavelist(family)
        return families
    
    def reconstruct_pattern(self, scale_idx: int, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal component from a specific scale.
        
        Args:
            scale_idx: Index of the scale to reconstruct
            coefficients: CWT coefficients
            
        Returns:
            Reconstructed signal component
        """
        # Zero out all scales except the one we want
        reconstructed_coeffs = np.zeros_like(coefficients)
        reconstructed_coeffs[scale_idx, :] = coefficients[scale_idx, :]
        
        # Inverse CWT (approximation)
        # Note: PyWavelets doesn't have built-in iCWT, so we approximate
        reconstructed = np.real(reconstructed_coeffs[scale_idx, :])
        
        return reconstructed