import numpy as np
from scipy.signal import hilbert, savgol_filter
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Transform, Pattern
from config import TransformConfig, DEFAULT_CONFIG

class HHTTransform(Transform):
    """
    Hilbert-Huang Transform implementation.
    
    HHT performs Empirical Mode Decomposition (EMD) to extract
    Intrinsic Mode Functions (IMFs), then applies Hilbert transform
    to get instantaneous frequency and amplitude.
    """
    
    def get_name(self) -> str:
        return "HHT"
    
    def analyze(self, signal: np.ndarray, signal_type: str = 'unknown', **kwargs) -> Tuple[List[Pattern], Dict]:
        """
        Perform HHT analysis on signal.
        
        Args:
            signal: Input time series
            signal_type: 'majority', 'minority', or 'unknown'
            
        Returns:
            patterns: List of discovered patterns
            artifacts: Dict containing IMFs and instantaneous frequencies
        """
        patterns = []
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal)
        
        # Apply minority amplification if needed
        if signal_type == 'minority':
            processed_signal = processed_signal * self.config.minority_amplification
        
        # Perform EMD
        imfs = self._empirical_mode_decomposition(processed_signal)
        
        # Analyze each IMF
        inst_freqs = []
        inst_amps = []
        
        for i, imf in enumerate(imfs[:-1]):  # Skip residual/trend
            # Estimate frequency from zero crossings
            freq, period = self._estimate_imf_frequency(imf)
            
            if freq > 0 and freq < 0.5:  # Valid frequency range (up to Nyquist)
                # Calculate energy
                energy = np.sum(imf**2) / len(imf)
                
                # Hilbert transform for instantaneous properties
                analytic = hilbert(imf)
                amplitude = np.abs(analytic)
                phase = np.unwrap(np.angle(analytic))
                
                # Instantaneous frequency
                inst_freq = np.diff(phase) / (2.0 * np.pi)
                inst_freq = np.append(inst_freq, inst_freq[-1])  # Pad to maintain length
                
                inst_freqs.append(inst_freq)
                inst_amps.append(amplitude)
                
                # Create pattern
                pattern = Pattern(
                    frequency=freq,
                    period=period,
                    amplitude=np.mean(amplitude),
                    confidence=self._calculate_confidence(energy, imfs),
                    source_method="HHT",
                    metadata={
                        'imf_index': i,
                        'energy': float(energy),
                        'inst_freq_mean': float(np.mean(inst_freq)),
                        'inst_freq_std': float(np.std(inst_freq)),
                        'is_stationary': float(np.std(inst_freq)) < 0.1 * float(np.mean(inst_freq))
                    }
                )
                patterns.append(pattern)
        
        # Add trend information (last IMF is residual/trend)
        trend = imfs[-1]
        if signal_type == 'minority':
            trend = trend / self.config.minority_amplification
        
        # Calculate trend characteristics
        trend_slope = (trend[-1] - trend[0]) / len(trend) if len(trend) > 0 else 0
        
        trend_pattern = Pattern(
            frequency=0,
            period=np.inf,
            amplitude=float(np.mean(np.abs(trend))),
            confidence=0.95,  # Trends are usually reliable
            source_method="HHT",
            metadata={
                'type': 'trend',
                'slope': float(trend_slope),
                'is_increasing': trend_slope > 0,
                'trend_strength': float(np.abs(trend_slope) / (np.std(signal) + 1e-10))
            }
        )
        patterns.append(trend_pattern)
        
        # Create Hilbert spectrum
        hilbert_spectrum = self._compute_hilbert_spectrum(imfs[:-1], inst_freqs, inst_amps)
        
        artifacts = {
            'imfs': imfs,
            'inst_frequencies': inst_freqs,
            'inst_amplitudes': inst_amps,
            'trend': trend,
            'hilbert_spectrum': hilbert_spectrum,
            'n_imfs': len(imfs)
        }
        
        return patterns, artifacts
    
    def _empirical_mode_decomposition(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Perform EMD to extract IMFs.
        
        Args:
            signal: Input signal
            
        Returns:
            List of IMFs, with the last being the residual/trend
        """
        imfs = []
        residual = signal.copy()
        
        for imf_num in range(self.config.hht_max_imfs):
            # Extract one IMF
            imf = self._extract_imf(residual)
            
            if imf is None:
                break
            
            # Check if IMF is valid
            if np.std(imf) < np.std(signal) * 0.001:
                break
            
            imfs.append(imf)
            residual = residual - imf
            
            # Stop if residual is monotonic or too small
            if self._is_monotonic(residual) or np.std(residual) < np.std(signal) * 0.05:
                break
        
        # Add final residual as trend
        imfs.append(residual)
        
        return imfs
    
    def _extract_imf(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract one IMF using the sifting process.
        
        Args:
            signal: Input signal
            
        Returns:
            Extracted IMF or None if extraction fails
        """
        if len(signal) < 4:
            return None
        
        imf = signal.copy()
        
        for _ in range(self.config.hht_sift_iterations):
            # Find extrema
            maxima, minima = self._find_extrema(imf)
            
            # Check if we have enough extrema
            if len(maxima) < 2 or len(minima) < 2:
                break
            
            # Compute envelope mean
            try:
                if self.config.hht_envelope_method == 'spline':
                    mean_env = self._compute_spline_envelope_mean(imf, maxima, minima)
                else:  # savgol
                    mean_env = self._compute_savgol_envelope_mean(imf)
                
                # Update IMF
                imf_new = imf - mean_env
                
                # Check convergence
                sd = np.sum((imf - imf_new) ** 2) / np.sum(imf ** 2)
                if sd < self.config.hht_stopping_criterion:
                    break
                
                imf = imf_new
                
            except Exception:
                # If envelope computation fails, return current IMF
                break
        
        return imf
    
    def _find_extrema(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find local maxima and minima in the signal."""
        maxima = []
        minima = []
        
        # Find local extrema
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima.append(i)
            elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                minima.append(i)
        
        return np.array(maxima), np.array(minima)
    
    def _compute_spline_envelope_mean(self, signal: np.ndarray, 
                                      maxima: np.ndarray, 
                                      minima: np.ndarray) -> np.ndarray:
        """Compute envelope mean using cubic splines."""
        t = np.arange(len(signal))
        
        # Add boundary points if needed
        if maxima[0] > 0:
            maxima = np.concatenate([[0], maxima])
        if maxima[-1] < len(signal) - 1:
            maxima = np.concatenate([maxima, [len(signal) - 1]])
            
        if minima[0] > 0:
            minima = np.concatenate([[0], minima])
        if minima[-1] < len(signal) - 1:
            minima = np.concatenate([minima, [len(signal) - 1]])
        
        # Create spline envelopes
        upper_spline = CubicSpline(maxima, signal[maxima], bc_type='natural')
        lower_spline = CubicSpline(minima, signal[minima], bc_type='natural')
        
        # Compute mean
        upper_env = upper_spline(t)
        lower_env = lower_spline(t)
        
        return (upper_env + lower_env) / 2
    
    def _compute_savgol_envelope_mean(self, signal: np.ndarray) -> np.ndarray:
        """Compute envelope mean using Savitzky-Golay filter (simpler/faster)."""
        # Window length should be odd and less than signal length
        window_length = min(11, len(signal) // 2 * 2 - 1)
        
        if window_length < 5:
            # Signal too short for savgol
            return np.zeros_like(signal)
        
        # Apply savgol filter as approximation of envelope mean
        try:
            mean_env = savgol_filter(signal, window_length, 3)
        except:
            mean_env = np.zeros_like(signal)
        
        return mean_env
    
    def _is_monotonic(self, signal: np.ndarray) -> bool:
        """Check if signal is monotonic (all increasing or all decreasing)."""
        diff = np.diff(signal)
        return np.all(diff >= 0) or np.all(diff <= 0)
    
    def _estimate_imf_frequency(self, imf: np.ndarray) -> Tuple[float, float]:
        """
        Estimate dominant frequency of IMF using zero-crossing rate.
        
        Returns:
            frequency: Dominant frequency (cycles per sample)
            period: Corresponding period (samples)
        """
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
        
        if zero_crossings > 2:
            # Period is twice the average distance between zero crossings
            period = 2 * len(imf) / zero_crossings
            frequency = 1 / period
        else:
            # No clear oscillation
            period = np.inf
            frequency = 0
        
        return frequency, period
    
    def _calculate_confidence(self, energy: float, imfs: List[np.ndarray]) -> float:
        """
        Calculate confidence score for a pattern based on its energy
        relative to other IMFs.
        """
        if len(imfs) <= 1:
            return 0.5
        
        # Calculate energies of all IMFs except residual
        all_energies = []
        for imf in imfs[:-1]:
            imf_energy = np.sum(imf**2) / len(imf)
            all_energies.append(imf_energy)
        
        if not all_energies:
            return 0.5
        
        # Relative energy as confidence
        max_energy = np.max(all_energies)
        if max_energy > 0:
            relative_energy = energy / max_energy
            return min(relative_energy, 1.0)
        
        return 0.5
    
    def _compute_hilbert_spectrum(self, imfs: List[np.ndarray],
                                  inst_freqs: List[np.ndarray],
                                  inst_amps: List[np.ndarray]) -> np.ndarray:
        """
        Compute Hilbert spectrum (time-frequency representation).
        
        Returns:
            2D array representing time-frequency energy distribution
        """
        if not imfs:
            return np.array([[]])
        
        # Define time and frequency grids
        n_time = len(imfs[0])
        n_freq = 50  # Frequency bins
        
        # Frequency range (0 to Nyquist)
        freq_grid = np.linspace(0, 0.5, n_freq)
        
        # Initialize spectrum
        spectrum = np.zeros((n_freq, n_time))
        
        # Fill spectrum with instantaneous frequency and amplitude data
        for inst_freq, inst_amp in zip(inst_freqs, inst_amps):
            for t in range(min(len(inst_freq), n_time)):
                if 0 <= inst_freq[t] <= 0.5:  # Valid frequency range
                    # Find nearest frequency bin
                    freq_idx = np.argmin(np.abs(freq_grid - inst_freq[t]))
                    # Add amplitude to spectrum
                    spectrum[freq_idx, t] += inst_amp[t] if t < len(inst_amp) else 0
        
        return spectrum
