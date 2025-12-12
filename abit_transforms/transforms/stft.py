
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, get_window
from typing import List, Tuple, Dict, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Transform, Pattern
from config import TransformConfig, DEFAULT_CONFIG

class STFTTransform(Transform):
    """
    Short-Time Fourier Transform implementation.
    
    STFT provides time-frequency analysis with fixed time-frequency resolution.
    Good for signals with stable frequency content.
    """
    
    def get_name(self) -> str:
        return "STFT"
    
    def analyze(
        self,
        signal_data: np.ndarray,
        window_size: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[Pattern], Dict]:
        """
        Perform STFT analysis on signal.
        
        Args:
            signal_data: Input time series
            window_size: Optional custom window size
            
        Returns:
            patterns: List of discovered patterns
            artifacts: Dict containing STFT results
        """
        patterns = []
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal_data)
        
        # Determine window size
        if window_size is not None:
            nperseg = window_size
        elif self.config.stft_adaptive_window and 'target_periods' in kwargs:
            # Adaptive window based on target periods
            nperseg = self._optimize_window_size(kwargs['target_periods'])
        else:
            nperseg = self.config.stft_nperseg
        
        # Ensure window size is valid
        nperseg = min(nperseg, len(processed_signal) // 2)
        noverlap = int(nperseg * 0.75)
        
        # Perform STFT
        f, t, Zxx = signal.stft(
            processed_signal,
            fs=self.config.stft_fs,
            window=self.config.stft_window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend='constant',
            return_onesided=True,
            boundary='zeros',
            padded=True
        )
        
        # Calculate magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Average power spectrum
        avg_power = np.mean(magnitude ** 2, axis=1)
        
        # Find peaks in frequency domain
        if len(avg_power) > 1:
            # Skip DC component (index 0)
            peaks, properties = find_peaks(
                avg_power[1:],
                height=np.max(avg_power[1:]) * 0.15,
                distance=1,
                prominence=np.max(avg_power[1:]) * 0.1
            )
            
            # Create patterns from peaks
            for i, peak_idx in enumerate(peaks):
                freq_idx = peak_idx + 1  # Account for skipped DC
                freq = f[freq_idx]
                
                if freq > 0 and freq < 0.5:  # Valid frequency range
                    # Analyze time-varying behavior
                    freq_magnitude = magnitude[freq_idx, :]
                    
                    # Calculate statistics
                    mean_amp = np.mean(freq_magnitude)
                    std_amp = np.std(freq_magnitude)
                    max_amp = np.max(freq_magnitude)
                    
                    # Check if pattern is time-varying or stable
                    is_time_varying = std_amp > 0.3 * mean_amp
                    
                    # Calculate pattern persistence (how often it's present)
                    threshold = mean_amp * 0.5
                    persistence = np.sum(freq_magnitude > threshold) / len(freq_magnitude)
                    
                    # Get phase coherence as additional confidence measure
                    phase_coherence = self._calculate_phase_coherence(phase[freq_idx, :])
                    
                    pattern = Pattern(
                        frequency=freq,
                        period=1/freq,
                        amplitude=mean_amp,
                        confidence=avg_power[freq_idx] / np.max(avg_power[1:]),
                        source_method="STFT",
                        metadata={
                            'avg_power': float(avg_power[freq_idx]),
                            'std_amplitude': float(std_amp),
                            'max_amplitude': float(max_amp),
                            'time_varying': bool(is_time_varying),
                            'persistence': float(persistence),
                            'phase_coherence': float(phase_coherence),
                            'window_size': int(nperseg),
                            'frequency_resolution': float(f[1] - f[0]),
                            'time_resolution': float(t[1] - t[0]) if len(t) > 1 else 0
                        }
                    )
                    
                    # Boost confidence for highly persistent patterns
                    if persistence > 0.8:
                        pattern.confidence = min(pattern.confidence * 1.2, 1.0)
                    
                    patterns.append(pattern)
        
        # Also look for trends in low-frequency components
        if len(f) > 0 and f[0] == 0:  # DC component
            dc_component = magnitude[0, :]
            if len(dc_component) > 1:
                # Check for trend
                trend_slope = (dc_component[-1] - dc_component[0]) / len(dc_component)
                if abs(trend_slope) > np.mean(dc_component) * 0.01:
                    trend_pattern = Pattern(
                        frequency=0,
                        period=np.inf,
                        amplitude=np.mean(dc_component),
                        confidence=0.8,
                        source_method="STFT",
                        metadata={
                            'type': 'trend',
                            'slope': float(trend_slope),
                            'dc_mean': float(np.mean(dc_component)),
                            'dc_std': float(np.std(dc_component))
                        }
                    )
                    patterns.append(trend_pattern)
        
        # Store artifacts
        artifacts = {
            'frequencies': f,
            'times': t,
            'stft_matrix': Zxx,
            'magnitude': magnitude,
            'phase': phase,
            'avg_power_spectrum': avg_power,
            'window_size': nperseg,
            'overlap': noverlap,
            'peaks': peaks if 'peaks' in locals() else []
        }
        
        return patterns, artifacts
    
    def _optimize_window_size(self, target_periods: List[float]) -> int:
        """
        Optimize STFT window size based on target periods.
        
        Window should be at least 2x the largest period for good
        frequency resolution.
        
        Args:
            target_periods: List of periods to detect
            
        Returns:
            Optimal window size in samples
        """
        if not target_periods:
            return self.config.stft_nperseg
        
        # Use 2-3x the median period for balanced resolution
        median_period = np.median(target_periods)
        optimal_size = int(median_period * 2.5)
        
        # Clip to reasonable range
        min_size = 12  # At least 1 year for monthly data
        max_size = 60  # At most 5 years
        
        return np.clip(optimal_size, min_size, max_size)
    
    def _calculate_phase_coherence(self, phase: np.ndarray) -> float:
        """
        Calculate phase coherence as a measure of signal stability.
        
        High coherence indicates a stable oscillation.
        Low coherence indicates noise or time-varying frequency.
        
        Args:
            phase: Phase values over time
            
        Returns:
            Coherence value between 0 and 1
        """
        if len(phase) < 2:
            return 0.0
        
        # Unwrap phase
        unwrapped = np.unwrap(phase)
        
        # Calculate phase differences
        phase_diff = np.diff(unwrapped)
        
        # Coherence is inverse of phase difference variance
        if np.std(phase_diff) > 0:
            coherence = 1.0 / (1.0 + np.std(phase_diff))
        else:
            coherence = 1.0
        
        return coherence
    
    def get_spectrogram(
        self,
        signal_data: np.ndarray,
        window_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get spectrogram for visualization.
        
        Returns:
            f: Frequency array
            t: Time array
            Sxx: Power spectral density
        """
        nperseg = window_size or self.config.stft_nperseg
        noverlap = int(nperseg * 0.75)
        
        f, t, Sxx = signal.spectrogram(
            signal_data,
            fs=self.config.stft_fs,
            window=self.config.stft_window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend='constant',
            return_onesided=True,
            mode='psd'
        )
        
        return f, t, Sxx
    
    def reconstruct_frequency(
        self,
        stft_matrix: np.ndarray,
        freq_idx: int,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct signal component at specific frequency.
        
        Args:
            stft_matrix: STFT coefficient matrix
            freq_idx: Index of frequency to reconstruct
            times: Time array from STFT
            
        Returns:
            Reconstructed signal component
        """
        # Extract coefficients for specific frequency
        freq_coeffs = stft_matrix[freq_idx, :]
        
        # Inverse STFT for single frequency
        # This is an approximation since we're only using one frequency
        reconstructed = np.real(freq_coeffs) * np.cos(2 * np.pi * freq_idx * times)
        reconstructed += np.imag(freq_coeffs) * np.sin(2 * np.pi * freq_idx * times)
        
        return reconstructed