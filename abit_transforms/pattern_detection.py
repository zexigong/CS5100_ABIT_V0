#!/usr/bin/env python3
"""
Enhanced Pattern Detection from Transform Features
Detects: Linear/Non-linear Trends, Time-varying Seasonality, Regime Changes, Volatility
Fully leverages HHT and Wavelet capabilities for complex pattern detection
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BASIC TREND DETECTION (Linear as baseline)
# ============================================================================

def detect_slow_trend(signal_data: pd.Series, threshold_slope: float = 0.05) -> Dict[str, Any]:
    """
    Detects linear trends using linear regression (baseline method).
    
    Args:
        signal_data: Time series data (indexed by datetime)
        threshold_slope: Minimum absolute slope to consider it a trend
    
    Returns:
        Dict with trend_type, percentage_change, slope, start_date, end_date
    """
    signal_data = pd.Series(np.asarray(signal_data))

    x = np.arange(len(signal_data))
    y = signal_data.values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    start_value = intercept
    end_value = slope * (len(signal_data) - 1) + intercept
    
    if start_value != 0:
        percentage_change = ((end_value - start_value) / abs(start_value)) * 100
    else:
        percentage_change = 0
    
    if abs(slope) < threshold_slope:
        trend_type = 'no_trend'
    elif slope > 0:
        trend_type = 'slow_trend_increase'
    else:
        trend_type = 'slow_trend_decrease'
    
    return {
        'pattern_type': trend_type,
        'percentage_change': percentage_change,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'start_date': signal_data.index[0],
        'end_date': signal_data.index[-1],
        'confidence': 'high' if r_value**2 > 0.7 and p_value < 0.05 else 'medium' if r_value**2 > 0.4 else 'low'
    }

# ============================================================================
# COMPLEX TREND DETECTION (Non-linear)
# ============================================================================

def detect_exponential_trend(signal_data: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Detects exponential growth/decay: y = a * exp(b*x)
    """
    signal_data = pd.Series(np.asarray(signal_data))

    x = np.arange(len(signal_data))
    y = signal_data.values
    
    # Avoid log of negative/zero values
    if np.any(y <= 0):
        y = y - np.min(y) + 1
    
    log_y = np.log(y)
    
    try:
        slope, intercept, r_value, p_value, _ = stats.linregress(x, log_y)
    except:
        return None
    
    # Check if exponential fits better than linear
    if r_value**2 > 0.75 and abs(slope) > 0.001:
        # Calculate growth rate
        growth_rate_per_period = (np.exp(slope) - 1) * 100
        
        # Calculate doubling/halving time
        if slope > 0:
            doubling_time = np.log(2) / slope
            time_interpretation = f"Doubles every {doubling_time:.1f} periods"
        else:
            halving_time = np.log(2) / abs(slope)
            time_interpretation = f"Halves every {halving_time:.1f} periods"
        
        # Total percentage change
        initial = np.exp(intercept)
        final = np.exp(slope * (len(x) - 1) + intercept)
        total_change = ((final - initial) / initial) * 100
        
        return {
            'pattern_type': 'exponential_growth' if slope > 0 else 'exponential_decay',
            'growth_rate_per_period': growth_rate_per_period,
            'total_percentage_change': total_change,
            'doubling_halving_time': doubling_time if slope > 0 else halving_time,
            'time_interpretation': time_interpretation,
            'r_squared': r_value**2,
            'p_value': p_value,
            'start_date': signal_data.index[0],
            'end_date': signal_data.index[-1],
            'confidence': 'high' if r_value**2 > 0.85 else 'medium'
        }
    
    return None

def detect_polynomial_trend(signal_data: pd.Series, max_degree: int = 3) -> Optional[Dict[str, Any]]:
    """
    Detects polynomial trends (accelerating/decelerating growth).
    """
    signal_data = pd.Series(np.asarray(signal_data))

    x = np.arange(len(signal_data))
    y = signal_data.values
    
    best_fit = None
    best_r2 = 0
    
    for degree in range(2, max_degree + 1):
        try:
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)
            
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > best_r2 and r2 > 0.75:
                best_r2 = r2
                
                # Calculate total percentage change
                initial = poly(0)
                final = poly(len(x) - 1)
                total_change = ((final - initial) / abs(initial)) * 100
                
                # Interpret the pattern
                if degree == 2:
                    a, b, c = coeffs
                    if a > 0:
                        interpretation = "Accelerating growth (growth rate is increasing)"
                        pattern_name = 'accelerating_growth'
                    else:
                        interpretation = "Decelerating growth (growth rate is slowing)"
                        pattern_name = 'decelerating_growth'
                else:
                    interpretation = f"Complex polynomial trend (degree {degree})"
                    pattern_name = f'polynomial_trend_degree_{degree}'
                
                best_fit = {
                    'pattern_type': pattern_name,
                    'degree': degree,
                    'coefficients': coeffs.tolist(),
                    'total_percentage_change': total_change,
                    'interpretation': interpretation,
                    'r_squared': r2,
                    'start_date': signal_data.index[0],
                    'end_date': signal_data.index[-1],
                    'confidence': 'high' if r2 > 0.85 else 'medium'
                }
        except:
            continue
    
    return best_fit

def detect_regime_changes(signal_data: pd.Series, min_segment_length: int = 12) -> Optional[Dict[str, Any]]:
    """
    Detects structural breaks/regime changes using change point detection.
    Falls back to simple method if ruptures library is not available.
    """
    try:
        import ruptures as rpt
        
        # Use PELT algorithm for change point detection
        model = "rbf"
        algo = rpt.Pelt(model=model, min_size=min_segment_length).fit(signal_data.values)
        breakpoints = algo.predict(pen=10)
        
    except ImportError:
        # Fallback: simple variance-based change detection
        breakpoints = detect_variance_changes(signal_data.values, min_segment_length)
    
    if len(breakpoints) > 1:  # At least one change point
        regimes = []
        start = 0
        
        for bp in breakpoints[:-1]:
            segment = signal_data.iloc[start:bp]
            
            if len(segment) < 3:
                start = bp
                continue
            
            # Analyze each regime
            x_seg = np.arange(len(segment))
            trend = stats.linregress(x_seg, segment.values)
            
            pct_change = ((segment.iloc[-1] - segment.iloc[0]) / abs(segment.iloc[0])) * 100 if segment.iloc[0] != 0 else 0
            
            regimes.append({
                'start_date': segment.index[0],
                'end_date': segment.index[-1],
                'trend_slope': trend.slope,
                'trend_type': 'increasing' if trend.slope > 0 else 'decreasing',
                'percentage_change': pct_change,
                'r_squared': trend.rvalue**2
            })
            
            start = bp
        
        if len(regimes) > 0:
            return {
                'pattern_type': 'regime_change',
                'num_regimes': len(regimes),
                'regimes': regimes,
                'interpretation': f"Detected {len(regimes)} distinct regimes with different growth patterns",
                'confidence': 'high' if len(regimes) >= 2 else 'medium'
            }
    
    return None

def detect_variance_changes(signal: np.ndarray, min_length: int = 12) -> List[int]:
    """
    Simple variance-based change point detection (fallback method).
    """
    n = len(signal)
    breakpoints = []
    
    for i in range(min_length, n - min_length, min_length // 2):
        left = signal[max(0, i - min_length):i]
        right = signal[i:min(n, i + min_length)]
        
        if len(left) > 0 and len(right) > 0:
            var_left = np.var(left)
            var_right = np.var(right)
            
            # Significant variance change
            if abs(var_left - var_right) > 0.5 * (var_left + var_right):
                breakpoints.append(i)
    
    breakpoints.append(n)  # Add end point
    return breakpoints

# ============================================================================
# TIME-VARYING SEASONALITY DETECTION
# ============================================================================

def detect_evolving_seasonality(imf: np.ndarray, time_index: pd.DatetimeIndex) -> Optional[Dict[str, Any]]:
    """
    Detects time-varying seasonality using Hilbert transform for instantaneous frequency.
    """
    if len(imf) < 10:
        return None
    
    try:
        # Hilbert transform
        analytic_signal = hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi)
        
        # Avoid division by zero
        instantaneous_freq = np.clip(instantaneous_freq, 1e-10, None)
        instantaneous_period = 1.0 / instantaneous_freq
        
        # Check if period changes over time
        x = np.arange(len(instantaneous_period))
        period_trend = stats.linregress(x, instantaneous_period)
        
        # Significant period evolution
        if abs(period_trend.slope) > 0.01 and period_trend.pvalue < 0.1:
            initial_period = np.median(instantaneous_period[:len(instantaneous_period)//4])
            final_period = np.median(instantaneous_period[-len(instantaneous_period)//4:])
            
            period_change_pct = ((final_period - initial_period) / initial_period) * 100
            
            if period_trend.slope > 0:
                evolution = "lengthening"
                interpretation = f"Seasonal cycles slowing down: period increased {abs(period_change_pct):.1f}%"
            else:
                evolution = "shortening"
                interpretation = f"Seasonal cycles accelerating: period decreased {abs(period_change_pct):.1f}%"
            
            return {
                'pattern_type': 'time_varying_seasonality',
                'evolution': evolution,
                'initial_period': float(initial_period),
                'final_period': float(final_period),
                'period_change_percent': period_change_pct,
                'interpretation': interpretation,
                'start_date': time_index[0],
                'end_date': time_index[-1],
                'confidence': 'high' if abs(period_trend.slope) > 0.05 else 'medium'
            }
        
        # Check for amplitude modulation
        instantaneous_amplitude = np.abs(analytic_signal)
        amp_trend = stats.linregress(np.arange(len(instantaneous_amplitude)), instantaneous_amplitude)
        
        if abs(amp_trend.slope) > 0.01 and amp_trend.pvalue < 0.1:
            amp_change = ((instantaneous_amplitude[-1] - instantaneous_amplitude[0]) / instantaneous_amplitude[0]) * 100
            
            return {
                'pattern_type': 'amplitude_modulated_seasonality',
                'amplitude_trend': 'increasing' if amp_trend.slope > 0 else 'decreasing',
                'amplitude_change_percent': amp_change,
                'interpretation': f"Seasonal strength {'increasing' if amp_trend.slope > 0 else 'decreasing'} by {abs(amp_change):.1f}%",
                'start_date': time_index[0],
                'end_date': time_index[-1],
                'confidence': 'medium'
            }
    
    except Exception as e:
        return None
    
    return None

# ============================================================================
# SEASONALITY DETECTION (Standard FFT-based)
# ============================================================================

def detect_seasonality(signal_data: pd.Series, min_period: int = 3, max_period: int = None) -> List[Dict[str, Any]]:
    """
    Detects seasonal patterns using FFT (Fourier Transform).
    """
    
    signal_data = pd.Series(np.asarray(signal_data))
    if max_period is None:
        max_period = len(signal_data) // 2
    
    detrended = signal.detrend(signal_data.values)
    
    n = len(detrended)
    yf = fft(detrended)
    xf = fftfreq(n, 1)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])
    
    peaks, properties = signal.find_peaks(power, height=np.std(power), distance=2)
    
    seasonal_patterns = []
    
    for peak_idx in peaks:
        freq = xf[peak_idx]
        if freq == 0:
            continue
            
        period = 1 / freq
        
        if period < min_period or period > max_period:
            continue
        
        strength = power[peak_idx] / np.max(power)
        
        period_months = int(round(period))
        if 11 <= period_months <= 13:
            period_description = 'Annual'
        elif 5 <= period_months <= 7:
            period_description = 'Semi-Annual'
        elif 2 <= period_months <= 4:
            period_description = 'Quarterly'
        else:
            period_description = f'{period_months}-Month Cycle'
        
        seasonal_patterns.append({
            'pattern_type': 'seasonal_periodic',
            'period': period,
            'period_months': period_months,
            'period_description': period_description,
            'frequency': freq,
            'strength': strength,
            'start_date': signal_data.index[0],
            'end_date': signal_data.index[-1],
            'confidence': 'high' if strength > 0.7 else 'medium' if strength > 0.4 else 'low'
        })
    
    seasonal_patterns.sort(key=lambda x: x['strength'], reverse=True)
    
    return seasonal_patterns

# ============================================================================
# VOLATILITY DETECTION
# ============================================================================

def detect_volatility(signal_data: pd.Series, window: int = 6) -> Dict[str, Any]:
    """
    Detects high volatility periods using rolling standard deviation.
    """
    signal_data = pd.Series(np.asarray(signal_data))

    rolling_std = signal_data.rolling(window=window, center=True).std()
    mean_volatility = rolling_std.mean()
    
    threshold = mean_volatility + 1.5 * rolling_std.std()
    high_volatility_mask = rolling_std > threshold
    
    volatility_periods = []
    in_volatile_period = False
    start_idx = None
    
    for idx, is_volatile in enumerate(high_volatility_mask):
        if is_volatile and not in_volatile_period:
            start_idx = idx
            in_volatile_period = True
        elif not is_volatile and in_volatile_period:
            volatility_periods.append({
                'start_date': signal_data.index[start_idx],
                'end_date': signal_data.index[idx-1],
                'max_volatility': rolling_std.iloc[start_idx:idx].max()
            })
            in_volatile_period = False
    
    if in_volatile_period and start_idx is not None:
        volatility_periods.append({
            'start_date': signal_data.index[start_idx],
            'end_date': signal_data.index[-1],
            'max_volatility': rolling_std.iloc[start_idx:].max()
        })
    
    return {
        'pattern_type': 'high_volatility',
        'mean_volatility': mean_volatility,
        'max_volatility': rolling_std.max(),
        'volatile_periods': volatility_periods,
        'overall_start': signal_data.index[0],
        'overall_end': signal_data.index[-1],
        'confidence': 'high' if len(volatility_periods) > 0 else 'low'
    }

# ============================================================================
# HHT-SPECIFIC PATTERN DETECTION (Enhanced)
# ============================================================================

def detect_patterns_from_hht(imfs: np.ndarray, time_index: pd.DatetimeIndex) -> List[Dict[str, Any]]:
    """
    Enhanced pattern detection from HHT IMFs - detects complex non-linear patterns.
    """
    patterns = []
    
    if imfs.shape[1] == 0:
        return patterns
    
    # Analyze trend (last IMF/residual) for non-linearity
    trend = pd.Series(imfs[:, -1], index=time_index)
    
    # Try exponential first
    exp_pattern = detect_exponential_trend(trend)
    if exp_pattern and exp_pattern['r_squared'] > 0.75:
        patterns.append(exp_pattern)
    else:
        # Try polynomial
        poly_pattern = detect_polynomial_trend(trend)
        if poly_pattern and poly_pattern['r_squared'] > 0.75:
            patterns.append(poly_pattern)
        else:
            # Try regime changes
            regime_pattern = detect_regime_changes(trend)
            if regime_pattern:
                patterns.append(regime_pattern)
            else:
                # Fallback to linear
                linear_trend = detect_slow_trend(trend)
                if linear_trend['pattern_type'] != 'no_trend':
                    patterns.append(linear_trend)
    
    # Analyze oscillatory IMFs for time-varying seasonality
    for i in range(max(1, imfs.shape[1] - 3), imfs.shape[1] - 1):
        if i < 0 or i >= imfs.shape[1]:
            continue
            
        imf = imfs[:, i]
        
        # Check for evolving patterns
        evolving = detect_evolving_seasonality(imf, time_index)
        if evolving:
            patterns.append(evolving)
        else:
            # Try standard seasonality detection
            imf_series = pd.Series(imf, index=time_index)
            seasonal = detect_seasonality(imf_series)
            if seasonal:
                patterns.extend(seasonal[:1])
    
    # Volatility clustering in high-frequency IMFs
    if imfs.shape[1] > 2:
        high_freq_imf = imfs[:, 0]
        squared = high_freq_imf ** 2
        
        # Check for volatility clustering (ARCH effect)
        if len(squared) > 10:
            acf = np.correlate(squared - squared.mean(), squared - squared.mean(), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]  # Normalize
            
            if np.any(acf[1:min(10, len(acf))] > 0.3):
                patterns.append({
                    'pattern_type': 'volatility_clustering',
                    'interpretation': 'High volatility periods cluster together (ARCH effect)',
                    'start_date': time_index[0],
                    'end_date': time_index[-1],
                    'confidence': 'medium'
                })
    
    return patterns

# ============================================================================
# WAVELET-SPECIFIC PATTERN DETECTION (Enhanced)
# ============================================================================

def detect_patterns_from_wavelet(coefficients: np.ndarray, time_index: pd.DatetimeIndex) -> List[Dict[str, Any]]:
    """
    Enhanced pattern detection from Wavelet coefficients - multi-scale analysis.
    """
    patterns = []
    
    if coefficients is None or coefficients.ndim < 2 or coefficients.shape[1] == 0:
        return patterns
    
    # Analyze coarse scale (trend)
    coarse_trend = pd.Series(coefficients[:, -1], index=time_index)
    
    # Try exponential
    exp_pattern = detect_exponential_trend(coarse_trend)
    if exp_pattern and exp_pattern['r_squared'] > 0.75:
        patterns.append(exp_pattern)
    else:
        # Try polynomial
        poly_pattern = detect_polynomial_trend(coarse_trend)
        if poly_pattern and poly_pattern['r_squared'] > 0.75:
            patterns.append(poly_pattern)
        else:
            # Linear trend
            linear_trend = detect_slow_trend(coarse_trend)
            if linear_trend['pattern_type'] != 'no_trend':
                patterns.append(linear_trend)
    
    # Multi-scale seasonality analysis
    if coefficients.shape[1] > 2:
        mid_scale = pd.Series(coefficients[:, coefficients.shape[1]//2], index=time_index)
        seasonal = detect_seasonality(mid_scale)
        patterns.extend(seasonal[:2])
    
    # Detect abrupt changes (singularities) using coefficient magnitudes
    singularities = detect_wavelet_singularities(coefficients, time_index)
    if singularities:
        patterns.append(singularities)
    
    # Fine scale volatility
    if coefficients.shape[1] > 1:
        fine_scale = pd.Series(coefficients[:, 0], index=time_index)
        volatility_pattern = detect_volatility(fine_scale)
        if volatility_pattern['confidence'] != 'low':
            patterns.append(volatility_pattern)
    
    return patterns

def detect_wavelet_singularities(coefficients: np.ndarray, time_index: pd.DatetimeIndex) -> Optional[Dict[str, Any]]:
    """
    Detect structural breaks using wavelet modulus maxima.
    """
    singularities = []
    
    if coefficients.shape[0] < 3 or coefficients.shape[1] < 3:
        return None
    
    # Find points that are local maxima across scales
    for time_idx in range(1, coefficients.shape[0] - 1):
        magnitude = np.abs(coefficients[time_idx, :])
        
        # Check if significant across multiple scales
        if np.mean(magnitude) > 1.5 * np.mean(np.abs(coefficients)):
            singularities.append({
                'time': time_index[time_idx],
                'magnitude': np.max(magnitude)
            })
    
    if len(singularities) > 0:
        return {
            'pattern_type': 'structural_breaks',
            'num_breaks': len(singularities),
            'breaks': singularities[:5],  # Top 5
            'interpretation': f"Detected {len(singularities)} potential structural changes",
            'confidence': 'medium'
        }
    
    return None

# ============================================================================
# STFT-SPECIFIC PATTERN DETECTION
# ============================================================================

def detect_patterns_from_stft(stft_matrix: np.ndarray, time_index: pd.DatetimeIndex) -> List[Dict[str, Any]]:
    """
    Detect patterns from STFT spectrogram.
    """
    patterns = []

    if stft_matrix is None or stft_matrix.size == 0 or stft_matrix.ndim < 2:
        return patterns
    
    time_signal = np.sum(np.abs(stft_matrix), axis=1)
    
    if len(time_signal) != len(time_index):
        x_old = np.linspace(0, 1, len(time_signal))
        x_new = np.linspace(0, 1, len(time_index))
        time_signal = np.interp(x_new, x_old, time_signal)
    
    signal_series = pd.Series(time_signal, index=time_index)
    
    # Detect trend
    exp_pattern = detect_exponential_trend(signal_series)
    if exp_pattern:
        patterns.append(exp_pattern)

    trend_pattern = detect_slow_trend(signal_series)
    if trend_pattern['pattern_type'] != 'no_trend':
        patterns.append(trend_pattern)
    
    # Detect seasonality
    seasonal = detect_seasonality(signal_series)
    patterns.extend(seasonal[:2])
    
    return patterns

# ============================================================================
# UNIFIED PATTERN DETECTOR
# ============================================================================

def detect_all_patterns(features: Any, feature_type: str, time_index: pd.DatetimeIndex, 
                        original_signal: pd.Series = None) -> List[Dict[str, Any]]:
    """
    Unified function to detect patterns from any transform type.
    """
    patterns = []
    
    # Route to appropriate detector
    if 'HHT' in feature_type or 'H' in feature_type:
        if isinstance(features, np.ndarray) and features.ndim == 2:
            patterns.extend(detect_patterns_from_hht(features, time_index))
    if 'Wavelet' in feature_type or 'W' in feature_type:
        if isinstance(features, np.ndarray):
            patterns.extend(detect_patterns_from_wavelet(features, time_index))
    if 'STFT' in feature_type or 'S' in feature_type:
        if isinstance(features, np.ndarray):
            patterns.extend(detect_patterns_from_stft(features, time_index))
    
    # Fallback: analyze original signal
    # Try exponential
    exp = detect_exponential_trend(original_signal)
    patterns.append(exp)
    # Try polynomial
    poly = detect_polynomial_trend(original_signal)
    if poly:
        patterns.append(poly)
    # Linear trend
    trend = detect_slow_trend(original_signal)
    if trend['pattern_type'] != 'no_trend':
        patterns.append(trend)

    # Seasonality
    patterns.extend(detect_seasonality(original_signal)[:2])
    
    # Volatility
    volatility = detect_volatility(original_signal)
    if volatility['confidence'] != 'low':
        patterns.append(volatility)
    
    return patterns

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='M')
    
    # Create test signal with exponential growth + seasonality
    x = np.arange(len(dates))
    trend = 100 * np.exp(0.02 * x)  # Exponential growth
    seasonal = 10 * np.sin(2 * np.pi * x / 12)
    noise = np.random.normal(0, 2, len(dates))
    
    signal_data = pd.Series(trend + seasonal + noise, index=dates)
    
    print("=" * 70)
    print("ENHANCED PATTERN DETECTION DEMONSTRATION")
    print("=" * 70)
    
    # Test exponential detection
    print("\n1. EXPONENTIAL TREND DETECTION:")
    exp_result = detect_exponential_trend(signal_data)
    if exp_result:
        print(f"   Type: {exp_result['pattern_type']}")
        print(f"   Growth Rate: {exp_result['growth_rate_per_period']:.2f}% per period")
        print(f"   Total Change: {exp_result['total_percentage_change']:.2f}%")
        print(f"   {exp_result['time_interpretation']}")
        print(f"   Confidence: {exp_result['confidence']} (R²={exp_result['r_squared']:.3f})")
    
    # Test seasonality
    print("\n2. SEASONALITY DETECTION:")
    seasonal_results = detect_seasonality(signal_data)
    for i, season in enumerate(seasonal_results[:2], 1):
        print(f"   Pattern {i}: {season['period_description']}")
        print(f"     Period: {season['period_months']} months")
        print(f"     Strength: {season['strength']:.2f}")
