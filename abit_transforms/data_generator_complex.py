import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

def generate_complex_cookie_sales(
    n_months: int = 120,
    product_type: str = 'majority',
    complexity: str = 'complex',
    random_seed: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Generate synthetic sales data with complex patterns that fairly test all transforms.
    
    Complexity types:
    - 'simple': Original periodic patterns (STFT favored)
    - 'complex': Non-stationary, evolving patterns (fair comparison)
    - 'hht_favored': Mode mixing, natural oscillations
    - 'wavelet_favored': Multi-scale transients and bursts
    
    Args:
        n_months: Number of months of data
        product_type: 'majority' or 'minority'
        complexity: Type of complexity to add
        random_seed: Random seed for reproducibility
        
    Returns:
        sales: Array of sales values
        components: Dictionary with ground truth
    """
    np.random.seed(random_seed)
    t = np.arange(n_months)
    
    # Base parameters
    if product_type == 'majority':
        base_sales = 10000
        amplitude_scale = 1.0
        noise_level = 500
    else:
        base_sales = 500
        amplitude_scale = 0.1  # Minority products have weaker patterns
        noise_level = 25
    
    # Initialize sales
    sales = np.full(n_months, float(base_sales))
    components = {
        'base_sales': base_sales,
        'complexity_type': complexity,
        'components': {}
    }
    
    if complexity == 'simple':
        # Original simple patterns (STFT performs well)
        # Fixed frequency components
        annual = 3000 * amplitude_scale * np.sin(2 * np.pi * t / 12 - np.pi/2)
        quarterly = 1000 * amplitude_scale * np.sin(2 * np.pi * t / 3)
        
        sales += annual + quarterly
        components['components']['annual'] = annual
        components['components']['quarterly'] = quarterly
        
    elif complexity == 'complex':
        # FAIR COMPARISON - All transforms have strengths
        
        # 1. TIME-VARYING FREQUENCY (HHT good at this)
        # Frequency that speeds up over time (chirp)
        f_start = 1/12  # Annual
        f_end = 1/6     # Semi-annual
        instantaneous_freq = np.linspace(f_start, f_end, n_months)
        phase = 2 * np.pi * np.cumsum(instantaneous_freq)
        evolving_pattern = 2000 * amplitude_scale * np.sin(phase)
        sales += evolving_pattern
        components['components']['chirp'] = evolving_pattern
        
        # 2. INTERMITTENT BURSTS (Wavelets good at this)
        # Seasonal sales spikes that come and go
        burst_pattern = np.zeros(n_months)
        burst_locations = [20, 45, 70, 95]  # Random months with sales events
        for loc in burst_locations:
            if loc < n_months:
                # Create localized burst using Gaussian envelope
                burst_envelope = 1000 * amplitude_scale * np.exp(-((t - loc) / 3)**2)
                burst_oscillation = np.sin(2 * np.pi * t / 2)  # Fast oscillation
                burst_pattern += burst_envelope * burst_oscillation
        sales += burst_pattern
        components['components']['bursts'] = burst_pattern
        
        # 3. MODE MIXING (HHT designed for this)
        # Multiple oscillations at similar frequencies
        mode1 = 1500 * amplitude_scale * np.sin(2 * np.pi * t / 11)  # ~11 month
        mode2 = 1200 * amplitude_scale * np.sin(2 * np.pi * t / 13)  # ~13 month
        # These interfere, creating beating pattern
        mode_mixed = mode1 + mode2
        sales += mode_mixed
        components['components']['mode_mixing'] = mode_mixed
        
        # 4. SUDDEN REGIME CHANGES (All transforms challenged)
        regime_change = np.zeros(n_months)
        if n_months > 60:
            # First regime: slow growth
            regime_change[:60] = 20 * t[:60] * amplitude_scale
            # Second regime: rapid oscillation
            regime_change[60:] = 1000 * amplitude_scale * np.sin(2 * np.pi * t[60:] / 3)
        sales += regime_change
        components['components']['regime_change'] = regime_change
        
        # 5. AMPLITUDE MODULATION (Wavelets handle well)
        # Seasonal pattern with changing amplitude
        modulation_envelope = 1 + 0.5 * np.sin(2 * np.pi * t / 24)  # 2-year cycle
        am_pattern = modulation_envelope * 1000 * amplitude_scale * np.sin(2 * np.pi * t / 6)
        sales += am_pattern
        components['components']['amplitude_modulation'] = am_pattern
        
    elif complexity == 'hht_favored':
        # Patterns where HHT should excel
        
        # 1. Natural mode decomposition - sum of non-harmonic IMFs
        imf1 = 2000 * amplitude_scale * np.sin(2 * np.pi * t / 7.3)   # Non-integer period
        imf2 = 1500 * amplitude_scale * np.sin(2 * np.pi * t / 15.7)
        imf3 = 1000 * amplitude_scale * np.sin(2 * np.pi * t / 31.4)
        
        # Add frequency modulation
        fm_signal = 1800 * amplitude_scale * np.sin(2 * np.pi * t / 12 + 0.2 * np.sin(2 * np.pi * t / 50))
        
        sales += imf1 + imf2 + imf3 + fm_signal
        components['components']['natural_modes'] = imf1 + imf2 + imf3
        components['components']['fm_signal'] = fm_signal
        
        # 2. Nonlinear trend with oscillations
        nonlinear_trend = 10 * amplitude_scale * (t**1.5)
        sales += nonlinear_trend
        components['components']['nonlinear_trend'] = nonlinear_trend
        
    elif complexity == 'wavelet_favored':
        # Patterns where Wavelets should excel
        
        # 1. Multi-scale structure
        # Different patterns at different scales
        scale1 = 3000 * amplitude_scale * np.sin(2 * np.pi * t / 12)  # Annual
        scale2 = 1000 * amplitude_scale * np.sin(2 * np.pi * t / 3)   # Quarterly
        scale3 = 500 * amplitude_scale * np.sin(2 * np.pi * t / 0.5)  # Bi-weekly
        
        sales += scale1 + scale2 + scale3
        components['components']['multiscale'] = scale1 + scale2 + scale3
        
        # 2. Transient events (wavelets excellent at localizing)
        for event_time in [15, 40, 65, 90]:
            if event_time < n_months:
                # Morlet-like wavelet
                sigma = 2
                wavelet = 2000 * amplitude_scale * np.exp(-((t - event_time) / sigma)**2) * \
                         np.cos(2 * np.pi * (t - event_time) / 4)
                sales += wavelet
        
        # 3. Edge/discontinuity
        if n_months > 60:
            sales[60:] += 2000 * amplitude_scale  # Step change
    
    # Add realistic features regardless of complexity
    
    # 1. Day-of-week effect (weakened for monthly data)
    weekly_effect = 200 * amplitude_scale * np.sin(2 * np.pi * t / 0.25)
    sales += weekly_effect
    
    # 2. Random walk component (market volatility)
    random_walk = np.cumsum(np.random.normal(0, 50 * amplitude_scale, n_months))
    sales += random_walk
    components['components']['random_walk'] = random_walk
    
    # 3. Multiplicative noise (realistic for sales)
    multiplicative_noise = sales * np.random.normal(1, 0.05, n_months)
    sales = multiplicative_noise
    
    # 4. Add outliers/anomalies
    n_outliers = max(1, n_months // 40)
    outlier_indices = np.random.choice(n_months, n_outliers, replace=False)
    for idx in outlier_indices:
        sales[idx] *= np.random.uniform(0.5, 2.0)  # Random spike or dip
    components['anomaly_indices'] = outlier_indices.tolist()
    
    # Ensure non-negative
    sales = np.maximum(sales, 100)
    
    # Store pattern characteristics
    components['pattern_types'] = {
        'simple': ['fixed_frequency', 'stationary'],
        'complex': ['chirp', 'bursts', 'mode_mixing', 'regime_change', 'amplitude_modulation'],
        'hht_favored': ['natural_modes', 'frequency_modulation', 'nonlinear_trend'],
        'wavelet_favored': ['multiscale', 'transients', 'discontinuities']
    }.get(complexity, [])
    
    # Calculate complexity metrics
    from scipy import stats
    components['complexity_metrics'] = {
        'mean': float(np.mean(sales)),
        'std': float(np.std(sales)),
        'skewness': float(stats.skew(sales)),
        'kurtosis': float(stats.kurtosis(sales)),
        'is_stationary': complexity == 'simple'
    }
    
    return sales, components


def visualize_complex_patterns(complexity_types: List[str] = None):
    """
    Visualize different complexity patterns to understand what each transform should detect.
    """
    if complexity_types is None:
        complexity_types = ['simple', 'complex', 'hht_favored', 'wavelet_favored']
    
    fig, axes = plt.subplots(len(complexity_types), 2, figsize=(15, 3*len(complexity_types)))
    
    for idx, complexity in enumerate(complexity_types):
        # Generate data
        sales, components = generate_complex_cookie_sales(
            n_months=120,
            product_type='majority',
            complexity=complexity
        )
        
        # Time axis
        t = np.arange(len(sales))
        
        # Plot signal
        ax_signal = axes[idx, 0]
        ax_signal.plot(t, sales, 'b-', linewidth=1)
        ax_signal.set_title(f'{complexity.upper()} Complexity')
        ax_signal.set_xlabel('Time (months)')
        ax_signal.set_ylabel('Sales')
        ax_signal.grid(True, alpha=0.3)
        
        # Plot FFT to show frequency content
        ax_fft = axes[idx, 1]
        fft_vals = np.abs(np.fft.rfft(sales))
        fft_freqs = np.fft.rfftfreq(len(sales))
        ax_fft.plot(fft_freqs[:50], fft_vals[:50])  # Show first 50 frequencies
        ax_fft.set_title(f'Frequency Content')
        ax_fft.set_xlabel('Frequency (cycles/month)')
        ax_fft.set_ylabel('Amplitude')
        ax_fft.grid(True, alpha=0.3)
        
        # Annotate with pattern types
        pattern_text = f"Patterns: {', '.join(components['pattern_types'])}"
        ax_signal.text(0.02, 0.98, pattern_text, transform=ax_signal.transAxes,
                      fontsize=8, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Complexity Comparison for Fair Transform Testing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def generate_fair_test_dataset():
    """
    Generate a complete dataset for fair transform comparison.
    """
    # Create date range
    dates = pd.date_range(start='2015-01-01', periods=120, freq='M')
    
    # Generate different complexity patterns for different products
    simple_sales, _ = generate_complex_cookie_sales(120, 'majority', 'simple', 42)
    complex_sales, _ = generate_complex_cookie_sales(120, 'majority', 'complex', 43)
    hht_sales, _ = generate_complex_cookie_sales(120, 'minority', 'hht_favored', 44)
    wavelet_sales, _ = generate_complex_cookie_sales(120, 'minority', 'wavelet_favored', 45)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'simple_pattern': simple_sales,      # STFT should win
        'complex_pattern': complex_sales,    # Fair competition
        'hht_pattern': hht_sales,           # HHT should win
        'wavelet_pattern': wavelet_sales     # Wavelets should win
    })
    
    return df


# Recommendation for your analysis
def get_recommended_test_signal(test_type: str = 'fair'):
    """
    Get recommended test signal for transform comparison.
    
    Args:
        test_type: 'fair', 'simple', 'hht_test', 'wavelet_test'
    """
    if test_type == 'fair':
        # Use complex pattern for fair comparison
        signal, components = generate_complex_cookie_sales(
            n_months=120,
            product_type='majority',
            complexity='complex'
        )
        print("Using COMPLEX pattern for fair comparison:")
        print("- Time-varying frequency (chirp)")
        print("- Intermittent bursts")
        print("- Mode mixing")
        print("- Regime changes")
        print("- Amplitude modulation")
        print("\nThis should give balanced results across all transforms!")
        
    elif test_type == 'simple':
        signal, components = generate_complex_cookie_sales(
            n_months=120,
            product_type='majority',
            complexity='simple'
        )
        print("Using SIMPLE pattern (STFT favored)")
        
    elif test_type == 'hht_test':
        signal, components = generate_complex_cookie_sales(
            n_months=120,
            product_type='majority',
            complexity='hht_favored'
        )
        print("Using HHT-FAVORED pattern")
        
    elif test_type == 'wavelet_test':
        signal, components = generate_complex_cookie_sales(
            n_months=120,
            product_type='majority',
            complexity='wavelet_favored'
        )
        print("Using WAVELET-FAVORED pattern")
    
    return signal, components


if __name__ == "__main__":
    print("Complex Data Generator for Fair Transform Comparison")
    print("="*60)
    
    # Visualize all complexity types
    print("\n1. Visualizing different complexity patterns...")
    visualize_complex_patterns()
    
    # Generate fair test dataset
    print("\n2. Generating fair test dataset...")
    df = generate_fair_test_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Save to CSV
    df.to_csv('complex_cookie_sales.csv', index=False)
    print("Saved to 'complex_cookie_sales.csv'")
    
    # Show recommendation
    print("\n3. RECOMMENDATION:")
    print("-"*40)
    print("Replace your data generation with:")
    print("\nfrom data_generator_complex import generate_complex_cookie_sales")
    print("sales, components = generate_complex_cookie_sales(")
    print("    n_months=120,")
    print("    product_type='majority',")
    print("    complexity='complex'  # <-- Use 'complex' for fair comparison")
    print(")")
