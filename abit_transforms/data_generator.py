import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import json

def generate_cookie_sales(
    n_months: int = 120,
    product_type: str = 'majority',
    random_seed: int = 42,
    include_anomalies: bool = False,
    custom_patterns: Dict = None
) -> Tuple[np.ndarray, Dict]:
    """
    Generate synthetic cookie sales data with known patterns.
    
    This is the main data generation function for testing transforms.
    It creates realistic sales data with embedded patterns that we know ground truth for.
    
    Args:
        n_months: Number of months of data to generate
        product_type: 'majority' (high volume) or 'minority' (low volume)
        random_seed: Random seed for reproducibility
        include_anomalies: Whether to include anomalous events
        custom_patterns: Override default pattern parameters
        
    Returns:
        sales: Array of sales values
        components: Dictionary containing all embedded pattern information
    """
    np.random.seed(random_seed)
    t = np.arange(n_months)
    
    # Set base parameters based on product type
    if product_type == 'majority':
        # Chocochip cookies - 90% of sales
        params = {
            'base_sales': 10000,
            'trend_slope': 50,
            'seasonal_amplitude': 3000,
            'semi_annual_amplitude': 900,
            'quarterly_amplitude': 600,
            'noise_level': 500,
            'growth_rate': 0.002  # Monthly growth rate
        }
    elif product_type == 'minority':
        # Oatmeal raisin - 5% of sales
        params = {
            'base_sales': 500,
            'trend_slope': 2,
            'seasonal_amplitude': 50,
            'semi_annual_amplitude': 15,
            'quarterly_amplitude': 10,
            'noise_level': 25,
            'growth_rate': 0.001
        }
    else:
        # Custom or balanced product
        params = {
            'base_sales': 2000,
            'trend_slope': 10,
            'seasonal_amplitude': 600,
            'semi_annual_amplitude': 180,
            'quarterly_amplitude': 120,
            'noise_level': 100,
            'growth_rate': 0.0015
        }
    
    # Override with custom patterns if provided
    if custom_patterns:
        params.update(custom_patterns)
    
    # Initialize with base sales
    sales = np.full(n_months, params['base_sales'], dtype=float)
    
    # Component tracking for ground truth
    components = {
        'base_sales': params['base_sales'],
        'components': {},
        'frequencies': {},
        'parameters': params
    }
    
    # 1. Linear trend
    trend = params['trend_slope'] * t
    sales += trend
    components['components']['trend'] = trend
    components['trend_slope'] = params['trend_slope']
    
    # 2. Exponential growth (more realistic for products)
    if params['growth_rate'] > 0:
        growth = params['base_sales'] * (np.exp(params['growth_rate'] * t) - 1)
        sales += growth
        components['components']['growth'] = growth
    
    # 3. Annual seasonal pattern (12-month period)
    annual_pattern = params['seasonal_amplitude'] * np.sin(2 * np.pi * t / 12 - np.pi/2)
    sales += annual_pattern
    components['components']['annual'] = annual_pattern
    components['frequencies']['annual'] = 1/12
    
    # 4. Semi-annual pattern (6-month period)
    semi_annual = params['semi_annual_amplitude'] * np.sin(2 * np.pi * t / 6)
    sales += semi_annual
    components['components']['semi_annual'] = semi_annual
    components['frequencies']['semi_annual'] = 1/6
    
    # 5. Quarterly pattern (3-month period)
    quarterly = params['quarterly_amplitude'] * np.sin(2 * np.pi * t / 3 + np.pi/4)
    sales += quarterly
    components['components']['quarterly'] = quarterly
    components['frequencies']['quarterly'] = 1/3
    
    # 6. Holiday effects (November-December boost)
    holiday_boost = np.zeros(n_months)
    for year in range(n_months // 12 + 1):
        nov_idx = year * 12 + 10  # November
        dec_idx = year * 12 + 11  # December
        
        if nov_idx < n_months:
            holiday_boost[nov_idx] = params['seasonal_amplitude'] * 0.5
        if dec_idx < n_months:
            holiday_boost[dec_idx] = params['seasonal_amplitude'] * 0.8
    
    sales += holiday_boost
    components['components']['holiday'] = holiday_boost
    
    # 7. Random noise
    noise = np.random.normal(0, params['noise_level'], n_months)
    sales += noise
    components['components']['noise'] = noise
    components['noise_level'] = params['noise_level']
    
    # 8. Anomalies (if requested)
    if include_anomalies:
        anomalies = np.zeros(n_months)
        n_anomalies = max(1, n_months // 30)  # One anomaly per 30 months
        
        for _ in range(n_anomalies):
            anomaly_idx = np.random.randint(10, n_months - 10)
            anomaly_magnitude = np.random.uniform(3, 5) * params['noise_level']
            anomaly_sign = np.random.choice([-1, 1])
            
            # Create anomaly with some persistence
            anomalies[anomaly_idx:anomaly_idx+3] = anomaly_sign * anomaly_magnitude
            
        sales += anomalies
        components['components']['anomalies'] = anomalies
        components['anomaly_indices'] = np.where(np.abs(anomalies) > 0)[0].tolist()
    
    # Ensure no negative sales
    sales = np.maximum(sales, 0)
    
    # Calculate SNR
    signal_power = np.var(sales - noise)
    noise_power = np.var(noise)
    components['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Summary statistics
    components['statistics'] = {
        'mean': float(np.mean(sales)),
        'std': float(np.std(sales)),
        'min': float(np.min(sales)),
        'max': float(np.max(sales)),
        'cv': float(np.std(sales) / (np.mean(sales) + 1e-10))
    }
    
    return sales, components


def generate_complex_sales(
    n_months: int = 120,
    scenario: str = 'regime_change',
    random_seed: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Generate complex sales patterns for advanced testing.
    
    Args:
        n_months: Length of time series
        scenario: Type of complex pattern
                 'regime_change': Sudden change in pattern
                 'evolving': Gradually changing frequencies
                 'intermittent': Patterns that come and go
                 'mixed': Multiple products combined
        
    Returns:
        sales: Complex sales signal
        components: Ground truth information
    """
    np.random.seed(random_seed)
    t = np.arange(n_months)
    components = {'scenario': scenario}
    
    if scenario == 'regime_change':
        # First half: one pattern
        sales1, comp1 = generate_cookie_sales(n_months // 2, 'majority', random_seed)
        
        # Second half: different pattern
        sales2, comp2 = generate_cookie_sales(
            n_months - n_months // 2, 
            'majority',
            random_seed + 1,
            custom_patterns={'seasonal_amplitude': 5000, 'trend_slope': 100}
        )
        
        sales = np.concatenate([sales1, sales2])
        components['change_point'] = n_months // 2
        components['before'] = comp1
        components['after'] = comp2
        
    elif scenario == 'evolving':
        # Frequency that changes over time
        base = 5000
        
        # Chirp signal (frequency increases linearly)
        f0 = 1/12  # Start at annual
        f1 = 1/3   # End at quarterly
        frequencies = np.linspace(f0, f1, n_months)
        
        phase = 2 * np.pi * np.cumsum(frequencies)
        evolving_pattern = 2000 * np.sin(phase)
        
        sales = base + evolving_pattern
        sales += np.random.normal(0, 200, n_months)
        
        components['frequency_start'] = f0
        components['frequency_end'] = f1
        components['pattern'] = 'linear_chirp'
        
    elif scenario == 'intermittent':
        # Pattern that appears and disappears
        sales = np.full(n_months, 5000.0)
        
        # Add seasonal pattern only in certain periods
        active_periods = []
        for start in range(0, n_months, 36):
            end = min(start + 12, n_months)
            sales[start:end] += 2000 * np.sin(2 * np.pi * np.arange(end - start) / 12)
            active_periods.append((start, end))
        
        sales += np.random.normal(0, 300, n_months)
        components['active_periods'] = active_periods
        
    elif scenario == 'mixed':
        # Combine multiple products
        sales1, _ = generate_cookie_sales(n_months, 'majority', random_seed)
        sales2, _ = generate_cookie_sales(n_months, 'minority', random_seed + 1)
        
        # Weighted combination
        sales = 0.7 * sales1 + 0.3 * sales2
        components['mixture'] = {'majority_weight': 0.7, 'minority_weight': 0.3}
        
    else:
        # Default to standard
        sales, components = generate_cookie_sales(n_months, 'majority', random_seed)
    
    # Ensure non-negative
    sales = np.maximum(sales, 0)
    
    return sales, components


def load_or_generate_data(
    filepath: str = 'cookie_sales.csv',
    regenerate: bool = False,
    n_months: int = 120
) -> pd.DataFrame:

    import os
    
    if not regenerate and os.path.exists(filepath):
        # Load existing data
        df = pd.read_csv(filepath)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        print(f"Loaded existing data from {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    else:
        # Generate new data
        print(f"Generating new dataset with {n_months} months...")
        
        # Create date range
        dates = pd.date_range(
            start='2015-01-01',
            periods=n_months,
            freq='M'
        )
        
        # Generate different products
        chocochip, choco_comp = generate_cookie_sales(n_months, 'majority', 42)
        oatmeal, oat_comp = generate_cookie_sales(n_months, 'minority', 43)
        sugar, sugar_comp = generate_cookie_sales(n_months, 'majority', 44,
                                                 custom_patterns={'seasonal_amplitude': 2000})
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'chocochip': chocochip,
            'oatmeal_raisin': oatmeal,
            'sugar': sugar,
            'total_sales': chocochip + oatmeal + sugar
        })
        
        # Add derived features
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['is_holiday'] = df['month'].isin([11, 12]).astype(int)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Generated and saved new data to {filepath}")
        
        # Also save components for validation
        components = {
            'chocochip': choco_comp,
            'oatmeal_raisin': oat_comp,
            'sugar': sugar_comp
        }
        
        with open('known_patterns.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_components = convert_to_serializable(components)
            json.dump(serializable_components, f, indent=2)
        
        print("Saved ground truth patterns to known_patterns.json")
    
    return df


def create_test_signals() -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    Create a variety of test signals for transform validation.
    
    Returns:
        Dictionary of test signals with their ground truth
    """
    test_signals = {}
    
    # 1. Pure sinusoid
    t = np.arange(120)
    test_signals['pure_annual'] = (
        1000 + 500 * np.sin(2 * np.pi * t / 12),
        {'frequency': 1/12, 'amplitude': 500, 'period': 12}
    )
    
    # 2. Multiple frequencies
    multi = 1000 + 300 * np.sin(2 * np.pi * t / 12) + 200 * np.sin(2 * np.pi * t / 3)
    test_signals['multi_frequency'] = (
        multi,
        {'frequencies': [1/12, 1/3], 'periods': [12, 3]}
    )
    
    # 3. Trend only
    test_signals['linear_trend'] = (
        1000 + 10 * t,
        {'trend_slope': 10, 'frequency': 0}
    )
    
    # 4. Noisy signal
    test_signals['high_noise'] = (
        1000 + 100 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 200, 120),
        {'frequency': 1/12, 'snr': -6}  # Low SNR
    )
    
    # 5. Impulse
    impulse = np.zeros(120)
    impulse[60] = 1000
    test_signals['impulse'] = (
        impulse,
        {'type': 'impulse', 'location': 60}
    )
    
    # 6. Step function
    step = np.ones(120) * 1000
    step[60:] = 2000
    test_signals['step'] = (
        step,
        {'type': 'step', 'change_point': 60}
    )
    
    return test_signals


def visualize_sales_data(df: pd.DataFrame, save_path: str = None):
    """
    Create comprehensive visualization of sales data.
    
    Args:
        df: DataFrame with sales data
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Cookie Sales Data Analysis', fontsize=16)
    
    # 1. Time series plot
    ax1 = axes[0, 0]
    for col in ['chocochip', 'oatmeal_raisin', 'sugar']:
        if col in df.columns:
            ax1.plot(df['date'], df[col], label=col.replace('_', ' ').title(), linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales Units')
    ax1.set_title('Sales Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log scale view
    ax2 = axes[0, 1]
    for col in ['chocochip', 'oatmeal_raisin', 'sugar']:
        if col in df.columns:
            ax2.semilogy(df['date'], df[col], label=col.replace('_', ' ').title())
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales Units (log scale)')
    ax2.set_title('Log Scale View (reveals minority patterns)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Seasonal patterns
    ax3 = axes[1, 0]
    if 'month' in df.columns:
        monthly_avg = df.groupby('month')['total_sales'].mean()
        ax3.bar(monthly_avg.index, monthly_avg.values, color='skyblue', edgecolor='navy')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Sales')
        ax3.set_title('Seasonal Pattern')
        ax3.set_xticks(range(1, 13))
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Distribution comparison
    ax4 = axes[1, 1]
    data_to_plot = []
    labels = []
    for col in ['chocochip', 'oatmeal_raisin', 'sugar']:
        if col in df.columns:
            data_to_plot.append(df[col].values)
            labels.append(col.replace('_', ' ').title())
    
    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    ax4.set_ylabel('Sales Units')
    ax4.set_title('Sales Distribution by Product')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Correlation heatmap
    ax5 = axes[2, 0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax5, cbar_kws={'label': 'Correlation'})
    ax5.set_title('Feature Correlation Matrix')
    
    # 6. Growth trend
    ax6 = axes[2, 1]
    if 'total_sales' in df.columns:
        # Calculate rolling mean
        rolling_mean = df['total_sales'].rolling(window=12, center=True).mean()
        ax6.plot(df['date'], df['total_sales'], alpha=0.3, label='Actual')
        ax6.plot(df['date'], rolling_mean, linewidth=2, color='red', label='12-Month Moving Avg')
        
        # Add trend line
        from scipy import stats
        x = np.arange(len(df))
        slope, intercept, _, _, _ = stats.linregress(x, df['total_sales'])
        trend_line = intercept + slope * x
        ax6.plot(df['date'], trend_line, '--', color='green', label=f'Trend (slope={slope:.1f})')
        
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Total Sales')
        ax6.set_title('Growth Trend Analysis')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    """Test data generation"""
    
    print("Testing Data Generation Module")
    print("="*50)
    
    # 1. Generate standard dataset
    print("\n1. Generating standard dataset...")
    df = load_or_generate_data(regenerate=True)
    print(f"   Shape: {df.shape}")
    print(f"   Products: {[col for col in df.columns if 'sales' in col or col in ['chocochip', 'oatmeal_raisin', 'sugar']]}")
    
    # 2. Test different product types
    print("\n2. Testing product types...")
    for product_type in ['majority', 'minority']:
        sales, components = generate_cookie_sales(120, product_type)
        print(f"   {product_type}: mean={components['statistics']['mean']:.0f}, "
              f"SNR={components['snr']:.1f}dB")
    
    # 3. Test complex scenarios
    print("\n3. Testing complex scenarios...")
    for scenario in ['regime_change', 'evolving', 'intermittent']:
        sales, components = generate_complex_sales(120, scenario)
        print(f"   {scenario}: {components.get('scenario', 'unknown')}")
    
    # 4. Create test signals
    print("\n4. Creating test signals...")
    test_signals = create_test_signals()
    for name, (signal, truth) in test_signals.items():
        print(f"   {name}: {truth}")
    
    # 5. Visualize
    print("\n5. Creating visualizations...")
    visualize_sales_data(df, 'sales_data_analysis.png')
    
    print("\n✓ Data generation module test complete!")