

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid') if 'seaborn-v0_8-darkgrid' in plt.style.available else plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


def plot_comparison_results(
    metrics_dict: Dict[str, Dict[str, float]],
    product_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    Create comprehensive visualization of comparison results.
    
    Args:
        metrics_dict: Dictionary of metrics for each method
        product_name: Name of product being analyzed
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Transform Comparison: {product_name}', fontsize=16, fontweight='bold')
    
    # Prepare data
    methods = list(metrics_dict.keys())
    if not methods:
        print("No methods to plot")
        return
    
    metrics = list(metrics_dict[methods[0]].keys())
    
    # Categorize methods by complexity
    single_methods = [m for m in methods if '→' not in m]
    dual_methods = [m for m in methods if m.count('→') == 1]
    triple_methods = [m for m in methods if m.count('→') == 2]
    
    # 1. Combined scores comparison
    ax1 = plt.subplot(2, 3, 1)
    combined_scores = [metrics_dict[m].get('combined_score', 0) for m in methods]
    colors = ['blue' if m in single_methods else 'green' if m in dual_methods else 'red' for m in methods]
    
    bars = ax1.barh(range(len(methods)), combined_scores, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel('Combined Score')
    ax1.set_title('Overall Performance')
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Highlight top 3
    if combined_scores:
        top_indices = np.argsort(combined_scores)[-3:]
        for idx in top_indices:
            bars[idx].set_edgecolor('gold')
            bars[idx].set_linewidth(2)
    
    # 2. Detection vs Noise Rejection scatter
    ax2 = plt.subplot(2, 3, 2)
    detection_scores = [metrics_dict[m].get('detection_score', 0) for m in methods]
    noise_rejection = [metrics_dict[m].get('noise_rejection', 0) for m in methods]
    
    ax2.scatter(detection_scores, noise_rejection, c=colors, s=100, alpha=0.6)
    
    # Add labels for top performers
    if combined_scores:
        for i, method in enumerate(methods):
            if combined_scores[i] > np.percentile(combined_scores, 75):
                ax2.annotate(method, (detection_scores[i], noise_rejection[i]),
                            fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('Detection Score')
    ax2.set_ylabel('Noise Rejection')
    ax2.set_title('Detection vs Noise Trade-off')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Pattern count distribution
    ax3 = plt.subplot(2, 3, 3)
    pattern_counts = [metrics_dict[m].get('num_patterns', 0) for m in methods]
    
    # Group by cascade type
    single_counts = [metrics_dict[m].get('num_patterns', 0) for m in single_methods] if single_methods else []
    dual_counts = [metrics_dict[m].get('num_patterns', 0) for m in dual_methods] if dual_methods else []
    triple_counts = [metrics_dict[m].get('num_patterns', 0) for m in triple_methods] if triple_methods else []
    
    box_data = [x for x in [single_counts, dual_counts, triple_counts] if x]
    box_labels = [label for label, data in zip(['Single', 'Dual', 'Triple'], 
                                                [single_counts, dual_counts, triple_counts]) if data]
    
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors_box = ['blue', 'green', 'red'][:len(box_data)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    
    ax3.set_ylabel('Number of Patterns')
    ax3.set_title('Pattern Count by Complexity')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Heatmap of all metrics
    ax4 = plt.subplot(2, 3, 4)
    
    # Create matrix for heatmap
    metric_matrix = []
    metric_names = ['Detection', 'Noise Rej', 'Confidence', 'Coverage', 'Combined']
    
    for method in methods[:10]:  # Limit to 10 methods for readability
        row = [
            metrics_dict[method].get('detection_score', 0),
            metrics_dict[method].get('noise_rejection', 0),
            metrics_dict[method].get('confidence_mean', 0),
            metrics_dict[method].get('frequency_coverage', 0),
            metrics_dict[method].get('combined_score', 0)
        ]
        metric_matrix.append(row)
    
    if metric_matrix:
        im = ax4.imshow(metric_matrix, aspect='auto', cmap='YlGn', vmin=0, vmax=1)
        ax4.set_xticks(range(len(metric_names)))
        ax4.set_xticklabels(metric_names, rotation=45)
        ax4.set_yticks(range(len(methods[:10])))
        ax4.set_yticklabels(methods[:10], fontsize=8)
        ax4.set_title('Metric Heatmap (Top 10)')
        plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 5. Category average comparison
    ax5 = plt.subplot(2, 3, 5)
    
    category_avgs = {}
    if single_methods:
        category_avgs['Single'] = np.mean([metrics_dict[m].get('combined_score', 0) for m in single_methods])
    if dual_methods:
        category_avgs['Dual'] = np.mean([metrics_dict[m].get('combined_score', 0) for m in dual_methods])
    if triple_methods:
        category_avgs['Triple'] = np.mean([metrics_dict[m].get('combined_score', 0) for m in triple_methods])
    
    if category_avgs:
        colors_cat = ['blue', 'green', 'red'][:len(category_avgs)]
        bars = ax5.bar(category_avgs.keys(), category_avgs.values(), color=colors_cat, alpha=0.7)
        ax5.set_ylabel('Average Combined Score')
        ax5.set_title('Performance by Complexity')
        ax5.set_ylim([0, 1])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', fontsize=10)
    
    # 6. Best method details
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Find best methods
    if methods:
        best_overall = max(methods, key=lambda m: metrics_dict[m].get('combined_score', 0))
        best_single = max(single_methods, key=lambda m: metrics_dict[m].get('combined_score', 0)) if single_methods else None
        best_dual = max(dual_methods, key=lambda m: metrics_dict[m].get('combined_score', 0)) if dual_methods else None
        best_triple = max(triple_methods, key=lambda m: metrics_dict[m].get('combined_score', 0)) if triple_methods else None
        
        summary_text = f"BEST METHODS\n{'='*30}\n\n"
        summary_text += f"Overall Winner:\n  {best_overall}\n"
        summary_text += f"  Score: {metrics_dict[best_overall].get('combined_score', 0):.3f}\n\n"
        
        if best_single:
            summary_text += f"Best Single:\n  {best_single} ({metrics_dict[best_single].get('combined_score', 0):.3f})\n"
        if best_dual:
            summary_text += f"Best Dual:\n  {best_dual} ({metrics_dict[best_dual].get('combined_score', 0):.3f})\n"
        if best_triple:
            summary_text += f"Best Triple:\n  {best_triple} ({metrics_dict[best_triple].get('combined_score', 0):.3f})\n"
        
        summary_text += f"\n{'='*30}\n"
        summary_text += "KEY INSIGHTS:\n"
        
        # Check if order matters
        hht_wav = next((m for m in dual_methods if m == 'HHT→Wavelet'), None)
        wav_hht = next((m for m in dual_methods if m == 'Wavelet→HHT'), None)
        
        if hht_wav and wav_hht:
            diff = abs(metrics_dict[hht_wav].get('combined_score', 0) - 
                      metrics_dict[wav_hht].get('combined_score', 0))
            if diff > 0.1:
                summary_text += f"• Order matters! (Δ={diff:.3f})\n"
            else:
                summary_text += f"• Order impact minimal (Δ={diff:.3f})\n"
        
        # Performance by complexity
        if category_avgs:
            if len(category_avgs) >= 2:
                if 'Dual' in category_avgs and 'Triple' in category_avgs:
                    if category_avgs['Dual'] > category_avgs['Triple']:
                        summary_text += "• Dual cascades optimal\n"
                    elif category_avgs['Triple'] > category_avgs['Dual'] * 1.1:
                        summary_text += "• Triple cascades worth it\n"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_pattern_summary(
    patterns: List,
    title: str = "Pattern Summary",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create a summary visualization of discovered patterns.
    
    Args:
        patterns: List of Pattern objects
        title: Title for the plot
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if not patterns:
        print("No patterns to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Extract pattern data
    frequencies = [p.frequency for p in patterns if hasattr(p, 'frequency') and p.frequency > 0]
    periods = [p.period for p in patterns if hasattr(p, 'period') and p.period < np.inf]
    amplitudes = [p.amplitude for p in patterns if hasattr(p, 'amplitude') and p.amplitude > 0]
    confidences = [p.confidence for p in patterns if hasattr(p, 'confidence')]
    sources = [p.source_method for p in patterns if hasattr(p, 'source_method')]
    
    # 1. Period distribution
    ax1 = axes[0, 0]
    if periods:
        ax1.hist(periods, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Period (months)')
        ax1.set_ylabel('Count')
        ax1.set_title('Period Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Mark business periods
        business_periods = [12, 6, 3, 1]
        for bp in business_periods:
            ax1.axvline(x=bp, color='red', linestyle='--', alpha=0.5)
    
    # 2. Confidence vs Amplitude scatter
    ax2 = axes[0, 1]
    if amplitudes and confidences:
        # Ensure equal length
        min_len = min(len(amplitudes), len(confidences))
        ax2.scatter(amplitudes[:min_len], confidences[:min_len], 
                   alpha=0.6, s=50, c=confidences[:min_len], cmap='viridis')
        ax2.set_xlabel('Amplitude')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Amplitude vs Confidence')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Confidence', rotation=270, labelpad=15)
    
    # 3. Source method distribution
    ax3 = axes[1, 0]
    if sources:
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Sort by count
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        source_names = [s[0] for s in sorted_sources[:10]]  # Top 10
        source_values = [s[1] for s in sorted_sources[:10]]
        
        ax3.barh(range(len(source_names)), source_values, color='coral')
        ax3.set_yticks(range(len(source_names)))
        ax3.set_yticklabels(source_names, fontsize=9)
        ax3.set_xlabel('Count')
        ax3.set_title('Pattern Discovery Methods')
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Pattern quality summary
    ax4 = axes[1, 1]
    
    # Create quality metrics
    if patterns:
        n_patterns = len(patterns)
        n_high_conf = len([p for p in patterns if hasattr(p, 'confidence') and p.confidence > 0.7])
        n_business = len([p for p in patterns if hasattr(p, 'is_business_relevant') and 
                         callable(p.is_business_relevant) and p.is_business_relevant()])
        
        # Text summary
        summary_text = "PATTERN QUALITY SUMMARY\n"
        summary_text += "="*25 + "\n\n"
        summary_text += f"Total Patterns: {n_patterns}\n"
        summary_text += f"High Confidence (>0.7): {n_high_conf}\n"
        summary_text += f"Business Relevant: {n_business}\n"
        
        if confidences:
            summary_text += f"\nConfidence Stats:\n"
            summary_text += f"  Mean: {np.mean(confidences):.3f}\n"
            summary_text += f"  Std: {np.std(confidences):.3f}\n"
            summary_text += f"  Min: {np.min(confidences):.3f}\n"
            summary_text += f"  Max: {np.max(confidences):.3f}\n"
        
        if periods:
            summary_text += f"\nPeriod Stats:\n"
            summary_text += f"  Mean: {np.mean(periods):.1f} months\n"
            summary_text += f"  Most common: {max(set(periods), key=periods.count):.1f} months\n"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_transform_comparison(
    signal: np.ndarray,
    transform_results: Dict[str, Tuple[List, Dict]],
    time_axis: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Compare transform outputs visually.
    
    Args:
        signal: Original signal
        transform_results: Dict with keys like 'HHT', 'Wavelet', 'STFT' 
                          and values as (patterns, artifacts) tuples
        time_axis: Optional time axis for x-axis labels
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    n_transforms = len(transform_results)
    if n_transforms == 0:
        print("No transform results to plot")
        return
    
    fig, axes = plt.subplots(n_transforms + 1, 2, figsize=figsize)
    if n_transforms == 1:
        axes = axes.reshape(2, 2)
    
    if time_axis is None:
        time_axis = np.arange(len(signal))
    
    # Plot original signal
    axes[0, 0].plot(time_axis, signal, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Signal statistics
    axes[0, 1].text(0.1, 0.5, 
                   f"Signal Statistics\n{'='*20}\n"
                   f"Length: {len(signal)}\n"
                   f"Mean: {np.mean(signal):.2f}\n"
                   f"Std: {np.std(signal):.2f}\n"
                   f"Min: {np.min(signal):.2f}\n"
                   f"Max: {np.max(signal):.2f}",
                   fontsize=10, family='monospace',
                   transform=axes[0, 1].transAxes)
    axes[0, 1].axis('off')
    
    # Plot each transform result
    for idx, (transform_name, (patterns, artifacts)) in enumerate(transform_results.items(), 1):
        if idx >= len(axes):
            break
            
        ax_left = axes[idx, 0]
        ax_right = axes[idx, 1]
        
        # Left: Pattern visualization
        if patterns:
            periods = [p.period for p in patterns if hasattr(p, 'period') and p.period < np.inf]
            confidences = [p.confidence for p in patterns if hasattr(p, 'confidence')]
            
            if periods and confidences:
                ax_left.scatter(periods, confidences, s=100, alpha=0.6, c=confidences, cmap='coolwarm')
                ax_left.set_xlabel('Period (months)')
                ax_left.set_ylabel('Confidence')
                ax_left.set_title(f'{transform_name}: Patterns')
                ax_left.grid(True, alpha=0.3)
                
                # Mark business periods
                for bp in [12, 6, 3, 1]:
                    ax_left.axvline(x=bp, color='gray', linestyle='--', alpha=0.3)
        
        # Right: Transform-specific visualization
        ax_right.set_title(f'{transform_name}: Transform Output')
        
        # Customize based on transform type
        if 'HHT' in transform_name.upper() and 'imfs' in artifacts:
            # Plot first few IMFs
            imfs = artifacts['imfs']
            for i, imf in enumerate(imfs[:3]):
                ax_right.plot(time_axis[:len(imf)], imf + i*np.std(signal)*2, 
                             alpha=0.7, label=f'IMF{i+1}')
            ax_right.legend(fontsize=8)
            
        elif 'WAVELET' in transform_name.upper() and 'power_spectrum' in artifacts:
            # Plot power spectrum
            power = artifacts['power_spectrum']
            scales = artifacts.get('scales', np.arange(len(power)))
            ax_right.plot(scales, power)
            ax_right.set_xlabel('Scale')
            ax_right.set_ylabel('Power')
            
        elif 'STFT' in transform_name.upper() and 'avg_power_spectrum' in artifacts:
            # Plot average power spectrum
            avg_power = artifacts['avg_power_spectrum']
            freqs = artifacts.get('frequencies', np.arange(len(avg_power)))
            ax_right.plot(freqs, avg_power)
            ax_right.set_xlabel('Frequency')
            ax_right.set_ylabel('Average Power')
        
        ax_right.grid(True, alpha=0.3)
    
    plt.suptitle('Transform Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# Export all visualization functions
__all__ = [
    'plot_comparison_results',
    'plot_pattern_summary',
    'plot_transform_comparison'
]