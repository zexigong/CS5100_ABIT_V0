

"""
Quick start example for ABIT Transform Framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import TransformConfig
from data_generator import generate_cookie_sales
from transforms import HHTTransform, WaveletTransform, STFTTransform
from cascades import DualTransformCascade
from evaluation.metrics import PatternEvaluator

def main():
    """Quick start demonstration"""
    
    print("ABIT Transform Framework - Quick Start")
    print("="*50)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    signal, components = generate_cookie_sales(n_months=120, product_type='majority')
    print(f"   Generated {len(signal)} months of sales data")
    print(f"   Embedded frequencies: {components['frequencies']}")
    
    # 2. Configure transforms
    print("\n2. Configuring transforms...")
    config = TransformConfig(
        minority_amplification=10.0,
        pattern_confidence_threshold=0.5
    )
    print(f"   Minority amplification: {config.minority_amplification}x")
    
    # 3. Single transform example
    print("\n3. Running single transform (HHT)...")
    hht = HHTTransform(config)
    patterns, artifacts = hht.analyze(signal)
    print(f"   Found {len(patterns)} patterns")
    for p in patterns[:3]:  # Show first 3
        if p.frequency > 0:
            print(f"   - Period: {p.period:.1f} months, Confidence: {p.confidence:.2f}")
    
    # 4. Dual cascade example
    print("\n4. Running dual cascade (HHT→Wavelet)...")
    cascade = DualTransformCascade('hht', 'wavelet', config)
    cascade_patterns = cascade.analyze(signal, signal_type='majority')
    print(f"   Found {len(cascade_patterns)} validated patterns")
    
    # 5. Evaluation
    print("\n5. Evaluating performance...")
    evaluator = PatternEvaluator(config)
    metrics = evaluator.evaluate_patterns(cascade_patterns)
    print(f"   Detection score: {metrics['detection_score']:.3f}")
    print(f"   Noise rejection: {metrics['noise_rejection']:.3f}")
    print(f"   Combined score: {metrics['combined_score']:.3f}")
    
    print("\n✓ Quick start complete!")
    print("Run 'python run_comparison.py' for full analysis")

if __name__ == "__main__":
    main()