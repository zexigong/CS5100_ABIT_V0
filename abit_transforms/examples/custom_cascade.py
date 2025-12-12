
"""
Example of creating a custom cascade for specific business needs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List
from base import Pattern, TransformCascade
from transforms import HHTTransform, WaveletTransform, STFTTransform
from config import TransformConfig

class BusinessFocusedCascade(TransformCascade):
    """
    Custom cascade that prioritizes business-relevant patterns
    (annual, semi-annual, quarterly)
    """
    
    def __init__(self, config: TransformConfig = None):
        # Use HHT for discovery and Wavelet for validation
        transforms = [
            HHTTransform(config),
            WaveletTransform(config)
        ]
        super().__init__(transforms, config)
        
        # Business-relevant periods (in months)
        self.business_periods = [12, 6, 3]  # Annual, semi-annual, quarterly
        
    def analyze(self, signal: np.ndarray, signal_type: str = 'unknown') -> List[Pattern]:
        """
        Custom analysis focusing on business cycles
        """
        print("Running Business-Focused Cascade Analysis...")
        
        # Step 1: HHT discovers all patterns
        hht_patterns, _ = self.transforms[0].analyze(signal, signal_type=signal_type)
        print(f"  HHT found {len(hht_patterns)} patterns")
        
        # Step 2: Filter for business-relevant periods
        business_patterns = []
        for pattern in hht_patterns:
            if pattern.frequency > 0:
                period = pattern.period
                # Check if close to business period
                for bus_period in self.business_periods:
                    if abs(period - bus_period) < 1.0:  # Within 1 month
                        pattern.metadata['business_period'] = bus_period
                        pattern.metadata['business_label'] = self._get_business_label(bus_period)
                        business_patterns.append(pattern)
                        break
        
        print(f"  Filtered to {len(business_patterns)} business-relevant patterns")
        
        # Step 3: Wavelet validation of business patterns
        if business_patterns:
            wavelet_patterns, wavelet_artifacts = self.transforms[1].analyze(signal)
            power_spectrum = wavelet_artifacts['power_spectrum']
            frequencies = wavelet_artifacts['frequencies']
            
            validated = []
            for pattern in business_patterns:
                # Find wavelet power at this frequency
                freq_idx = np.argmin(np.abs(frequencies - pattern.frequency))
                wavelet_power = power_spectrum[freq_idx]
                
                # Strong validation for business patterns
                validation_score = wavelet_power / (np.mean(power_spectrum) + 1e-10)
                
                if validation_score > 0.3:  # Lower threshold for business patterns
                    pattern.metadata['wavelet_validated'] = True
                    pattern.metadata['validation_score'] = validation_score
                    pattern.confidence = (pattern.confidence + validation_score) / 2
                    pattern.source_method = "BusinessFocusedCascade"
                    validated.append(pattern)
                    
                    print(f"  ✓ Validated: {pattern.metadata['business_label']} "
                          f"(confidence: {pattern.confidence:.2f})")
            
            return validated
        
        return business_patterns
    
    def _get_business_label(self, period: float) -> str:
        """Map period to business label"""
        labels = {
            12: "Annual Cycle",
            6: "Semi-Annual Cycle",
            3: "Quarterly Cycle"
        }
        
        # Find closest match
        closest = min(labels.keys(), key=lambda x: abs(x - period))
        if abs(closest - period) < 1.0:
            return labels[closest]
        return f"~{period:.0f} Month Cycle"

def demonstrate_custom_cascade():
    """Demonstrate the custom cascade"""
    
    print("Custom Business-Focused Cascade Demo")
    print("="*50)
    
    # Generate test data with known business cycles
    from data_generator import generate_cookie_sales
    
    signal, components = generate_cookie_sales(n_months=120, product_type='majority')
    
    # Create custom cascade
    config = TransformConfig()
    cascade = BusinessFocusedCascade(config)
    
    # Run analysis
    patterns = cascade.analyze(signal, signal_type='majority')
    
    # Display results
    print(f"\nFinal Results: {len(patterns)} business patterns identified")
    print("-"*50)
    for pattern in patterns:
        print(f"Pattern: {pattern.metadata['business_label']}")
        print(f"  Period: {pattern.period:.1f} months")
        print(f"  Confidence: {pattern.confidence:.3f}")
        print(f"  Validated: {pattern.metadata.get('wavelet_validated', False)}")
        print()
    
    return patterns

if __name__ == "__main__":
    patterns = demonstrate_custom_cascade()
    print(f"\n✓ Custom cascade complete! Found {len(patterns)} business patterns.")
