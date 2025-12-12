
import numpy as np
from typing import List, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Pattern
from config import TransformConfig, DEFAULT_CONFIG

class PatternEvaluator:
    """Evaluate pattern detection performance"""
    
    def __init__(self, config: TransformConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def evaluate_patterns(self, patterns: List[Pattern]) -> Dict[str, float]:
        """
        Comprehensive evaluation of detected patterns.
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'num_patterns': len(patterns),
            'detection_score': self.calculate_detection_score(patterns),
            'noise_rejection': self.calculate_noise_rejection(patterns),
            'confidence_mean': self.calculate_mean_confidence(patterns),
            'confidence_std': self.calculate_confidence_std(patterns),
            'frequency_coverage': self.calculate_frequency_coverage(patterns),
            'combined_score': 0.0
        }
        
        # Combined score with weighted components
        metrics['combined_score'] = (
            self.config.detection_weight * metrics['detection_score'] +
            self.config.noise_rejection_weight * metrics['noise_rejection']
        )
        
        return metrics
    
    def calculate_detection_score(self, patterns: List[Pattern]) -> float:
        """Calculate how well known frequencies are detected"""
        if not patterns:
            return 0.0
            
        score = 0.0
        max_score = len(self.config.known_frequencies)
        
        for known_freq in self.config.known_frequencies:
            # Find best matching pattern
            best_match = None
            best_distance = float('inf')
            
            for pattern in patterns:
                if pattern.frequency > 0:
                    distance = abs(pattern.frequency - known_freq)
                    if distance < best_distance and distance < self.config.frequency_tolerance:
                        best_match = pattern
                        best_distance = distance
            
            if best_match:
                # Score based on confidence and accuracy
                accuracy_score = 1.0 - (best_distance / self.config.frequency_tolerance)
                score += best_match.confidence * accuracy_score
        
        return score / max_score if max_score > 0 else 0.0
    
    def calculate_noise_rejection(self, patterns: List[Pattern]) -> float:
        """Calculate how well false patterns are rejected"""
        if not patterns:
            return 1.0  # No false patterns is perfect
        
        false_patterns = 0
        for pattern in patterns:
            if pattern.frequency > 0:
                is_false = True
                for known_freq in self.config.known_frequencies:
                    if abs(pattern.frequency - known_freq) < self.config.frequency_tolerance:
                        is_false = False
                        break
                if is_false:
                    false_patterns += 1
        
        # Better score = fewer false patterns
        return 1.0 / (1 + false_patterns)
    
    def calculate_mean_confidence(self, patterns: List[Pattern]) -> float:
        """Calculate average confidence of patterns"""
        if not patterns:
            return 0.0
        
        confidences = [p.confidence for p in patterns if p.frequency > 0]
        return np.mean(confidences) if confidences else 0.0
    
    def calculate_confidence_std(self, patterns: List[Pattern]) -> float:
        """Calculate confidence standard deviation"""
        if not patterns:
            return 0.0
            
        confidences = [p.confidence for p in patterns if p.frequency > 0]
        return np.std(confidences) if len(confidences) > 1 else 0.0
    
    def calculate_frequency_coverage(self, patterns: List[Pattern]) -> float:
        """Calculate coverage of frequency spectrum"""
        if not patterns:
            return 0.0
            
        frequencies = [p.frequency for p in patterns if p.frequency > 0]
        if not frequencies:
            return 0.0
            
        # Check coverage of expected frequency range
        freq_range = max(frequencies) - min(frequencies)
        expected_range = max(self.config.known_frequencies) - min(self.config.known_frequencies)
        
        return min(freq_range / expected_range, 1.0) if expected_range > 0 else 0.0
    
    def compare_methods(
        self, 
        results: Dict[str, List[Pattern]]
    ) -> Tuple[Dict[str, Dict[str, float]], str]:
        """
        Compare multiple methods.
        
        Args:
            results: Dictionary mapping method names to pattern lists
            
        Returns:
            all_metrics: Dictionary of metrics for each method
            best_method: Name of best performing method
        """
        all_metrics = {}
        
        for method_name, patterns in results.items():
            all_metrics[method_name] = self.evaluate_patterns(patterns)
        
        # Find best method by combined score
        best_method = max(all_metrics.items(), key=lambda x: x[1]['combined_score'])[0]
        
        return all_metrics, best_method