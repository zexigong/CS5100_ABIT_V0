

"""
Example of integrating transforms with NLP-guided selection
Demonstrates how to connect with your PRISM/MVC architecture
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config import TransformConfig
from cascades import SingleTransformCascade, DualTransformCascade, TripleTransformCascade
from base import Pattern, TransformCascade
from data_generator import load_or_generate_data


@dataclass
class NLPQuery:
    """Represents parsed NLP query"""
    raw_prompt: str
    intent: str  # 'discover', 'quantify', 'validate', 'anomaly'
    focus: str   # 'minority', 'seasonal', 'trend', 'all'
    urgency: str # 'real-time', 'thorough', 'balanced'
    mentioned_periods: List[float] = None

class NLPGuidedAnalyzer:
    """
    Selects and runs appropriate transform cascade based on NLP input.
    This would integrate with your MCP server.
    """
    
    def __init__(self, config: TransformConfig = None):
        self.config = config or TransformConfig()
        
    def parse_prompt(self, prompt: str) -> NLPQuery:
        """
        Simple prompt parsing (in practice, use actual NLP)
        """
        prompt_lower = prompt.lower()
        
        # Determine intent
        if any(word in prompt_lower for word in ['find', 'discover', 'hidden']):
            intent = 'discover'
        elif any(word in prompt_lower for word in ['measure', 'quantify', 'how much']):
            intent = 'quantify'
        elif any(word in prompt_lower for word in ['anomaly', 'unusual', 'spike']):
            intent = 'anomaly'
        else:
            intent = 'discover'
        
        # Determine focus
        if any(word in prompt_lower for word in ['minority', 'underperforming', 'low-volume']):
            focus = 'minority'
        elif any(word in prompt_lower for word in ['seasonal', 'annual', 'quarterly']):
            focus = 'seasonal'
        elif any(word in prompt_lower for word in ['trend', 'growth', 'decline']):
            focus = 'trend'
        else:
            focus = 'all'
        
        # Determine urgency
        if any(word in prompt_lower for word in ['quick', 'fast', 'real-time']):
            urgency = 'real-time'
        elif any(word in prompt_lower for word in ['thorough', 'comprehensive', 'detailed']):
            urgency = 'thorough'
        else:
            urgency = 'balanced'
        
        # Extract mentioned periods
        periods = []
        if 'annual' in prompt_lower or 'yearly' in prompt_lower:
            periods.append(12)
        if 'quarterly' in prompt_lower:
            periods.append(3)
        if 'monthly' in prompt_lower:
            periods.append(1)
        
        return NLPQuery(
            raw_prompt=prompt,
            intent=intent,
            focus=focus,
            urgency=urgency,
            mentioned_periods=periods if periods else None
        )
    
    def select_cascade(self, query: NLPQuery, signal_properties: Dict) -> TransformCascade:
        """
        Select optimal cascade based on parsed query and signal properties
        """
        print(f"\nSelecting cascade for: {query.intent} + {query.focus} + {query.urgency}")
        
        # Decision matrix
        if query.focus == 'minority' and query.intent == 'discover':
            if query.urgency == 'thorough':
                print("  → Selected: HHT→Wavelet→STFT (thorough minority discovery)")
                return TripleTransformCascade('hht', 'wavelet', 'stft', self.config)
            else:
                print("  → Selected: HHT→Wavelet (balanced minority discovery)")
                return DualTransformCascade('hht', 'wavelet', self.config)
        
        elif query.focus == 'seasonal' and query.intent == 'quantify':
            print("  → Selected: Wavelet→STFT (seasonal quantification)")
            return DualTransformCascade('wavelet', 'stft', self.config)
        
        elif query.intent == 'anomaly':
            if query.urgency == 'real-time':
                print("  → Selected: HHT (fast anomaly detection)")
                return SingleTransformCascade('hht', self.config)
            else:
                print("  → Selected: HHT→Wavelet (validated anomaly detection)")
                return DualTransformCascade('hht', 'wavelet', self.config)
        
        elif query.focus == 'trend':
            print("  → Selected: HHT (trend extraction)")
            return SingleTransformCascade('hht', self.config)
        
        elif query.urgency == 'real-time':
            print("  → Selected: Wavelet (fast general analysis)")
            return SingleTransformCascade('wavelet', self.config)
        
        elif query.urgency == 'thorough':
            print("  → Selected: HHT→Wavelet→STFT (comprehensive analysis)")
            return TripleTransformCascade('hht', 'wavelet', 'stft', self.config)
        
        else:
            # Default balanced approach
            print("  → Selected: Wavelet→STFT (balanced general analysis)")
            return DualTransformCascade('wavelet', 'stft', self.config)
    
    def analyze_with_prompt(
        self, 
        signal: np.ndarray, 
        prompt: str,
        signal_type: str = 'unknown'
    ) -> Tuple[List[Pattern], str]:
        """
        Complete analysis pipeline from prompt to patterns
        """
        # Parse the prompt
        query = self.parse_prompt(prompt)
        
        # Determine signal properties (in practice, calculate these)
        signal_properties = {
            'snr': np.std(signal) / (np.mean(signal) + 1e-10),
            'length': len(signal),
            'type': signal_type
        }
        
        # Select appropriate cascade
        cascade = self.select_cascade(query, signal_properties)
        
        # Run analysis
        patterns = cascade.analyze(signal, signal_type)
        
        # Generate explanation
        explanation = self.generate_explanation(patterns, query, cascade.get_name())
        
        return patterns, explanation
    
    def generate_explanation(
        self, 
        patterns: List[Pattern], 
        query: NLPQuery,
        method_used: str
    ) -> str:
        """
        Generate business-friendly explanation of findings
        """
        explanation = f"Analysis Results ({method_used}):\n"
        explanation += "="*50 + "\n"
        
        if not patterns:
            explanation += "No significant patterns detected.\n"
            return explanation
        
        # Group patterns by type
        seasonal = [p for p in patterns if 2 <= p.period <= 13]
        trends = [p for p in patterns if p.frequency == 0]
        other = [p for p in patterns if p not in seasonal and p not in trends]
        
        if query.focus == 'seasonal' or seasonal:
            explanation += f"\nSeasonal Patterns ({len(seasonal)} found):\n"
            for p in seasonal:
                period_name = self._get_period_name(p.period)
                explanation += f"  • {period_name}: Confidence {p.confidence:.2%}\n"
        
        if query.focus == 'trend' or trends:
            explanation += f"\nTrend Analysis:\n"
            for p in trends:
                if 'slope' in p.metadata:
                    direction = "increasing" if p.metadata['slope'] > 0 else "decreasing"
                    explanation += f"  • Overall {direction} trend detected\n"
        
        if query.intent == 'anomaly' and other:
            explanation += f"\nUnusual Patterns:\n"
            for p in other:
                explanation += f"  • {p.period:.1f}-month cycle (uncommon)\n"
        
        explanation += f"\nMethod: {method_used} was selected based on "
        explanation += f"your focus on {query.focus} with {query.urgency} priority.\n"
        
        return explanation
    
    def _get_period_name(self, period: float) -> str:
        """Convert period to business name"""
        period_names = {
            12: "Annual cycle",
            6: "Semi-annual cycle",
            3: "Quarterly cycle",
            1: "Monthly pattern"
        }
        
        for p, name in period_names.items():
            if abs(period - p) < 0.5:
                return name
        
        return f"{period:.1f}-month cycle"

def demonstrate_nlp_integration():
    """Demonstrate NLP-guided cascade selection"""
    
    print("NLP-Guided Transform Selection Demo")
    print("="*70)
    
    # Load data
    df = load_or_generate_data()
    
    # Create analyzer
    analyzer = NLPGuidedAnalyzer()
    
    # Test different prompts
    test_prompts = [
        "Find hidden patterns in our underperforming products",
        "Quick analysis of seasonal trends",
        "Thoroughly analyze all patterns in the sales data",
        "Detect any anomalies or unusual spikes",
        "Quantify the quarterly and annual patterns",
        "What's the overall growth trend?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        print("-"*50)
        
        # Analyze with the prompt
        patterns, explanation = analyzer.analyze_with_prompt(
            df['chocochip'].values,
            prompt,
            signal_type='majority'
        )
        
        print(explanation)
    
    print("\n✓ NLP integration demo complete!")

# MCP Server Integration Example
def mcp_tool_example(signal: np.ndarray, prompt: str) -> Dict:
    """
    Example of how this would work as an MCP tool
    
    @mcp_server.tool()
    async def analyze_sales_pattern(signal: np.ndarray, prompt: str):
    """
    analyzer = NLPGuidedAnalyzer()
    patterns, explanation = analyzer.analyze_with_prompt(signal, prompt)
    
    # Format for MCP response
    return {
        'patterns': [
            {
                'period': p.period,
                'frequency': p.frequency,
                'confidence': p.confidence,
                'business_label': analyzer._get_period_name(p.period)
            }
            for p in patterns
        ],
        'explanation': explanation,
        'method_used': patterns[0].source_method if patterns else "None",
        'pattern_count': len(patterns)
    }

if __name__ == "__main__":
    demonstrate_nlp_integration()