"""
ML-based polymer-solvent solubility predictor using corrected Hansen parameters.
"""

import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SolubilityPredictor:
    """Predict polymer-solvent solubility using Random Forest ML model."""

    def __init__(self, model_dir='./models'):
        """Load trained model artifacts."""
        model_dir = Path(model_dir)

        # Load model
        self.model = joblib.load(
            model_dir / 'corrected_Random_Forest_20251231_212903_model.pkl'
        )
        self.scaler = joblib.load(
            model_dir / 'corrected_Random_Forest_20251231_212903_scaler.pkl'
        )

        with open(model_dir / 'corrected_Random_Forest_20251231_212903_metadata.json') as f:
            self.metadata = json.load(f)

        self.threshold = self.metadata['classification_threshold']
        logger.info(f"Loaded ML model with threshold={self.threshold:.4f}")

    def predict(self, polymer_hsp: Dict[str, float], solvent_hsp: Dict[str, float],
                r0: float, molar_volume: float = 100.0) -> Dict[str, Any]:
        """
        Predict solubility using Hansen Solubility Parameters.

        Args:
            polymer_hsp: {'Dispersion': float, 'Polar': float, 'Hydrogen': float}
            solvent_hsp: {'Dispersion': float, 'Polar': float, 'Hydrogen': float}
            r0: Interaction radius (float)
            molar_volume: Solvent molar volume (default: 100.0)

        Returns:
            {
                'soluble': bool,
                'probability': float (0-1),
                'confidence': float (0-1),
                'red': float,
                'ra': float,
                'r0': float,
                'threshold': float
            }
        """
        # Extract HSP values
        p_d = polymer_hsp['Dispersion']
        p_p = polymer_hsp['Polar']
        p_h = polymer_hsp['Hydrogen']

        s_d = solvent_hsp['Dispersion']
        s_p = solvent_hsp['Polar']
        s_h = solvent_hsp['Hydrogen']

        # Calculate derived features
        delta_d = abs(p_d - s_d)
        delta_p = abs(p_p - s_p)
        delta_h = abs(p_h - s_h)

        # CORRECT Hansen distance formula
        ra = np.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)
        red = ra / r0 if r0 > 0 else float('inf')

        # Create feature vector (10 features)
        features = np.array([[
            p_d, p_p, p_h,          # Polymer HSP
            s_d, s_p, s_h,          # Solvent HSP
            molar_volume,           # Solvent molar volume
            r0,                     # Interaction radius
            ra,                     # Hansen distance
            red                     # RED value
        ]])

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0, 1]
        soluble = proba >= self.threshold

        # Calculate confidence (distance from threshold)
        confidence = abs(proba - self.threshold) / max(self.threshold, 1 - self.threshold)
        confidence = min(confidence, 1.0)

        return {
            'soluble': bool(soluble),
            'probability': float(proba),
            'confidence': float(confidence),
            'red': float(red),
            'ra': float(ra),
            'r0': float(r0),
            'threshold': float(self.threshold)
        }

# Global instance (lazy loaded)
_predictor = None

def get_predictor():
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = SolubilityPredictor()
    return _predictor
