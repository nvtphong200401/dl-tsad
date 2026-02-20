#!/usr/bin/env python
"""Simple test script for foundation models

This script tests foundation models on synthetic data.
Run this to verify foundation models are working before full pipeline integration.

Usage:
    python experiments/test_foundation_models_simple.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_chronos_simple():
    """Test Chronos on simple synthetic data"""
    print("\n" + "="*60)
    print("Testing Chronos (Amazon)")
    print("="*60)

    try:
        from foundation_models import ChronosWrapper

        # Create simple sine wave
        t = np.linspace(0, 4*np.pi, 200)
        train_data = np.sin(t)

        print(f"Training data: {len(train_data)} points")
        print(f"Training data range: [{train_data.min():.2f}, {train_data.max():.2f}]")

        # Initialize model
        print("\nLoading Chronos model...")
        model = ChronosWrapper(model_name="amazon/chronos-t5-tiny")  # Use tiny for fast testing

        # Generate forecast
        print("Generating forecast...")
        result = model.forecast(
            context=train_data,
            horizon=50,
            num_samples=20  # Fewer samples for faster testing
        )

        print(f"\n✓ Forecast generated successfully!")
        print(f"  Forecast shape: {result['forecast'].shape}")
        print(f"  Forecast range: [{result['forecast'].min():.2f}, {result['forecast'].max():.2f}]")
        print(f"  Quantiles available: {list(result['quantiles'].keys())}")
        print(f"  Uncertainty (P90-P10): mean={result['uncertainty'].mean():.2f}")

        return True

    except ImportError as e:
        print(f"\n✗ Chronos not available: {e}")
        print("  Install with: pip install chronos-forecasting")
        return False
    except Exception as e:
        print(f"\n✗ Chronos test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timesfm_simple():
    """Test TimesFM on simple synthetic data"""
    print("\n" + "="*60)
    print("Testing TimesFM (Google)")
    print("="*60)

    try:
        from foundation_models import TimesFMWrapper

        # Create simple sine wave
        t = np.linspace(0, 4*np.pi, 200)
        train_data = np.sin(t)

        print(f"Training data: {len(train_data)} points")
        print(f"Training data range: [{train_data.min():.2f}, {train_data.max():.2f}]")

        # Initialize model
        print("\nLoading TimesFM model...")
        model = TimesFMWrapper(model_name="google/timesfm-1.0-200m")

        # Generate forecast
        print("Generating forecast...")
        result = model.forecast(
            context=train_data,
            horizon=50
        )

        print(f"\n✓ Forecast generated successfully!")
        print(f"  Forecast shape: {result['forecast'].shape}")
        print(f"  Forecast range: [{result['forecast'].min():.2f}, {result['forecast'].max():.2f}]")

        return True

    except ImportError as e:
        print(f"\n✗ TimesFM not available: {e}")
        print("  Install with: pip install timesfm")
        return False
    except Exception as e:
        print(f"\n✗ TimesFM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_simple():
    """Test ensemble forecaster"""
    print("\n" + "="*60)
    print("Testing Ensemble Forecaster")
    print("="*60)

    try:
        from foundation_models import EnsembleForecaster

        # Create simple sine wave
        t = np.linspace(0, 4*np.pi, 200)
        train_data = np.sin(t)

        # Try with Chronos only first (more likely to be available)
        print("\nLoading ensemble with Chronos...")
        ensemble = EnsembleForecaster(
            models=['chronos'],
            chronos_model="amazon/chronos-t5-tiny"
        )

        # Generate forecast
        print("Generating ensemble forecast...")
        result = ensemble.forecast(
            context=train_data,
            horizon=50,
            strategy='average',
            num_samples=20
        )

        print(f"\n✓ Ensemble forecast generated successfully!")
        print(f"  Forecast shape: {result['forecast'].shape}")
        print(f"  Models used: {result['models_used']}")
        print(f"  Ensemble strategy: {result['ensemble_strategy']}")
        print(f"  Uncertainty range: [{result['uncertainty'].min():.2f}, {result['uncertainty'].max():.2f}]")

        # Test model agreement (if multiple models)
        if len(result['individual_forecasts']) > 1:
            agreement = ensemble.get_model_agreement(result)
            print(f"  Model agreement: {agreement:.2f}")

        return True

    except Exception as e:
        print(f"\n✗ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FOUNDATION MODEL SIMPLE TESTS")
    print("="*60)
    print("\nThese tests verify that foundation models can be loaded and run.")
    print("Install requirements: pip install chronos-forecasting timesfm")

    results = {}

    # Test Chronos (most likely to work)
    results['chronos'] = test_chronos_simple()

    # Test TimesFM
    results['timesfm'] = test_timesfm_simple()

    # Test Ensemble
    results['ensemble'] = test_ensemble_simple()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Foundation models are working.")
        return 0
    elif total_passed > 0:
        print("\n⚠️  Some tests passed. Install missing dependencies.")
        return 0
    else:
        print("\n❌ All tests failed. Check installation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
