"""Tests for foundation model wrappers"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Check if foundation model libraries are installed
try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False

try:
    import chronos
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False


@pytest.mark.skipif(not TIMESFM_AVAILABLE, reason="TimesFM not installed")
def test_timesfm_forecast():
    """Test TimesFM wrapper"""
    from src.foundation_models import TimesFMWrapper

    model = TimesFMWrapper()

    # Synthetic data: simple sine wave
    train_data = np.sin(np.linspace(0, 4*np.pi, 200))

    # Generate forecast
    result = model.forecast(train_data, horizon=20)

    # Verify output format
    assert 'forecast' in result, "Missing 'forecast' key"
    assert 'model' in result, "Missing 'model' key"
    assert result['model'] == 'timesfm'
    assert len(result['forecast']) == 20, f"Expected 20 forecasts, got {len(result['forecast'])}"
    assert isinstance(result['forecast'], np.ndarray)

    print(f"✓ TimesFM forecast shape: {result['forecast'].shape}")


@pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="Chronos not installed")
def test_chronos_forecast():
    """Test Chronos wrapper"""
    from src.foundation_models import ChronosWrapper

    model = ChronosWrapper()

    # Synthetic data
    train_data = np.sin(np.linspace(0, 4*np.pi, 200))

    # Generate probabilistic forecast
    result = model.forecast(train_data, horizon=20, num_samples=50)

    # Verify output format
    assert 'forecast' in result, "Missing 'forecast' key"
    assert 'quantiles' in result, "Missing 'quantiles' key"
    assert 'model' in result, "Missing 'model' key"
    assert result['model'] == 'chronos'

    # Check forecast shape
    assert len(result['forecast']) == 20

    # Check quantiles
    assert 'P50' in result['quantiles'], "Missing P50 quantile"
    assert 'P90' in result['quantiles'], "Missing P90 quantile"
    assert len(result['quantiles']['P50']) == 20

    # Check samples
    assert 'samples' in result
    assert result['samples'].shape == (50, 20), f"Expected (50, 20), got {result['samples'].shape}"

    print(f"✓ Chronos forecast shape: {result['forecast'].shape}")
    print(f"✓ Chronos quantiles: {list(result['quantiles'].keys())}")


@pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="Chronos not installed")
def test_ensemble_forecaster():
    """Test ensemble forecasting"""
    from src.foundation_models import EnsembleForecaster

    # Use only Chronos for testing (TimesFM might not be available)
    ensemble = EnsembleForecaster(models=['chronos'])

    # Synthetic data
    train_data = np.sin(np.linspace(0, 4*np.pi, 200))

    # Generate ensemble forecast
    result = ensemble.forecast(train_data, horizon=20, strategy='average', num_samples=50)

    # Verify output format
    assert 'forecast' in result
    assert 'quantiles' in result or len(result['individual_forecasts']) == 1
    assert 'individual_forecasts' in result
    assert 'uncertainty' in result
    assert 'ensemble_strategy' in result

    # Check forecast shape
    assert len(result['forecast']) == 20

    # Check models used
    assert 'chronos' in result['models_used']

    print(f"✓ Ensemble forecast shape: {result['forecast'].shape}")
    print(f"✓ Models used: {result['models_used']}")


def test_foundation_model_imports():
    """Test that foundation model module can be imported"""
    try:
        from src.foundation_models import (
            FoundationModel,
            TimesFMWrapper,
            ChronosWrapper,
            EnsembleForecaster
        )
        print("✓ All foundation model classes imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import foundation models: {e}")


def test_ensemble_no_models():
    """Test that ensemble raises error with no models"""
    from src.foundation_models import EnsembleForecaster

    with pytest.raises(ValueError, match="Must specify at least one model"):
        EnsembleForecaster(models=[])


if __name__ == '__main__':
    # Run tests
    print("Running foundation model tests...\n")

    print("Test 1: Imports")
    test_foundation_model_imports()

    print("\nTest 2: Ensemble validation")
    test_ensemble_no_models()
    print("✓ Ensemble validation works")

    if CHRONOS_AVAILABLE:
        print("\nTest 3: Chronos forecast")
        test_chronos_forecast()
    else:
        print("\nTest 3: Chronos forecast - SKIPPED (not installed)")

    if TIMESFM_AVAILABLE:
        print("\nTest 4: TimesFM forecast")
        test_timesfm_forecast()
    else:
        print("\nTest 4: TimesFM forecast - SKIPPED (not installed)")

    if CHRONOS_AVAILABLE or TIMESFM_AVAILABLE:
        print("\nTest 5: Ensemble forecaster")
        test_ensemble_forecaster()
    else:
        print("\nTest 5: Ensemble forecaster - SKIPPED (no models installed)")

    print("\n" + "="*60)
    print("✓ All available tests passed!")
    print("="*60)
