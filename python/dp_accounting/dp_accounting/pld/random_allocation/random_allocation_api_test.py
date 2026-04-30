"""Delivery API tests: only PLD builders are public."""

from __future__ import annotations

import numpy as np
import pytest
from dp_accounting.pld import privacy_loss_distribution

from .random_allocation_api import gaussian_allocation_pld, general_allocation_pld
from .random_allocation_distributions import PLDRealization
from .random_allocation_types import AllocationSchemeConfig, BoundType, PrivacyParams


def _simple_realization() -> PLDRealization:
    return PLDRealization(
        x_min=0.0,
        step=1.0,
        prob_arr=np.array([0.7, 0.3], dtype=np.float64),
        p_max=0.0,
    )


def test_gaussian_allocation_pld_returns_dp_accounting_pld():
    params = PrivacyParams(sigma=2.0, num_steps=5, num_selected=1, num_epochs=1)
    config = AllocationSchemeConfig(loss_discretization=0.05, tail_truncation=1e-6)

    pld = gaussian_allocation_pld(params=params, config=config, bound_type=BoundType.DOMINATES)

    assert isinstance(pld, privacy_loss_distribution.PrivacyLossDistribution)


def test_general_allocation_pld_returns_dp_accounting_pld():
    config = AllocationSchemeConfig(loss_discretization=0.05, tail_truncation=1e-6)

    pld = general_allocation_pld(
        num_steps=5,
        num_selected=1,
        num_epochs=1,
        remove_realization=_simple_realization(),
        add_realization=_simple_realization(),
        config=config,
        bound_type=BoundType.DOMINATES,
    )

    assert isinstance(pld, privacy_loss_distribution.PrivacyLossDistribution)


def test_non_pld_exports_are_not_exposed():
    import importlib

    package = importlib.import_module(__package__)

    assert not hasattr(package, "gaussian_allocation_epsilon_range")
    assert not hasattr(package, "gaussian_distribution")
    assert not hasattr(package, "subsample_pld")


def test_general_allocation_rejects_non_realization_inputs():
    config = AllocationSchemeConfig()
    with pytest.raises(TypeError, match="remove_realization must be PLDRealization"):
        general_allocation_pld(
            num_steps=2,
            num_selected=1,
            num_epochs=1,
            remove_realization=object(),
            add_realization=_simple_realization(),
            config=config,
        )
