"""Delivery API tests: only PLD builders are public."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from dp_accounting.pld import privacy_loss_distribution

from dp_accounting.pld.random_allocation import random_allocation_api
from dp_accounting.pld.random_allocation import random_allocation_distributions
from dp_accounting.pld.random_allocation import random_allocation_types


def _simple_realization() -> random_allocation_distributions.PLDRealization:
    return random_allocation_distributions.PLDRealization(
        x_min=0.0,
        step=1.0,
        prob_arr=np.array([0.7, 0.3], dtype=np.float64),
        p_max=0.0,
    )


def test_gaussian_allocation_pld_returns_dp_accounting_pld():
    params = random_allocation_types.PrivacyParams(
        sigma=2.0, num_steps=5, num_selected=1, num_epochs=1
    )
    config = random_allocation_types.AllocationSchemeConfig(
        value_discretization_interval=0.05, tail_truncation=1e-6
    )

    pld = random_allocation_api.gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=random_allocation_types.BoundType.DOMINATES,
    )

    assert isinstance(pld, privacy_loss_distribution.PrivacyLossDistribution)


def test_general_allocation_pld_returns_dp_accounting_pld():
    config = random_allocation_types.AllocationSchemeConfig(
        value_discretization_interval=0.05, tail_truncation=1e-6
    )

    pld = random_allocation_api.general_allocation_pld(
        num_steps=5,
        num_selected=1,
        num_epochs=1,
        remove_realization=_simple_realization(),
        add_realization=_simple_realization(),
        config=config,
        bound_type=random_allocation_types.BoundType.DOMINATES,
    )

    assert isinstance(pld, privacy_loss_distribution.PrivacyLossDistribution)


def test_non_pld_exports_are_not_exposed():
    import importlib

    package = importlib.import_module(__package__)

    assert not hasattr(package, "gaussian_allocation_epsilon_range")
    assert not hasattr(package, "gaussian_distribution")
    assert not hasattr(package, "subsample_pld")


def test_general_allocation_rejects_non_realization_inputs():
    config = random_allocation_types.AllocationSchemeConfig()
    with pytest.raises(TypeError, match="remove_realization must be PLDRealization"):
        random_allocation_api.general_allocation_pld(
            num_steps=2,
            num_selected=1,
            num_epochs=1,
            remove_realization=cast(Any, object()),
            add_realization=_simple_realization(),
            config=config,
        )
