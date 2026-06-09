"""Delivery API tests: only PLD builders are public."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest
from dp_accounting.pld import privacy_loss_distribution

from dp_accounting.pld.random_allocation import ra_api
from dp_accounting.pld.random_allocation import ra_distributions
from dp_accounting.pld.random_allocation import ra_types


def _simple_realization() -> ra_distributions.PLDRealization:
    return ra_distributions.PLDRealization(
        x_min=0.0,
        step=1.0,
        prob_arr=np.array([0.7, 0.3], dtype=np.float64),
        p_max=0.0,
    )


class RandomAllocationApiTest(absltest.TestCase):

    def test_gaussian_allocation_pld_returns_dp_accounting_pld(self):
        params = ra_types.PrivacyParams(
            sigma=2.0, num_steps=5, num_selected=1, num_epochs=1
        )
        config = ra_types.AllocationSchemeConfig(
            value_discretization_interval=0.05, tail_truncation=1e-6
        )

        pld = ra_api.gaussian_allocation_pld(
            params=params,
            config=config,
            bound_type=ra_types.BoundType.DOMINATES,
        )

        self.assertIsInstance(pld, privacy_loss_distribution.PrivacyLossDistribution)

    def test_general_allocation_pld_returns_dp_accounting_pld(self):
        config = ra_types.AllocationSchemeConfig(
            value_discretization_interval=0.05, tail_truncation=1e-6
        )

        pld = ra_api.general_allocation_pld(
            num_steps=5,
            num_selected=1,
            num_epochs=1,
            remove_realization=_simple_realization(),
            add_realization=_simple_realization(),
            config=config,
            bound_type=ra_types.BoundType.DOMINATES,
        )

        self.assertIsInstance(pld, privacy_loss_distribution.PrivacyLossDistribution)

    def test_non_pld_exports_are_not_exposed(self):
        import importlib

        package = importlib.import_module("dp_accounting.pld.random_allocation")

        self.assertFalse(hasattr(package, "gaussian_allocation_epsilon_range"))
        self.assertFalse(hasattr(package, "gaussian_distribution"))
        self.assertFalse(hasattr(package, "subsample_pld"))

    def test_general_allocation_rejects_non_realization_inputs(self):
        config = ra_types.AllocationSchemeConfig()
        with self.assertRaisesRegex(TypeError, "remove_realization must be PLDRealization"):
            ra_api.general_allocation_pld(
                num_steps=2,
                num_selected=1,
                num_epochs=1,
                remove_realization=cast(Any, object()),
                add_realization=_simple_realization(),
                config=config,
            )


if __name__ == "__main__":
    absltest.main()
