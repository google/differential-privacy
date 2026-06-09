"""Public API surface for random-allocation PLD construction."""

from __future__ import annotations

from functools import partial

from dp_accounting.pld import privacy_loss_distribution

from dp_accounting.pld.random_allocation import random_allocation_core
from dp_accounting.pld.random_allocation import random_allocation_utils
from dp_accounting.pld.random_allocation.random_allocation_distributions import PLDRealization
from dp_accounting.pld.random_allocation.random_allocation_types import (
    AllocationSchemeConfig,
    BoundType,
    Direction,
    PrivacyParams,
)


def gaussian_allocation_pld(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Compute upper / lower PLD for random-allocation with the Gaussian mechanism.

    Args:
        params: Privacy parameters describing noise scale, number of steps,
            and optional delta/epsilon query target.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for both privacy directions.

    """
    # Input validation
    random_allocation_utils._validate_privacy_params(params)
    random_allocation_utils._validate_allocation_scheme_config(config)
    random_allocation_utils._validate_bound_type(bound_type)

    compute_base_pld_remove = partial(
        random_allocation_core._gaussian_allocation_pld_core,
        direction=Direction.REMOVE,
        sigma=params.sigma,
    )
    compute_base_pld_add = partial(
        random_allocation_core._gaussian_allocation_pld_core,
        direction=Direction.ADD,
        sigma=params.sigma,
    )
    return random_allocation_core._allocation_full_pld(
        compute_base_pld_remove=compute_base_pld_remove,
        compute_base_pld_add=compute_base_pld_add,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs,
        loss_discretization=config.value_discretization_interval,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )


def general_allocation_pld(
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Build a random-allocation PLD from explicit PLD realizations.

    Args:
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Explicit remove-direction PLD realization.
        add_realization: Explicit add-direction PLD realization.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for the composed realization.

    Notes:
        The delivery package always uses the geometric convolution path.

    """
    # Input validation
    random_allocation_utils._validate_allocation_params(
        num_steps, num_selected, num_epochs
    )
    if not isinstance(remove_realization, PLDRealization):
        raise TypeError(
            f"remove_realization must be PLDRealization, got {type(remove_realization)}"
        )
    if not isinstance(add_realization, PLDRealization):
        raise TypeError(
            f"add_realization must be PLDRealization, got {type(add_realization)}"
        )
    random_allocation_utils._validate_allocation_scheme_config(config)
    random_allocation_utils._validate_bound_type(bound_type)

    compute_base_pld_remove = partial(
        random_allocation_core._geometric_allocation_pld_base_remove,
        base_distributions_creation=partial(
            random_allocation_core._realization_remove_base_distributions,
            realization=remove_realization,
        ),
    )
    compute_base_pld_add = partial(
        random_allocation_core._geometric_allocation_pld_base_add,
        base_distributions_creation=partial(
            random_allocation_core._realization_add_base_distribution,
            realization=add_realization,
        ),
    )
    return random_allocation_core._allocation_full_pld(
        compute_base_pld_remove=compute_base_pld_remove,
        compute_base_pld_add=compute_base_pld_add,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=config.value_discretization_interval,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )
