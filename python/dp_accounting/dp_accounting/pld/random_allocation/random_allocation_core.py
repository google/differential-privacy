"""Random-allocation composition internals (bridge, realizations, Gaussian path)."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.pld.pld_pmf import DensePLDPmf
from scipy import stats

from .random_allocation_convolution import (
    fft_convolve,
    fft_self_convolve,
    geometric_convolve,
    geometric_self_convolve,
)
from .random_allocation_distributions import (
    DenseDiscreteDist,
    PLDRealization,
    discretize_aligned_grid,
    discretize_continuous_grid,
    rediscretize_dist,
)
from .random_allocation_types import (
    BoundType,
    Direction,
    SpacingType,
)
from .random_allocation_utils import (
    calc_pld_dual,
    exp_linear_to_geometric,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
    validate_allocation_params,
    validate_bound_type,
    validate_discretization_params,
)

def linear_dist_to_dp_accounting_pmf(
    *,
    dist: DenseDiscreteDist,
    pessimistic_estimate: bool = True,
) -> DensePLDPmf:
    """Convert a linear-grid loss PMF to a dp_accounting PMF.

    Args:
        dist: Linear-grid loss distribution compatible with dp_accounting.
            Must be a linear DenseDiscreteDist. ``x_min`` is rounded to the
            nearest step multiple when forming dp_accounting's integer
            ``lower_loss`` index.
        pessimistic_estimate: Whether to use pessimistic estimate in dp_accounting.

    Returns:
        dp_accounting DensePLDPmf with infinity mass taken from dist.p_max.
    """
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.LINEAR):
        raise TypeError(
            f"linear_dist_to_dp_accounting_pmf requires DenseDiscreteDist, got {type(dist)}."
        )

    base_index = int(np.rint(dist.x_min / dist.step))
    return DensePLDPmf(
        discretization=dist.step,
        lower_loss=base_index,
        probs=dist.prob_arr.astype(np.float64),
        infinity_mass=dist.p_max,
        pessimistic_estimate=pessimistic_estimate,
    )

def realization_remove_base_distributions(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Prepare remove-direction factors from a loss-space realization.

    Algorithm 1 (`rand-alloc-rem`) in Appendix C.

    Args:
        realization: REMOVE-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.

    Returns:
        Tuple ``(base, dual_base)`` aligned to the requested linear grid.

    """
    # Since dual can be derived only from a PLD realization, discretization can
    # come first for DOMINATES, but dual derivation must come first for IS_DOMINATED.
    if bound_type == BoundType.DOMINATES:
        # Avoid inflating the grid when the target is finer than the original one.
        effective_disc = max(realization.step, loss_discretization)
        coarsened_base = rediscretize_dist(
            dist=realization,
            tail_truncation=tail_truncation,
            loss_discretization=effective_disc,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
        if not (
            isinstance(coarsened_base, DenseDiscreteDist)
            and coarsened_base.spacing_type == SpacingType.LINEAR
        ):
            _st = getattr(coarsened_base, "spacing_type", "?")
            raise TypeError(
                "Expected DenseDiscreteDist with LINEAR spacing, "
                f"got {type(coarsened_base).__name__} with spacing {_st}"
            )
        base_realization = PLDRealization.from_linear_dist(coarsened_base)
        neg_dual_dist = negate_reverse_linear_distribution(calc_pld_dual(base_realization))
        return base_realization, neg_dual_dist

    # Lower-bound truncation can move left-tail mass into p_min and must consume
    # any +inf mass before exp-space composition, so keep the lower path on the
    # plain DenseDiscreteDist rediscretization route unconditionally.
    dual_realization = calc_pld_dual(realization)
    neg_dual_linear = negate_reverse_linear_distribution(dual_realization)
    # Avoid inflating the grid when the target is finer than the original one.
    effective_disc = max(realization.step, loss_discretization)
    lower_realization_input = DenseDiscreteDist(
        x_min=realization.x_min,
        step=realization.step,
        prob_arr=realization.prob_arr.copy(),
        p_min=realization.p_min,
        p_max=realization.p_max,
    )
    lower_base_dist = rediscretize_dist(
        dist=lower_realization_input,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not (
        isinstance(lower_base_dist, DenseDiscreteDist)
        and lower_base_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(lower_base_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(lower_base_dist).__name__} with spacing {_st}"
        )
    neg_dual_dist = rediscretize_dist(
        dist=neg_dual_linear,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not (
        isinstance(neg_dual_dist, DenseDiscreteDist)
        and neg_dual_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(neg_dual_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(neg_dual_dist).__name__} with spacing {_st}"
        )
    return lower_base_dist, neg_dual_dist


def realization_add_base_distribution(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Prepare add-direction factors from a loss-space realization.

    Algorithm 2 (`rand-alloc-add`) in Appendix C.

    Args:
        realization: ADD-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.

    Returns:
        One ADD loss factor aligned to the requested linear grid.

    """
    # Avoid inflating the grid when the target is finer than the original one.
    effective_disc = max(realization.step, loss_discretization)
    coarsened = rediscretize_dist(
        dist=realization,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not (
        isinstance(coarsened, DenseDiscreteDist) and coarsened.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(coarsened, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(coarsened).__name__} with spacing {_st}"
        )
    return coarsened

# =============================================================================
# Public API
# =============================================================================


def allocation_full_pld(
    *,
    compute_base_pld_remove: Callable[..., DenseDiscreteDist],
    compute_base_pld_add: Callable[..., DenseDiscreteDist],
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Orchestrate full allocation PLD construction for both directions.

    This function builds REMOVE and ADD directional PLDs via
    ``allocation_directional_pld(...)`` and then converts them to the final
    ``dp_accounting`` PLD object.
    """
    # Input validation
    validate_allocation_params(num_steps, num_selected, num_epochs)
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)

    remove_dist = allocation_directional_pld(
        compute_base_pld=compute_base_pld_remove,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    add_dist = allocation_directional_pld(
        compute_base_pld=compute_base_pld_add,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    return _compose_full_pld(
        remove_dist=remove_dist,
        add_dist=add_dist,
        bound_type=bound_type,
    )


def allocation_directional_pld(
    *,
    compute_base_pld: Callable[..., DenseDiscreteDist],
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build one-direction allocation PLD with adaptive floor/ceil decomposition.

    For divisible ``num_steps / num_selected``, this builds one component. For
    non-divisible cases, it builds floor and ceil components via
    ``_allocation_directional_pld_core(...)`` and combines them with one final
    ``fft_convolve(...)``.
    """
    # Input validation
    validate_allocation_params(num_steps, num_selected, num_epochs)
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
    new_num_steps_floor = int(num_steps // num_selected)
    if new_num_steps_floor < 1:
        raise ValueError("num_steps must be >= num_selected")
    num_epochs_remainder = num_steps - num_selected * new_num_steps_floor
    new_num_steps_ceil = new_num_steps_floor + 1
    new_num_epochs_floor = (num_selected - num_epochs_remainder) * num_epochs
    new_num_epochs_ceil = num_epochs_remainder * num_epochs
    # Tail: active tail-consuming ops = one _allocation_directional_pld_core per component
    # plus one fft_convolve when both components are active, giving 2*component_count - 1
    # ops total (1 for component_count=1, 3 for component_count=2).  After
    # tail_truncation /= (2*component_count - 1), each op consumes at most the rescaled
    # budget, and all ops together sum to <= (2*component_count - 1) * rescaled = tail_truncation.
    component_count = int(new_num_epochs_floor > 0) + int(new_num_epochs_ceil > 0)
    tail_truncation /= 2 * component_count - 1

    dist_floor = None
    dist_ceil = None
    if new_num_epochs_floor > 0:
        dist_floor = _allocation_directional_pld_core(
            compute_base_pld=compute_base_pld,
            num_steps=new_num_steps_floor,
            num_epochs=new_num_epochs_floor,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    if new_num_epochs_ceil > 0:
        dist_ceil = _allocation_directional_pld_core(
            compute_base_pld=compute_base_pld,
            num_steps=new_num_steps_ceil,
            num_epochs=new_num_epochs_ceil,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )

    if dist_floor is None:
        if dist_ceil is None:
            raise RuntimeError(
                "allocation_directional_pld failed to build either floor or ceil component"
            )
        return dist_ceil
    if dist_ceil is None:
        return dist_floor
    if dist_floor.step != dist_ceil.step:
        raise ValueError(
            "Cannot convolve floor and ceil allocation components with different "
            f"grid steps: {dist_floor.step:.12g} vs {dist_ceil.step:.12g}."
        )
    return fft_convolve(
        dist_1=dist_floor,
        dist_2=dist_ceil,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


def geometric_allocation_pld_base_remove(
    *,
    base_distributions_creation: Callable[..., tuple[DenseDiscreteDist, DenseDiscreteDist]],
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build the REMOVE component PLD via exp-space geometric composition.

    The callback ``base_distributions_creation`` provides one-step
    ``(base, neg_dual_base)`` factors, which are shifted and composed.
    """
    # Input validation
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
    # For num_steps > 1 there are active convolution stages beyond base construction.
    # For num_steps == 1 neither convolution stages nor Phases 2/3 execute, so no
    # tail-budget division is needed.
    if num_steps > 1:
        # Tail: three phases each receive tail_truncation / 3 after division.  Contributions
        # to output p_max (letting T = num_steps, budget/3 = rescaled per-phase share):
        #   Phase 1 (base_distributions_creation): base_factor_tail_truncation = budget/(3T).
        #            neg_dual is self-convolved T-1 times, base convolved once; total amplified
        #            contribution = ((T-1) + 1) * budget/(3T) = budget/3.
        #   Phase 2 (geometric_self_convolve, T-1): called with budget/3 -> <= budget/3.
        #   Phase 3 (geometric_convolve): called with budget/3 -> <= budget/3.
        # Sum <= budget/3 + budget/3 + budget/3 = budget
        tail_truncation /= 3
    # Each base factor is one of num_steps terms in the final product; its individual
    # tail error is amplified by num_steps through self-convolution, so scale its budget
    # down by num_steps so the amplified contribution stays within the phase budget.
    base_factor_tail_truncation = tail_truncation / num_steps

    base, neg_dual_base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    # For num_steps == 1 the centering shift is log(1) = 0 and the exp/log round-trip
    # is an identity, so base is already the final result.
    if num_steps == 1:
        return base

    # Subtract the average loss
    log_num_steps = float(np.log(num_steps))
    centered_neg_dual = DenseDiscreteDist(
        x_min=neg_dual_base.x_min - log_num_steps,
        step=neg_dual_base.step,
        prob_arr=neg_dual_base.prob_arr.copy(),
        p_min=neg_dual_base.p_min,
        p_max=neg_dual_base.p_max,
    )
    centered_base = DenseDiscreteDist(
        x_min=base.x_min - log_num_steps,
        step=base.step,
        prob_arr=base.prob_arr.copy(),
        p_min=base.p_min,
        p_max=base.p_max,
    )

    # Factor preparation in exp-space.
    exp_neg_dual = exp_linear_to_geometric(centered_neg_dual)
    exp_base = exp_linear_to_geometric(centered_base)

    # V_{t-1} <- self-conv(V1, t-1, ...).
    exp_convolved_dual = geometric_self_convolve(
        dist=exp_neg_dual,
        T=num_steps - 1,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    # U_t <- conv(V_{t-1}, U1, ...).
    exp_convolved = geometric_convolve(
        dist_1=exp_convolved_dual,
        dist_2=exp_base,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    # L_t <- log(U_t).
    return log_geometric_to_linear(exp_convolved)


def geometric_allocation_pld_base_add(
    *,
    base_distributions_creation: Callable[..., DenseDiscreteDist],
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build the ADD component PLD via exp-space geometric self-composition.

    The callback ``base_distributions_creation`` provides the one-step ADD
    factor, which is shifted and composed before mapping back to linear loss.
    """
    # Input validation
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    # For num_steps > 1 there are active convolution stages beyond base construction.
    # For num_steps == 1 neither convolution stages nor Phase 2 execute, so no
    # tail-budget division is needed.
    if num_steps > 1:
        # Tail: after dividing by 2, each of the two phases receives tail_truncation:
        #   Phase 1 (base creation): tail_truncation / num_steps per call; num_steps-fold
        #            self-convolution amplifies the contribution back to tail_truncation.
        #   Phase 2 (geometric_self_convolve, num_steps): tail_truncation directly.
        # Sum: tail_truncation + tail_truncation = 2 * tail_truncation = original ✓
        tail_truncation /= 2
    # Each base factor's tail error is amplified by num_steps through self-convolution,
    # so scale its budget down by num_steps so the amplified contribution stays within the
    # phase budget.
    base_factor_tail_truncation = tail_truncation / num_steps

    base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    # For num_steps == 1 the centering shift is log(1) = 0 and the exp/log round-trip
    # is an identity, so base is already the final result.
    if num_steps == 1:
        return base

    log_num_steps = float(np.log(num_steps))

    neg_base = negate_reverse_linear_distribution(base)
    centered_neg_base = DenseDiscreteDist(
        x_min=neg_base.x_min - log_num_steps,
        step=neg_base.step,
        prob_arr=neg_base.prob_arr.copy(),
        p_min=neg_base.p_min,
        p_max=neg_base.p_max,
    )

    # Factor preparation in exp-space.
    exp_base = exp_linear_to_geometric(centered_neg_base)
    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    # U_t <- self-conv(U, t, lower).
    exp_convolved = geometric_self_convolve(
        dist=exp_base,
        T=num_steps,
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
    )
    # L_t <- -log(U_t).
    log_dist = log_geometric_to_linear(exp_convolved)
    return negate_reverse_linear_distribution(log_dist)


def gaussian_allocation_pld_core(
    *,
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    direction: Direction,
    sigma: float,
) -> DenseDiscreteDist:
    """Route one Gaussian component through the GEOM backend.

    This is the Gaussian-side orchestrator used by the shared allocation core.
    """
    # Input validation
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
    if direction not in (Direction.ADD, Direction.REMOVE):
        raise ValueError(f"Invalid direction: {direction}")

    return _gaussian_allocation_geom(
        num_steps=num_steps,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        direction=direction,
        sigma=sigma,
    )


def _allocation_directional_pld_core(
    *,
    compute_base_pld: Callable[..., DenseDiscreteDist],
    num_steps: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build and finalize one floor/ceil decomposition component.

    This function derives component-level budgets, calls
    ``compute_base_pld(...)``, trims tails before and after epoch
    composition, and returns the resulting linear distribution.
    """
    # Tail: divide by 3; each of the three phases receives tail_truncation (the divided value):
    #   Phase 1 (base creation, amplified by num_epochs): base construction and
    #            truncate_edges each use base_tail_truncation.  Amplified total:
    #            num_epochs * 2 * tail_truncation / (2 * num_epochs) = tail_truncation.
    #   Phase 2 (fft_self_convolve, T=num_epochs): tail_truncation directly
    #            (skipped if num_epochs=1).
    #   Phase 3 (final truncate_edges): tail_truncation directly.
    # Sum: tail_truncation + tail_truncation + tail_truncation = 3 * tail_truncation = original ✓
    tail_truncation /= 3
    base_tail_truncation = tail_truncation / (2 * num_epochs)

    base_dist = compute_base_pld(
        num_steps=num_steps,
        loss_discretization=loss_discretization,
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
    )
    prepared_base_dist = base_dist.truncate_edges(
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
    )
    if not (
        isinstance(prepared_base_dist, DenseDiscreteDist)
        and prepared_base_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(prepared_base_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(prepared_base_dist).__name__} with spacing {_st}"
        )

    if num_epochs == 1:
        composed_dist = prepared_base_dist
    else:
        composed_dist = fft_self_convolve(
            dist=prepared_base_dist,
            T=num_epochs,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            use_direct=True,
        )
    final_dist = composed_dist.truncate_edges(
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    if not (
        isinstance(final_dist, DenseDiscreteDist) and final_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(final_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(final_dist).__name__} with spacing {_st}"
        )
    return final_dist


def _compose_full_pld(
    *,
    remove_dist: DenseDiscreteDist | None,
    add_dist: DenseDiscreteDist | None,
    bound_type: BoundType,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Convert remove/add directional PLDs into a ``dp_accounting`` PLD.

    Args:
        remove_dist: REMOVE-direction linear PLD.
        add_dist: Optional ADD-direction linear PLD.
        bound_type: Bound direction used for pessimistic conversion.

    Returns:
        A ``dp_accounting`` privacy loss distribution.

    """
    if remove_dist is None:
        raise ValueError(
            "PLD construction requires remove-direction distribution. "
            "Provide remove_realization or use both directions."
        )
    pessimistic_estimate = bound_type == BoundType.DOMINATES
    pmf_remove = linear_dist_to_dp_accounting_pmf(
        dist=remove_dist,
        pessimistic_estimate=pessimistic_estimate,
    )
    if add_dist is None:
        return privacy_loss_distribution.PrivacyLossDistribution(
            pmf_remove=pmf_remove,
        )
    pmf_add = linear_dist_to_dp_accounting_pmf(
        dist=add_dist,
        pessimistic_estimate=pessimistic_estimate,
    )
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove=pmf_remove,
        pmf_add=pmf_add,
    )


# =============================================================================
# Internal GEOM Route
# =============================================================================


def _gaussian_allocation_geom(
    *,
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    direction: Direction,
    sigma: float,
) -> DenseDiscreteDist:
    """GEOM path intentionally mirrors realization path after base creation.

    Both call geometric_allocation_PLD_base_* with identical wiring.

    """
    if direction == Direction.ADD:
        return geometric_allocation_pld_base_add(
            base_distributions_creation=partial(
                _gaussian_add_geom_loss_factor,
                sigma=sigma,
            ),
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    if direction == Direction.REMOVE:
        return geometric_allocation_pld_base_remove(
            base_distributions_creation=partial(
                _gaussian_remove_geom_loss_factors,
                sigma=sigma,
            ),
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    raise ValueError(f"Invalid direction: {direction}")


def _gaussian_remove_geom_loss_factors(
    *,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    sigma: float,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Build REMOVE GEOM one-step PLD factors as ``(base, dual_base)``."""
    sigma_inv = 1.0 / sigma
    factor_tail_truncation = tail_truncation / 2

    dual_norm_mean = -(sigma_inv**2) / 2
    base_norm_mean = sigma_inv**2 / 2
    exp_dual = stats.lognorm(s=sigma_inv, scale=np.exp(dual_norm_mean))
    exp_base = stats.lognorm(s=sigma_inv, scale=np.exp(base_norm_mean))

    geom_step = float(loss_discretization)
    dual_x_min = float(exp_dual.ppf(factor_tail_truncation))
    dual_x_max = float(exp_dual.isf(factor_tail_truncation))
    base_x_min = float(exp_base.ppf(factor_tail_truncation))
    base_x_max = float(exp_base.isf(factor_tail_truncation))
    dual_grid = discretize_aligned_grid(
        x_min=dual_x_min,
        x_max=dual_x_max,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=geom_step,
    )
    base_grid = discretize_aligned_grid(
        x_min=base_x_min,
        x_max=base_x_max,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=geom_step,
    )

    dual_factor_dist = discretize_continuous_grid(
        dist=exp_dual,
        grid=dual_grid,
        bound_type=bound_type,
        PMF_min_increment=factor_tail_truncation,
    )
    if not (
        isinstance(dual_factor_dist, DenseDiscreteDist)
        and dual_factor_dist.spacing_type == SpacingType.GEOMETRIC
    ):
        _st = getattr(dual_factor_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(dual_factor_dist).__name__} with spacing {_st}"
        )

    base_factor_dist = discretize_continuous_grid(
        dist=exp_base,
        grid=base_grid,
        bound_type=bound_type,
        PMF_min_increment=factor_tail_truncation,
    )
    if not (
        isinstance(base_factor_dist, DenseDiscreteDist)
        and base_factor_dist.spacing_type == SpacingType.GEOMETRIC
    ):
        _st = getattr(base_factor_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(base_factor_dist).__name__} with spacing {_st}"
        )

    dual_loss_factor = log_geometric_to_linear(dual_factor_dist)
    base_loss_factor = log_geometric_to_linear(base_factor_dist)
    # geometric_allocation_pld_base_remove expects (base, dual_base).
    return base_loss_factor, dual_loss_factor


def _gaussian_add_geom_loss_factor(
    *,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    sigma: float,
) -> DenseDiscreteDist:
    """Build ADD GEOM one-step linear PLD factor."""
    sigma_inv = 1.0 / sigma

    base_lognorm = stats.lognorm(s=sigma_inv, scale=np.exp(+(sigma_inv**2) / 2))
    geom_step = float(loss_discretization)

    base_x_min = float(base_lognorm.ppf(tail_truncation))
    base_x_max = float(base_lognorm.isf(tail_truncation))
    base_grid = discretize_aligned_grid(
        x_min=base_x_min,
        x_max=base_x_max,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=geom_step,
    )
    base_dist = discretize_continuous_grid(
        dist=base_lognorm,
        grid=base_grid,
        bound_type=bound_type,
        PMF_min_increment=tail_truncation,
    )
    if not (
        isinstance(base_dist, DenseDiscreteDist) and base_dist.spacing_type == SpacingType.GEOMETRIC
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(base_dist).__name__} with spacing {getattr(base_dist, 'spacing_type', '?')}"
    )
    return log_geometric_to_linear(base_dist)
