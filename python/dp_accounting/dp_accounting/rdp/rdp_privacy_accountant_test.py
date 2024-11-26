# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for rdp_privacy_accountant."""

import math
import sys

from absl.testing import absltest
from absl.testing import parameterized
import mpmath
import numpy as np

from dp_accounting import dp_event
from dp_accounting import privacy_accountant
from dp_accounting import privacy_accountant_test
from dp_accounting.rdp import rdp_privacy_accountant


def _get_test_rdp(event, count=1):
  accountant = rdp_privacy_accountant.RdpAccountant(orders=[2.71828])
  accountant.compose(event, count)
  return accountant._rdp[0]


def _log_float_mp(x):
  # Convert multi-precision input to float log space.
  if x >= sys.float_info.min:
    return float(mpmath.log(x))
  else:
    return -np.inf


def _compute_a_mp(sigma, q, alpha):
  """Compute A_alpha for arbitrary alpha by numerical integration."""

  def mu0(x):
    return mpmath.npdf(x, mu=0, sigma=sigma)

  def _mu_over_mu0(x, q, sigma):
    return (1 - q) + q * mpmath.exp((2 * x - 1) / (2 * sigma**2))

  def a_alpha_fn(z):
    return mu0(z) * _mu_over_mu0(z, q, sigma) ** alpha

  bounds = (-mpmath.inf, mpmath.inf)
  a_alpha, _ = mpmath.quad(a_alpha_fn, bounds, error=True, maxdegree=8)
  return a_alpha


def _compose_trees(noise_multiplier, step_counts, orders):
  accountant = rdp_privacy_accountant.RdpAccountant(
      orders, privacy_accountant.NeighboringRelation.REPLACE_SPECIAL
  )
  accountant.compose(
      dp_event.ComposedDpEvent([
          dp_event.SingleEpochTreeAggregationDpEvent(
              noise_multiplier, step_count
          )
          for step_count in step_counts
      ])
  )
  return accountant


def _compose_trees_single_epoch(noise_multiplier, step_counts, orders):
  accountant = rdp_privacy_accountant.RdpAccountant(
      orders, privacy_accountant.NeighboringRelation.REPLACE_SPECIAL
  )
  accountant.compose(
      dp_event.SingleEpochTreeAggregationDpEvent(noise_multiplier, step_counts)
  )
  return accountant


class RdpPrivacyAccountantTest(
    privacy_accountant_test.PrivacyAccountantTest, parameterized.TestCase
):

  def _make_test_accountants(self):
    return [
        rdp_privacy_accountant.RdpAccountant(
            [2.0], privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE
        ),
        rdp_privacy_accountant.RdpAccountant(
            [2.0], privacy_accountant.NeighboringRelation.REPLACE_ONE
        ),
        rdp_privacy_accountant.RdpAccountant(
            [2.0], privacy_accountant.NeighboringRelation.REPLACE_SPECIAL
        ),
    ]

  def test_supports(self):
    aor_accountant = rdp_privacy_accountant.RdpAccountant(
        [2.0], privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE
    )
    ro_accountant = rdp_privacy_accountant.RdpAccountant(
        [2.0], privacy_accountant.NeighboringRelation.REPLACE_ONE
    )
    rs_accountant = rdp_privacy_accountant.RdpAccountant(
        [2.0], privacy_accountant.NeighboringRelation.REPLACE_SPECIAL
    )

    event = dp_event.GaussianDpEvent(1.0)
    self.assertTrue(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(1.0), 6)
    self.assertTrue(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.ComposedDpEvent(
        [dp_event.GaussianDpEvent(1.0), dp_event.GaussianDpEvent(2.0)]
    )
    self.assertTrue(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0))
    self.assertTrue(aor_accountant.supports(event))
    self.assertFalse(ro_accountant.supports(event))

    composed_gaussian = dp_event.ComposedDpEvent(
        [dp_event.GaussianDpEvent(1.0), dp_event.GaussianDpEvent(2.0)]
    )
    event = dp_event.PoissonSampledDpEvent(0.1, composed_gaussian)
    self.assertTrue(aor_accountant.supports(event))
    self.assertFalse(ro_accountant.supports(event))

    event = dp_event.SampledWithoutReplacementDpEvent(
        1000, 10, dp_event.GaussianDpEvent(1.0)
    )
    self.assertFalse(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.SampledWithoutReplacementDpEvent(
        1000, 10, composed_gaussian
    )
    self.assertFalse(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.SampledWithReplacementDpEvent(
        1000, 10, dp_event.GaussianDpEvent(1.0)
    )
    self.assertFalse(aor_accountant.supports(event))
    self.assertFalse(ro_accountant.supports(event))

    event = dp_event.RandomizedResponseDpEvent(0.3, 15)
    self.assertTrue(rs_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))
    self.assertFalse(aor_accountant.supports(event))

  def test_rdp_composition(self):
    base_event = dp_event.GaussianDpEvent(3.14159)
    base_rdp = _get_test_rdp(base_event)

    rdp_with_count = _get_test_rdp(base_event, count=6)
    self.assertAlmostEqual(rdp_with_count, base_rdp * 6)

    rdp_with_self_compose = _get_test_rdp(
        dp_event.SelfComposedDpEvent(base_event, 6)
    )
    self.assertAlmostEqual(rdp_with_self_compose, base_rdp * 6)

    rdp_with_self_compose_and_count = _get_test_rdp(
        dp_event.SelfComposedDpEvent(base_event, 2), count=3
    )
    self.assertAlmostEqual(rdp_with_self_compose_and_count, base_rdp * 6)

    rdp_with_compose = _get_test_rdp(dp_event.ComposedDpEvent([base_event] * 6))
    self.assertAlmostEqual(rdp_with_compose, base_rdp * 6)

    rdp_with_compose_and_self_compose = _get_test_rdp(
        dp_event.ComposedDpEvent([
            dp_event.SelfComposedDpEvent(base_event, 1),
            dp_event.SelfComposedDpEvent(base_event, 2),
            dp_event.SelfComposedDpEvent(base_event, 3),
        ])
    )
    self.assertAlmostEqual(rdp_with_compose_and_self_compose, base_rdp * 6)

    base_event_2 = dp_event.GaussianDpEvent(1.61803)
    base_rdp_2 = _get_test_rdp(base_event_2)
    rdp_with_heterogeneous_compose = _get_test_rdp(
        dp_event.ComposedDpEvent([base_event, base_event_2])
    )
    self.assertAlmostEqual(
        rdp_with_heterogeneous_compose, base_rdp + base_rdp_2
    )

  @parameterized.parameters(
      dp_event.PoissonSampledDpEvent(0, dp_event.GaussianDpEvent(1.0)),
      dp_event.PoissonSampledDpEvent(0, dp_event.GaussianDpEvent(0.0)),
  )
  def test_zero_poisson_sample(self, event):
    accountant = rdp_privacy_accountant.RdpAccountant([3.14159])
    accountant.compose(event)
    self.assertEqual(accountant.get_epsilon(1e-10), 0)
    self.assertEqual(accountant.get_delta(1e-10), 0)

  def test_zero_fixed_batch_sample(self):
    accountant = rdp_privacy_accountant.RdpAccountant(
        [3.14159], privacy_accountant.NeighboringRelation.REPLACE_ONE
    )
    accountant.compose(
        dp_event.SampledWithoutReplacementDpEvent(
            1000, 0, dp_event.GaussianDpEvent(1.0)
        )
    )
    self.assertEqual(accountant.get_epsilon(1e-10), 0)
    self.assertEqual(accountant.get_delta(1e-10), 0)

  def test_epsilon_non_private_gaussian(self):
    accountant = rdp_privacy_accountant.RdpAccountant([3.14159])
    accountant.compose(dp_event.GaussianDpEvent(0))
    self.assertEqual(accountant.get_epsilon(1e-1), np.inf)

  def test_compute_rdp_gaussian(self):
    alpha = 3.14159
    sigma = 2.71828
    event = dp_event.GaussianDpEvent(sigma)
    accountant = rdp_privacy_accountant.RdpAccountant(orders=[alpha])
    accountant.compose(event)
    self.assertAlmostEqual(accountant._rdp[0], alpha / (2 * sigma**2))

  def test_compute_rdp_multi_gaussian(self):
    alpha = 3.14159
    sigma1, sigma2 = 2.71828, 6.28319

    rdp1 = alpha / (2 * sigma1**2)
    rdp2 = alpha / (2 * sigma2**2)
    rdp = rdp1 + rdp2

    accountant = rdp_privacy_accountant.RdpAccountant(orders=[alpha])
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            1.0,
            dp_event.ComposedDpEvent([
                dp_event.GaussianDpEvent(sigma1),
                dp_event.GaussianDpEvent(sigma2),
            ]),
        )
    )
    self.assertAlmostEqual(accountant._rdp[0], rdp)

  def test_effective_gaussian_noise_multiplier_basic(self):
    sigma = 2.71828
    event = dp_event.GaussianDpEvent(sigma)
    sigma_out = rdp_privacy_accountant._effective_gaussian_noise_multiplier(
        event
    )
    self.assertEqual(sigma_out, sigma)

  def test_effective_gaussian_noise_multiplier_composed(self):
    np.random.seed(0xBAD5EED)
    sigmas = np.random.uniform(size=(4,))

    event = dp_event.ComposedDpEvent([
        dp_event.GaussianDpEvent(sigmas[0]),
        dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(sigmas[1]), 3),
        dp_event.ComposedDpEvent([
            dp_event.GaussianDpEvent(sigmas[2]),
            dp_event.GaussianDpEvent(sigmas[3]),
        ]),
    ])

    sigma = rdp_privacy_accountant._effective_gaussian_noise_multiplier(event)
    multi_sigmas = list(sigmas) + [sigmas[1]] * 2
    expected = sum(s**-2 for s in multi_sigmas) ** -0.5
    self.assertAlmostEqual(sigma, expected)

  _LAPLACE_EVENT = dp_event.LaplaceDpEvent(1.0)

  @parameterized.named_parameters(
      ('simple', _LAPLACE_EVENT),
      ('composed', dp_event.ComposedDpEvent([_LAPLACE_EVENT])),
      ('self_composed', dp_event.SelfComposedDpEvent(_LAPLACE_EVENT, 3)),
  )
  def test_effective_gaussian_noise_multiplier_invalid_event(self, event):
    result = rdp_privacy_accountant._effective_gaussian_noise_multiplier(event)
    self.assertEqual(result, self._LAPLACE_EVENT)

  def test_compute_rdp_poisson_sampled_gaussian(self):
    orders = [1.5, 2.5, 5, 50, 100, np.inf]
    noise_multiplier = 2.5
    sampling_probability = 0.01
    count = 50
    event = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(
            sampling_probability, dp_event.GaussianDpEvent(noise_multiplier)
        ),
        count,
    )
    accountant = rdp_privacy_accountant.RdpAccountant(orders=orders)
    accountant.compose(event)
    self.assertTrue(
        np.allclose(
            accountant._rdp,
            [
                6.70579741e-04,
                1.08548710e-03,
                2.18075656e-03,
                2.38460375e-02,
                1.67416308e02,
                np.inf,
            ],
            rtol=1e-4,
        )
    )

  @parameterized.named_parameters(
      (
          'small_eps',
          0.3,
          10,
          [
              1.5030552584146897,
              1.7801182478869704,
              1.9091969256833639,
              1.9814516798923105,
              1.9846954517418232,
              1.9878743481543455,
          ],
      ),
      (
          'medium_eps',
          0.05,
          20,
          [
              2.8985312222056674,
              2.9615414620514047,
              2.982908949957726,
              2.994685471627712,
              2.9952141594692665,
              2.995732273553991,
          ],
      ),
      (
          'large_eps',
          0.01,
          3,
          [
              3.851345343526459,
              4.334924907685137,
              4.503803908976894,
              4.596895387863428,
              4.601074578835478,
              4.605170185988092,
          ],
      ),
  )
  def test_compute_rdp_randomized_response_replace_special(
      self, noise_parameter, num_buckets, expected_rdps
  ):
    orders = [1.5, 2.5, 5, 50, 100, np.inf]
    event = dp_event.RandomizedResponseDpEvent(noise_parameter, num_buckets)
    accountant = rdp_privacy_accountant.RdpAccountant(
        orders,
        privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
    )
    accountant.compose(event)
    self.assertTrue(
        np.allclose(
            accountant._rdp,
            expected_rdps,
            rtol=1e-4,
        )
    )

  @parameterized.named_parameters(
      (
          'small_eps',
          0.3,
          10,
          [
              2.6946320226546314,
              2.983865364028855,
              3.11316970070507,
              3.1854244842182466,
              3.1886682560677593,
              3.1918471524802814,
          ],
      ),
      (
          'medium_eps',
          0.05,
          20,
          [
              5.850317703374384,
              5.910360162307462,
              5.930633082131939,
              5.94180620835157,
              5.942307807732949,
              5.942799375126701,
          ],
      ),
      (
          'large_eps',
          0.01,
          3,
          [
              5.684126770932727,
              5.692634596034571,
              5.695421239467805,
              5.696956976543137,
              5.697025920968522,
              5.697093486505405,
          ],
      ),
  )
  def test_compute_rdp_randomized_response_replace_one(
      self, noise_parameter, num_buckets, expected_rdps
  ):
    orders = [1.5, 2.5, 5, 50, 100, np.inf]
    event = dp_event.RandomizedResponseDpEvent(noise_parameter, num_buckets)
    accountant = rdp_privacy_accountant.RdpAccountant(
        orders,
        privacy_accountant.NeighboringRelation.REPLACE_ONE,
    )
    accountant.compose(event)
    self.assertTrue(
        np.allclose(
            accountant._rdp,
            expected_rdps,
            rtol=1e-4,
        )
    )

  @parameterized.named_parameters(
      ('noise_param_1', 1, 2, [1.1, 1.2], [0, 0]),
      ('noise_param_0', 0.0, 2, [1.1, 1.2], [np.inf, np.inf]),
      ('num_buckets_1', 0.1, 1, [1.1, 1.2], [0, 0]),
  )
  def test_compute_rdp_randomized_response_boundary(
      self, noise_param, num_buckets, alphas, expected_rdps
  ):
    for rel in [
        privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
        privacy_accountant.NeighboringRelation.REPLACE_ONE,
    ]:
      self.assertSequenceAlmostEqual(
          rdp_privacy_accountant._compute_randomized_response_rdp(
              noise_param, num_buckets, alphas, rel
          ),
          expected_rdps,
      )

  @parameterized.named_parameters(
      (
          'replace_special',
          privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          2.302585092994046,
      ),
      (
          'replace_one',
          privacy_accountant.NeighboringRelation.REPLACE_ONE,
          4.51085950651685,
      ),
  )
  def test_compute_rdp_randomized_response_inf_order(self, rel, expected_rdp):
    noise_param = 0.1
    num_buckets = 10
    orders = [np.inf]
    self.assertSequenceAlmostEqual(
        rdp_privacy_accountant._compute_randomized_response_rdp(
            noise_param, num_buckets, orders, rel
        ),
        [expected_rdp],
    )

  def test_compute_rdp_randomized_response_raises(self):
    with self.assertRaisesRegex(ValueError, 'noise_parameter must be in'):
      for rel in [
          privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          privacy_accountant.NeighboringRelation.REPLACE_ONE,
      ]:
        rdp_privacy_accountant._compute_randomized_response_rdp(
            -1, 10, [1.1, 1.2], rel
        )
      with self.assertRaisesRegex(ValueError, 'noise_parameter must be in'):
        rdp_privacy_accountant._compute_randomized_response_rdp(
            2, 10, [1.1, 1.2], rel
        )
      with self.assertRaisesRegex(ValueError, 'num_buckets must be >= 1'):
        rdp_privacy_accountant._compute_randomized_response_rdp(
            0.1, 0, [1.1, 1.2], rel
        )
      with self.assertRaisesRegex(
          ValueError, 'Renyi divergence order alpha must be > 1'
      ):
        rdp_privacy_accountant._compute_randomized_response_rdp(
            0.1, 10, [1, 1.2], rel
        )

  def test_compute_epsilon_delta_pure_dp(self):
    orders = range(2, 33)
    rdp = [1.1 for _ in orders]  # Constant corresponds to pure DP.

    epsilon, optimal_order = rdp_privacy_accountant.compute_epsilon(
        orders, rdp, delta=1e-5
    )
    # Compare with epsilon computed by hand.
    self.assertAlmostEqual(epsilon, 1.32783806176)
    self.assertEqual(optimal_order, 32)

    delta, optimal_order = rdp_privacy_accountant.compute_delta(
        orders, rdp, epsilon=1.32783806176
    )
    self.assertAlmostEqual(delta, 1e-5)
    self.assertEqual(optimal_order, 32)

  def test_compute_epsilon_delta_gaussian(self):
    orders = [0.001 * i for i in range(1000, 100000)]

    # noise multiplier is chosen to obtain exactly (1,1e-6)-DP.
    rdp = rdp_privacy_accountant._compute_rdp_poisson_subsampled_gaussian(
        1, 4.530877117, orders
    )

    eps = rdp_privacy_accountant.compute_epsilon(orders, rdp, delta=1e-6)[0]
    self.assertAlmostEqual(eps, 1)

    delta = rdp_privacy_accountant.compute_delta(orders, rdp, epsilon=1)[0]
    self.assertAlmostEqual(delta, 1e-6)

  params = (
      {'q': 1e-7, 'sigma': 0.1, 'order': 1.01},
      {'q': 1e-6, 'sigma': 0.1, 'order': 256},
      {'q': 1e-5, 'sigma': 0.1, 'order': 256.1},
      {'q': 1e-6, 'sigma': 1, 'order': 27},
      {'q': 1e-4, 'sigma': 1.0, 'order': 1.5},
      {'q': 1e-3, 'sigma': 1.0, 'order': 2},
      {'q': 0.01, 'sigma': 10, 'order': 20},
      {'q': 0.1, 'sigma': 100, 'order': 20.5},
      {'q': 0.99, 'sigma': 0.1, 'order': 256},
      {'q': 0.999, 'sigma': 100, 'order': 256.1},
  )

  # pylint:disable=undefined-variable
  @parameterized.parameters(p for p in params)
  def test_compute_log_a_equals_mp(self, q, sigma, order):
    # Compare the cheap computation of log(A) with an expensive, multi-precision
    # computation.
    log_a = rdp_privacy_accountant._compute_log_a(q, sigma, order)
    log_a_mp = _log_float_mp(_compute_a_mp(sigma, q, order))
    np.testing.assert_allclose(log_a, log_a_mp, rtol=1e-4, atol=1e-8)

  def test_delta_bounds_gaussian(self):
    # Compare the optimal bound for Gaussian with the one derived from RDP.
    # Also compare the RDP upper bound with the "standard" upper bound.
    orders = [0.1 * x for x in range(10, 505)]
    eps_vec = [0.1 * x for x in range(500)]
    rdp = rdp_privacy_accountant._compute_rdp_poisson_subsampled_gaussian(
        1, 1, orders
    )
    for eps in eps_vec:
      delta = rdp_privacy_accountant.compute_delta(orders, rdp, epsilon=eps)[0]
      # For comparison, we compute the optimal guarantee for Gaussian
      # using https://arxiv.org/abs/1805.06530 Theorem 8 (in v2).
      delta0 = math.erfc((eps - 0.5) / math.sqrt(2)) / 2
      delta0 = (
          delta0 - math.exp(eps) * math.erfc((eps + 0.5) / math.sqrt(2)) / 2
      )
      self.assertLessEqual(delta0, delta + 1e-300)  # need tolerance 10^-300

      # Compute the "standard" upper bound, which should be an upper bound.
      # Note, if orders is too sparse, this will NOT be an upper bound.
      if eps >= 0.5:
        delta1 = math.exp(-0.5 * (eps - 0.5) ** 2)
      else:
        delta1 = 1
      self.assertLessEqual(delta, delta1 + 1e-300)

  def test_epsilon_delta_consistency(self):
    orders = range(2, 50)  # Large range of orders (helps test for overflows).
    for q in [0, 0.01, 0.1, 0.8, 1.0]:
      for multiplier in [0.0, 0.1, 1.0, 10.0, 100.0]:
        event = dp_event.PoissonSampledDpEvent(
            q, dp_event.GaussianDpEvent(multiplier)
        )
        accountant = rdp_privacy_accountant.RdpAccountant(orders)
        accountant.compose(event)
        for delta in [0.99, 0.9, 0.1, 0.01, 1e-3, 1e-5, 1e-9, 1e-12]:
          epsilon = accountant.get_epsilon(delta)
          delta2 = accountant.get_delta(epsilon)
          if np.isposinf(epsilon):
            self.assertEqual(delta2, 1.0)
          elif epsilon == 0:
            self.assertLessEqual(delta2, delta)
          else:
            self.assertAlmostEqual(delta, delta2)

  @parameterized.named_parameters(
      ('add_remove', privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE),
      ('replace', privacy_accountant.NeighboringRelation.REPLACE_ONE),
  )
  def test_tree_wrong_neighbor_rel(self, neighboring_relation):
    event = dp_event.SingleEpochTreeAggregationDpEvent(1.0, 1)
    accountant = rdp_privacy_accountant.RdpAccountant(
        neighboring_relation=neighboring_relation
    )
    self.assertFalse(accountant.supports(event))

  @parameterized.named_parameters(('eps20', 1.13, 19.74), ('eps2', 8.83, 2.04))
  def test_compute_eps_tree(self, noise_multiplier, eps):
    orders = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64))
    # This test is based on the StackOverflow setting in "Practical and
    # Private (Deep) Learning without Sampling or Shuffling". The calculated
    # epsilon could be better as the method in this package keeps improving.
    step_counts, target_delta = 1600, 1e-6
    new_eps = _compose_trees_single_epoch(
        noise_multiplier, step_counts, orders
    ).get_epsilon(target_delta)
    self.assertLess(new_eps, eps)

  @parameterized.named_parameters(
      ('restart4', [400] * 4),
      ('restart2', [800] * 2),
      ('adaptive', [10, 400, 400, 400, 390]),
  )
  def test_compose_tree_rdp(self, step_counts):
    noise_multiplier, orders = 0.1, [1]

    def get_rdp(step_count):
      return _compose_trees_single_epoch(
          noise_multiplier, [step_count], orders
      )._rdp[0]

    rdp_summed = sum(get_rdp(step_count) for step_count in step_counts)
    rdp_composed = _compose_trees(noise_multiplier, step_counts, orders)._rdp[0]
    self.assertTrue(np.allclose(rdp_composed, rdp_summed, rtol=1e-12))

  def test_single_epoch_multi_tree_rdp(self):
    noise_multiplier, orders = 0.1, [1]
    step_counts = [10, 40, 30, 20]
    single_rdp = _compose_trees_single_epoch(
        noise_multiplier, step_counts, orders
    )._rdp[0]

    max_rdp = max(
        _compose_trees_single_epoch(noise_multiplier, step_count, orders)._rdp[
            0
        ]
        for step_count in step_counts
    )

    self.assertEqual(single_rdp, max_rdp)

  @parameterized.named_parameters(
      ('restart4', [400] * 4),
      ('restart2', [800] * 2),
      ('adaptive', [10, 400, 400, 400, 390]),
  )
  def test_compute_eps_tree_decreasing(self, step_counts):
    # Test privacy epsilon decreases with noise multiplier increasing when
    # keeping other parameters the same.
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    target_delta = 1e-6
    prev_eps = np.inf
    for noise_multiplier in [0.1 * x for x in range(1, 100, 5)]:
      accountant = _compose_trees(noise_multiplier, step_counts, orders)
      eps = accountant.get_epsilon(target_delta=target_delta)
      self.assertLess(eps, prev_eps)
      prev_eps = eps

  @parameterized.named_parameters(
      ('negative_noise', -1, [3]),
      ('negative_steps', 1, [-3]),
  )
  def test_compute_rdp_tree_restart_raise(self, noise_multiplier, step_counts):
    with self.assertRaisesRegex(ValueError, 'non-negative'):
      _compose_trees(noise_multiplier, step_counts, orders=[1])

  @parameterized.named_parameters(
      ('t100n0.1', 100, 0.1),
      ('t1000n0.01', 1000, 0.01),
  )
  def test_no_tree_no_sampling(self, total_steps, noise_multiplier):
    orders = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64))
    tree_rdp = _compose_trees(noise_multiplier, [1] * total_steps, orders)._rdp
    accountant = rdp_privacy_accountant.RdpAccountant(orders)
    event = dp_event.SelfComposedDpEvent(
        dp_event.GaussianDpEvent(noise_multiplier), total_steps
    )
    accountant.compose(event)
    base_rdp = accountant._rdp
    self.assertTrue(np.allclose(tree_rdp, base_rdp, rtol=1e-12))

  @parameterized.named_parameters(
      ('small_eps', 0.01, 1),
      ('medium_eps', 1.0, 1),
      ('large_eps', 100.0, 1),
      ('repetition', 1.0, 100),
  )
  def test_laplace(self, eps, count):
    event = dp_event.LaplaceDpEvent(1 / eps)
    if count != 1:
      event = dp_event.SelfComposedDpEvent(event, count)
    # Simulate Pure DP by using a large Renyi order.
    accountant = rdp_privacy_accountant.RdpAccountant(
        orders=[1.0, 1 + 1e-6, 1.1 - 1e-6, 1.1 + 1e-6, 1e10]
    )
    accountant.compose(event)
    # Check basic composition by having small delta.
    self.assertAlmostEqual(accountant.get_epsilon(1e-10), eps * count)
    # Check KL divergence, a.k.a. expected privacy loss, a.k.a. order=1.
    self.assertLessEqual(accountant._rdp[0], min(eps, eps * eps / 2) * count)
    # Check consistency between the three formulas.
    self.assertAlmostEqual(accountant._rdp[0], accountant._rdp[1], places=3)
    self.assertAlmostEqual(accountant._rdp[2], accountant._rdp[3], places=3)

  # The function _truncated_negative_binomial_mean computes the mean in
  # multiple ways to ensure numerical stability.
  # This test checks that those different ways of computing are consistent.
  @parameterized.named_parameters(
      ('gamma_shape0', 0.9, 0, 0.9 - 1e-9, 0),
      ('gamma_shape2', 0.9, 2, 0.9 - 1e-9, 2),
      ('gamma_shape_0.5', 0.9, 0.5, 0.9 - 1e-9, 0.5),
      (
          'x_shape2',
          math.exp(-0.05),
          2,
          math.exp(-0.05) - 1e-9,
          2,
      ),  # x = shape * math.log(gamma) = -0.1
      (
          'x_shape0.5',
          math.exp(-0.2),
          0.5,
          math.exp(-0.2) - 1e-9,
          0.5,
      ),  # x = shape * math.log(gamma) = -0.1
      ('shape_0', 0.6, 0, 0.6, 1e-9),
      ('shape_1', 0.6, 1, 0.6, 1 + 1e-9),
  )
  def test_truncated_negative_binomial_mean(
      self, gamma1, shape1, gamma2, shape2
  ):
    mean1 = rdp_privacy_accountant._truncated_negative_binomial_mean(
        gamma1, shape1
    )
    mean2 = rdp_privacy_accountant._truncated_negative_binomial_mean(
        gamma2, shape2
    )
    self.assertAlmostEqual(mean1, mean2)

  @parameterized.named_parameters(
      ('1e-7', 1e-7), ('.1', 0.1), ('0.999999', 1 - 1e-6), ('1', 1)
  )
  def test_truncated_negative_binomial_mean2(self, gamma):
    # Test this function by simply applying the non-numerically stable formula.
    # logarithmic distribution
    mean = rdp_privacy_accountant._truncated_negative_binomial_mean(gamma, 0)
    if gamma == 1:
      ans = 1
    else:
      ans = (1 - 1 / gamma) / math.log(gamma)
    self.assertAlmostEqual(mean, ans)

    # geometric Distribution
    mean = rdp_privacy_accountant._truncated_negative_binomial_mean(gamma, 1)
    self.assertAlmostEqual(mean, 1 / gamma)

    # general TNB Distribution
    for shape in [0.01, 0.5, 0.99, 1.01, 2, 10]:
      mean = rdp_privacy_accountant._truncated_negative_binomial_mean(
          gamma, shape
      )
      if gamma == 1:
        ans = 1
      else:
        ans = shape * (1 / gamma - 1) / (1 - gamma**shape)
      self.assertAlmostEqual(mean, ans)

  # _gamma_truncated_negative_binomial is meant to be the inverse of
  # _truncated_negative_binomial_mean, so we test this.
  @parameterized.named_parameters(
      ('shape0a', 0.1, 0),
      ('shape0.5a', 0.1, 0.5),
      ('shape1a', 0.1, 1),
      ('shape2a', 0.1, 2),
      ('shape0b', 0.0001, 0),
      ('shape0.5b', 0.0001, 0.5),
      ('shape1b', 0.0001, 1),
      ('shape2b', 0.0001, 2),
      ('shape0c', 1, 0),
      ('shape0.5c', 1, 0.5),
      ('shape1c', 1, 1),
      ('shape2c', 1, 2),
      ('shape0', 0.999, 0),
      ('shape0.5', 0.999, 0.5),
      ('shape1', 0.999, 1),
      ('shape2', 0.999, 2),
  )
  def test_gamma_truncated_negative_binomial(self, gamma, shape):
    mean = rdp_privacy_accountant._truncated_negative_binomial_mean(
        gamma, shape
    )
    g = rdp_privacy_accountant._gamma_truncated_negative_binomial(shape, mean)
    self.assertAlmostEqual(g, gamma)

  @parameterized.named_parameters(
      ('logarithmic', 1, 1000, 0),
      ('geometric', 1, 1000, 1),
      ('negative binomial 0.5', 1, 1000, 0.5),
      ('negative binomial 2', 1, 100, 2),
      ('negative binomial 5', 1, 1000, 5),
  )
  def test_repeat_select_pure_negative_binomial(self, eps, mean, shape):
    # Test the Repeat and Select DP event in the almost-pure DP case.
    event = dp_event.LaplaceDpEvent(1 / eps)
    event = dp_event.RepeatAndSelectDpEvent(event, mean, shape)
    # Use single large order to simulate pure DP.
    accountant = rdp_privacy_accountant.RdpAccountant(orders=[1e10])
    accountant.compose(event)
    # Correct answer is given by Corollary 3 https://arxiv.org/abs/2110.03620
    self.assertAlmostEqual(accountant._rdp[0], eps * (2 + shape))
    self.assertAlmostEqual(accountant.get_epsilon(1e-10), eps * (2 + shape))

  @parameterized.named_parameters(
      ('shape0', 0, 1),
      ('shape0.5', 0.5, 10),
      ('shape1', 1, 0.1),
      ('shape2', 2, 1),
  )
  def test_repeat_select_trivial(self, shape, sigma):
    # Test the repeat and select function in the trivial mean=1 case.
    orders = [
        1,
        1 + 1e-6,  # We include 1, as otherwise this test fails.
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        20,
        24,
        28,
        32,
        48,
        64,
        128,
        256,
        512,
        1024,
    ]
    event1 = dp_event.GaussianDpEvent(sigma)
    accountant1 = rdp_privacy_accountant.RdpAccountant(orders=orders)
    accountant1.compose(event1)
    event2 = dp_event.RepeatAndSelectDpEvent(event1, 1, shape)
    accountant2 = rdp_privacy_accountant.RdpAccountant(orders=orders)
    accountant2.compose(event2)
    for i in range(len(accountant1._orders)):
      if orders[i] > 1:  # Otherwise our formula doesn't work.
        self.assertAlmostEqual(accountant1._rdp[i], accountant2._rdp[i])

  @parameterized.named_parameters(
      ('small0', 0.01, 0.01, 0),
      ('med0', 1, 0.1, 0),
      ('large0', 10, 0.99, 0),
      ('small0.5', 0.01, 0.01, 0.5),
      ('med0.5', 1, 0.1, 0.5),
      ('large0.5', 10, 0.99, 0.5),
      ('small1', 0.01, 0.01, 1),
      ('med1', 1, 0.1, 1),
      ('large1', 10, 0.99, 1),
      ('small5', 0.01, 0.01, 5),
      ('med5', 1, 0.1, 5),
      ('large5', 10, 0.99, 5),
  )
  def test_repeat_select_gaussian_negative_binomial(self, rho, gamma, shape):
    # Test the Repeat and Select DP event in the Gaussian case.
    # Correct answer is given by Corollary 4 https://arxiv.org/abs/2110.03620
    mean = rdp_privacy_accountant._truncated_negative_binomial_mean(
        gamma, shape
    )
    rho = min(rho, -math.log(gamma))  # We need rho<=log(1/gamma).
    self.assertGreater(rho, 0)  # Otherwise we get division by zero.
    orders = [
        1,
        1.1,
        2,
        math.sqrt(-math.log(gamma) / rho),
        1 + math.sqrt(math.log(mean) / rho),
        3,
        5,
        10,
        100,
        1000,
        10000,
    ]
    event = dp_event.GaussianDpEvent(math.sqrt(0.5 / rho))
    event = dp_event.RepeatAndSelectDpEvent(event, mean, shape)
    accountant = rdp_privacy_accountant.RdpAccountant(orders=orders)
    accountant.compose(event)
    for i in range(len(orders)):
      order = accountant._orders[i]
      rdp = accountant._rdp[i]
      if order <= 1 + math.sqrt(math.log(mean) / rho):
        eps = (
            2 * math.sqrt(rho * math.log(mean))
            + 2 * (1 + shape) * math.sqrt(-rho * math.log(gamma))
            - shape * rho
        )
      else:
        eps = (
            rho * (order - 1)
            + math.log(mean) / (order - 1)
            + 2 * (1 + shape) * math.sqrt(-rho * math.log(gamma))
            - shape * rho
        )
      self.assertAlmostEqual(rdp, eps, msg='order=' + str(order))

  @parameterized.named_parameters(
      ('mean1', 1, 1),
      ('mean2', 0.1, 2),
      ('mean10', 10, 10),
      ('mean100', 0.001, 100),
      ('mean10^4', 2, 1000),
      ('mean10^10', 1, 1e10),
  )
  def test_repeat_and_select_pure_poisson(self, eps, mean):
    event = dp_event.LaplaceDpEvent(1 / eps)
    event = dp_event.RepeatAndSelectDpEvent(event, mean, np.inf)
    alpha = 1 + 1 / math.expm1(eps)
    orders = [alpha, 1e10, 1e100]
    accountant = rdp_privacy_accountant.RdpAccountant(orders=orders)
    accountant.compose(event)
    ans = (
        eps
        + math.log1p(
            math.expm1((1 - 2 * alpha) * eps) * (alpha - 1) / (2 * alpha - 1)
        )
        / (alpha - 1)
        + math.log(mean) * math.expm1(eps)
    )
    self.assertAlmostEqual(accountant._orders[0], alpha)
    self.assertAlmostEqual(accountant._rdp[0] / ans, 1, places=3)

  @parameterized.named_parameters(('q_neg', -0.1), ('q_large', 1.1))
  def test_raises_on_invalid_sampling_rate(self, q):
    with self.assertRaisesRegex(ValueError, 'Sampling rate'):
      rdp_privacy_accountant.RdpAccountant().compose(
          dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(1))
      )

  def test_raises_on_invalid_noise_multiplier(self):
    with self.assertRaisesRegex(ValueError, 'Noise multiplier'):
      rdp_privacy_accountant.RdpAccountant().compose(
          dp_event.GaussianDpEvent(-1)
      )

  @parameterized.named_parameters(
      ('small_small', 0.001, 1),
      ('small_med', 0.001, 1000),
      ('small_large', 0.001, 1e9),
      ('med_small', 1, 1),
      ('med_med', 1, 1000),
      ('med_large', 1, 1e9),
      ('large_small', 1000, 1),
      ('large_med', 1000, 1000),
      ('large_large', 1000, 1e9),
  )
  def test_repeat_and_select_gaussian_poisson(self, sigma, mean):
    event = dp_event.GaussianDpEvent(sigma)
    event = dp_event.RepeatAndSelectDpEvent(event, mean, np.inf)
    accountant = rdp_privacy_accountant.RdpAccountant()
    accountant.compose(event)
    orders = accountant._orders
    rdp = []
    for order in orders:
      if order <= 1:  # Avoid division by zero.
        rdp.append(np.inf)
        continue
      eps = math.log1p(1 / (order - 1))
      x = (eps * sigma - 0.5 / sigma) / math.sqrt(2)
      y = (eps * sigma + 0.5 / sigma) / math.sqrt(2)
      delta = math.erfc(x) / 2 - math.exp(eps) * math.erfc(y) / 2
      rdp.append(
          order * 0.5 / (sigma**2) + mean * delta + math.log(mean) / (order - 1)
      )
    for order, accountant_rdp in zip(orders, accountant._rdp):
      lb = min(rdp[j] for j in range(len(orders)) if orders[j] >= order)
      self.assertLessEqual(lb, accountant_rdp)

  def test_log_a_frac_positive(self):
    # Testing a combination of q, sigma and alpha that formerly returned a
    # negative log_a_frac.
    for order in np.linspace(58.5, 59.5, 21):
      log_a = rdp_privacy_accountant._compute_log_a_frac(0.4, 12, order)
      self.assertGreater(log_a, 0)

  def test_log_a_frac_early_termination(self):
    # Test an event that is known to not converge for small orders.
    event = dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0))
    accountant = rdp_privacy_accountant.RdpAccountant()
    with self.assertLogs(level='WARNING') as log:
      accountant.compose(event)
    self.assertNotEmpty([l for l in log.output if 'failed to converge' in l])
    self.assertIn(np.inf, accountant._rdp)


if __name__ == '__main__':
  absltest.main()
