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
    return mu0(z) * _mu_over_mu0(z, q, sigma)**alpha

  bounds = (-mpmath.inf, mpmath.inf)
  a_alpha, _ = mpmath.quad(a_alpha_fn, bounds, error=True, maxdegree=8)
  return a_alpha


def _compose_trees(noise_multiplier, step_counts, orders):
  accountant = rdp_privacy_accountant.RdpAccountant(
      orders, privacy_accountant.NeighboringRelation.REPLACE_SPECIAL)
  accountant.compose(
      dp_event.ComposedDpEvent([
          dp_event.SingleEpochTreeAggregationDpEvent(noise_multiplier,
                                                     step_count)
          for step_count in step_counts
      ]))
  return accountant


def _compose_trees_single_epoch(noise_multiplier, step_counts, orders):
  accountant = rdp_privacy_accountant.RdpAccountant(
      orders, privacy_accountant.NeighboringRelation.REPLACE_SPECIAL)
  accountant.compose(
      dp_event.SingleEpochTreeAggregationDpEvent(noise_multiplier, step_counts))
  return accountant


class RdpPrivacyAccountantTest(privacy_accountant_test.PrivacyAccountantTest,
                               parameterized.TestCase):

  def _make_test_accountants(self):
    return [
        rdp_privacy_accountant.RdpAccountant(
            [2.0], privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE),
        rdp_privacy_accountant.RdpAccountant(
            [2.0], privacy_accountant.NeighboringRelation.REPLACE_ONE),
        rdp_privacy_accountant.RdpAccountant(
            [2.0], privacy_accountant.NeighboringRelation.REPLACE_SPECIAL)
    ]

  def test_supports(self):
    aor_accountant = rdp_privacy_accountant.RdpAccountant(
        [2.0], privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE)
    ro_accountant = rdp_privacy_accountant.RdpAccountant(
        [2.0], privacy_accountant.NeighboringRelation.REPLACE_ONE)

    event = dp_event.GaussianDpEvent(1.0)
    self.assertTrue(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(1.0), 6)
    self.assertTrue(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.ComposedDpEvent(
        [dp_event.GaussianDpEvent(1.0),
         dp_event.GaussianDpEvent(2.0)])
    self.assertTrue(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0))
    self.assertTrue(aor_accountant.supports(event))
    self.assertFalse(ro_accountant.supports(event))

    composed_gaussian = dp_event.ComposedDpEvent(
        [dp_event.GaussianDpEvent(1.0),
         dp_event.GaussianDpEvent(2.0)])
    event = dp_event.PoissonSampledDpEvent(0.1, composed_gaussian)
    self.assertTrue(aor_accountant.supports(event))
    self.assertFalse(ro_accountant.supports(event))

    event = dp_event.SampledWithoutReplacementDpEvent(
        1000, 10, dp_event.GaussianDpEvent(1.0))
    self.assertFalse(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.SampledWithoutReplacementDpEvent(1000, 10,
                                                      composed_gaussian)
    self.assertFalse(aor_accountant.supports(event))
    self.assertTrue(ro_accountant.supports(event))

    event = dp_event.SampledWithReplacementDpEvent(
        1000, 10, dp_event.GaussianDpEvent(1.0))
    self.assertFalse(aor_accountant.supports(event))
    self.assertFalse(ro_accountant.supports(event))

  def test_rdp_composition(self):
    base_event = dp_event.GaussianDpEvent(3.14159)
    base_rdp = _get_test_rdp(base_event)

    rdp_with_count = _get_test_rdp(base_event, count=6)
    self.assertAlmostEqual(rdp_with_count, base_rdp * 6)

    rdp_with_self_compose = _get_test_rdp(
        dp_event.SelfComposedDpEvent(base_event, 6))
    self.assertAlmostEqual(rdp_with_self_compose, base_rdp * 6)

    rdp_with_self_compose_and_count = _get_test_rdp(
        dp_event.SelfComposedDpEvent(base_event, 2), count=3)
    self.assertAlmostEqual(rdp_with_self_compose_and_count, base_rdp * 6)

    rdp_with_compose = _get_test_rdp(dp_event.ComposedDpEvent([base_event] * 6))
    self.assertAlmostEqual(rdp_with_compose, base_rdp * 6)

    rdp_with_compose_and_self_compose = _get_test_rdp(
        dp_event.ComposedDpEvent([
            dp_event.SelfComposedDpEvent(base_event, 1),
            dp_event.SelfComposedDpEvent(base_event, 2),
            dp_event.SelfComposedDpEvent(base_event, 3)
        ]))
    self.assertAlmostEqual(rdp_with_compose_and_self_compose, base_rdp * 6)

    base_event_2 = dp_event.GaussianDpEvent(1.61803)
    base_rdp_2 = _get_test_rdp(base_event_2)
    rdp_with_heterogeneous_compose = _get_test_rdp(
        dp_event.ComposedDpEvent([base_event, base_event_2]))
    self.assertAlmostEqual(rdp_with_heterogeneous_compose,
                           base_rdp + base_rdp_2)

  def test_zero_poisson_sample(self):
    accountant = rdp_privacy_accountant.RdpAccountant([3.14159])
    accountant.compose(
        dp_event.PoissonSampledDpEvent(0, dp_event.GaussianDpEvent(1.0)))
    self.assertEqual(accountant.get_epsilon(1e-10), 0)
    self.assertEqual(accountant.get_delta(1e-10), 0)

  def test_zero_fixed_batch_sample(self):
    accountant = rdp_privacy_accountant.RdpAccountant(
        [3.14159], privacy_accountant.NeighboringRelation.REPLACE_ONE)
    accountant.compose(
        dp_event.SampledWithoutReplacementDpEvent(
            1000, 0, dp_event.GaussianDpEvent(1.0)))
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
                dp_event.GaussianDpEvent(sigma2)
            ])))
    self.assertAlmostEqual(accountant._rdp[0], rdp)

  def test_effective_gaussian_noise_multiplier(self):
    np.random.seed(0xBAD5EED)
    sigmas = np.random.uniform(size=(4,))

    event = dp_event.ComposedDpEvent([
        dp_event.GaussianDpEvent(sigmas[0]),
        dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(sigmas[1]), 3),
        dp_event.ComposedDpEvent([
            dp_event.GaussianDpEvent(sigmas[2]),
            dp_event.GaussianDpEvent(sigmas[3])
        ])
    ])

    sigma = rdp_privacy_accountant._effective_gaussian_noise_multiplier(event)
    multi_sigmas = list(sigmas) + [sigmas[1]] * 2
    expected = sum(s**-2 for s in multi_sigmas)**-0.5
    self.assertAlmostEqual(sigma, expected)

  def test_compute_rdp_poisson_sampled_gaussian(self):
    orders = [1.5, 2.5, 5, 50, 100, np.inf]
    noise_multiplier = 2.5
    sampling_probability = 0.01
    count = 50
    event = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(
            sampling_probability, dp_event.GaussianDpEvent(noise_multiplier)),
        count)
    accountant = rdp_privacy_accountant.RdpAccountant(orders=orders)
    accountant.compose(event)
    self.assertTrue(
        np.allclose(
            accountant._rdp, [
                6.5007e-04, 1.0854e-03, 2.1808e-03, 2.3846e-02, 1.6742e+02,
                np.inf
            ],
            rtol=1e-4))

  def test_compute_epsilon_delta_pure_dp(self):
    orders = range(2, 33)
    rdp = [1.1 for o in orders]  # Constant corresponds to pure DP.

    epsilon, optimal_order = rdp_privacy_accountant.compute_epsilon(
        orders, rdp, delta=1e-5)
    # Compare with epsilon computed by hand.
    self.assertAlmostEqual(epsilon, 1.32783806176)
    self.assertEqual(optimal_order, 32)

    delta, optimal_order = rdp_privacy_accountant.compute_delta(
        orders, rdp, epsilon=1.32783806176)
    self.assertAlmostEqual(delta, 1e-5)
    self.assertEqual(optimal_order, 32)

  def test_compute_epsilon_delta_gaussian(self):
    orders = [0.001 * i for i in range(1000, 100000)]

    # noise multiplier is chosen to obtain exactly (1,1e-6)-DP.
    rdp = rdp_privacy_accountant._compute_rdp_poisson_subsampled_gaussian(
        1, 4.530877117, orders)

    eps = rdp_privacy_accountant.compute_epsilon(orders, rdp, delta=1e-6)[0]
    self.assertAlmostEqual(eps, 1)

    delta = rdp_privacy_accountant.compute_delta(orders, rdp, epsilon=1)[0]
    self.assertAlmostEqual(delta, 1e-6)

  params = ({
      'q': 1e-7,
      'sigma': .1,
      'order': 1.01
  }, {
      'q': 1e-6,
      'sigma': .1,
      'order': 256
  }, {
      'q': 1e-5,
      'sigma': .1,
      'order': 256.1
  }, {
      'q': 1e-6,
      'sigma': 1,
      'order': 27
  }, {
      'q': 1e-4,
      'sigma': 1.,
      'order': 1.5
  }, {
      'q': 1e-3,
      'sigma': 1.,
      'order': 2
  }, {
      'q': .01,
      'sigma': 10,
      'order': 20
  }, {
      'q': .1,
      'sigma': 100,
      'order': 20.5
  }, {
      'q': .99,
      'sigma': .1,
      'order': 256
  }, {
      'q': .999,
      'sigma': 100,
      'order': 256.1
  })

  # pylint:disable=undefined-variable
  @parameterized.parameters(p for p in params)
  def test_compute_log_a_equals_mp(self, q, sigma, order):
    # Compare the cheap computation of log(A) with an expensive, multi-precision
    # computation.
    log_a = rdp_privacy_accountant._compute_log_a(q, sigma, order)
    log_a_mp = _log_float_mp(_compute_a_mp(sigma, q, order))
    np.testing.assert_allclose(log_a, log_a_mp, rtol=1e-4)

  def test_delta_bounds_gaussian(self):
    # Compare the optimal bound for Gaussian with the one derived from RDP.
    # Also compare the RDP upper bound with the "standard" upper bound.
    orders = [0.1 * x for x in range(10, 505)]
    eps_vec = [0.1 * x for x in range(500)]
    rdp = rdp_privacy_accountant._compute_rdp_poisson_subsampled_gaussian(
        1, 1, orders)
    for eps in eps_vec:
      delta = rdp_privacy_accountant.compute_delta(orders, rdp, epsilon=eps)[0]
      # For comparison, we compute the optimal guarantee for Gaussian
      # using https://arxiv.org/abs/1805.06530 Theorem 8 (in v2).
      delta0 = math.erfc((eps - .5) / math.sqrt(2)) / 2
      delta0 = delta0 - math.exp(eps) * math.erfc((eps + .5) / math.sqrt(2)) / 2
      self.assertLessEqual(delta0, delta + 1e-300)  # need tolerance 10^-300

      # Compute the "standard" upper bound, which should be an upper bound.
      # Note, if orders is too sparse, this will NOT be an upper bound.
      if eps >= 0.5:
        delta1 = math.exp(-0.5 * (eps - 0.5)**2)
      else:
        delta1 = 1
      self.assertLessEqual(delta, delta1 + 1e-300)

  def test_epsilon_delta_consistency(self):
    orders = range(2, 50)  # Large range of orders (helps test for overflows).
    for q in [0, 0.01, 0.1, 0.8, 1.]:
      for multiplier in [0.0, 0.1, 1., 10., 100.]:
        event = dp_event.PoissonSampledDpEvent(
            q, dp_event.GaussianDpEvent(multiplier))
        accountant = rdp_privacy_accountant.RdpAccountant(orders)
        accountant.compose(event)
        for delta in [.99, .9, .1, .01, 1e-3, 1e-5, 1e-9, 1e-12]:
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
      ('replace', privacy_accountant.NeighboringRelation.REPLACE_ONE))
  def test_tree_wrong_neighbor_rel(self, neighboring_relation):
    event = dp_event.SingleEpochTreeAggregationDpEvent(1.0, 1)
    accountant = rdp_privacy_accountant.RdpAccountant(
        neighboring_relation=neighboring_relation)
    self.assertFalse(accountant.supports(event))

  @parameterized.named_parameters(('eps20', 1.13, 19.74), ('eps2', 8.83, 2.04))
  def test_compute_eps_tree(self, noise_multiplier, eps):
    orders = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64))
    # This test is based on the StackOverflow setting in "Practical and
    # Private (Deep) Learning without Sampling or Shuffling". The calculated
    # epsilon could be better as the method in this package keeps improving.
    step_counts, target_delta = 1600, 1e-6
    new_eps = _compose_trees_single_epoch(noise_multiplier, step_counts,
                                          orders).get_epsilon(target_delta)
    self.assertLess(new_eps, eps)

  @parameterized.named_parameters(
      ('restart4', [400] * 4),
      ('restart2', [800] * 2),
      ('adaptive', [10, 400, 400, 400, 390]),
  )
  def test_compose_tree_rdp(self, step_counts):
    noise_multiplier, orders = 0.1, [1]

    def get_rdp(step_count):
      return _compose_trees_single_epoch(noise_multiplier, [step_count],
                                         orders)._rdp[0]

    rdp_summed = sum(get_rdp(step_count) for step_count in step_counts)
    rdp_composed = _compose_trees(noise_multiplier, step_counts, orders)._rdp[0]
    self.assertTrue(np.allclose(rdp_composed, rdp_summed, rtol=1e-12))

  def test_single_epoch_multi_tree_rdp(self):
    noise_multiplier, orders = 0.1, [1]
    step_counts = [10, 40, 30, 20]
    single_rdp = _compose_trees_single_epoch(noise_multiplier, step_counts,
                                             orders)._rdp[0]

    max_rdp = max(
        _compose_trees_single_epoch(noise_multiplier, step_count,
                                    orders)._rdp[0]
        for step_count in step_counts)

    self.assertEqual(single_rdp, max_rdp)

  @parameterized.named_parameters(
      ('restart4', [400] * 4),
      ('restart2', [800] * 2),
      ('adaptive', [10, 400, 400, 400, 390]),
  )
  def test_compute_eps_tree_decreasing(self, step_counts):
    # Test privacy epsilon decreases with noise multiplier increasing when
    # keeping other parameters the same.
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
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
        dp_event.GaussianDpEvent(noise_multiplier), total_steps)
    accountant.compose(event)
    base_rdp = accountant._rdp
    self.assertTrue(np.allclose(tree_rdp, base_rdp, rtol=1e-12))


if __name__ == '__main__':
  absltest.main()
