"""Focused scientific invariance and V2-only parameterization checks."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mealpy import FloatVar

from algorithm_acronym_list import optimizer_class, resolve_optimizer_name
from de_mc_cf_optimizer import DE_MC_CF
from de_mc_cf_v2_optimizer import DE_MC_CF_V2
from gpu_batching import BatchedDEEngine
from de_mc_cf_v2_plans import random_plan
import main


def sphere(x):
    return float(np.sum(x * x))


def problem():
    return dict(bounds=FloatVar(lb=(-5.,) * 5, ub=(5.,) * 5),
                minmax='min', obj_func=sphere, log_to=None)


def engine(name='DE-MC-CF-v2', device='cpu', **kwargs):
    return BatchedDEEngine(name, sphere, [-5.] * 5, [5.] * 5, 3, 10,
                           compute_device=device, beta_min=.1, beta_max=.6,
                           pcr=.1, mahalanobis_q=.5, objective_strategy='serial', **kwargs)


class ScientificTests(unittest.TestCase):
    def test_original_sources_unchanged(self):
        # Source snapshot taken before introducing the experimental implementation.
        manifest = '''0501f80af33f3bf6d5926b45d88fa545afb261ab3c487697ca8d914c6c42f5eb  de_mc_cf_optimizer.py
424907cd4c97bf84540851a45775e83583c6cc21706fd81cdce902dc4a1a6750  de_mc_optimizer.py
ae7ab5ef49eee6a5c536b56e5117ba7c6695562b9a590263e2477144a8d610f0  de_ablation_base.py
fffd33f956215812f4236945d8106e094810dd0ca290e172d8b62570a760c890  macro_de_optimizer.py
e179d451805f5a93d5e103e5930fb4b0accd908b8ce0e9462667ed3f87462b52  macro_gpu_random.py'''
        for line in manifest.splitlines():
            expected, name = line.split()
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), expected)
        self.assertIs(optimizer_class('MaCRO-DE-t'), DE_MC_CF)
        self.assertIs(optimizer_class('DE-MC-CF'), DE_MC_CF)
        old = DE_MC_CF(epoch=3, pop_size=10)
        self.assertEqual((old.wf, old.cr), (.5, .9))

    def test_aliases_cache_and_ofat(self):
        for name in ('DE-MC-CF-v2', 'DE_MC_CF_V2', 'MaCRO-DE-t-v2'):
            self.assertIs(optimizer_class(name), DE_MC_CF_V2)
            self.assertEqual(resolve_optimizer_name(name), 'DE-MC-CF-v2')
        with patch('sys.argv', ['main.py', '--experiment-mode', 'sensitivity',
                                '--sensitivity-optimizer', 'MaCRO-DE-t-v2']):
            args = main.parse_args()
        main.apply_experiment_mode(args, 'sensitivity')
        self.assertEqual((args.dims, args.pop_size, args.epochs, args.runs,
                          args.seed_base, args.compute_device), (30, 50, 2000, 30, 1234, 'gpu'))
        names = ['DE-MC-CF-v2', 'DE-MC-CF', 'MaCRO-DE', 'DSADE']
        signatures = [main.build_cache_signature(args, name, 'F12017') for name in names]
        self.assertEqual(len(set(signatures)), 4)
        self.assertEqual(signatures[0], main.build_cache_signature(args, 'MaCRO-DE-t-v2', 'F12017'))
        self.assertEqual(main.optimizer_scientific_parameters(names[0], args)['implementation_revision'],
                         'awad-close-far-beta-v2')
        configurations = list(main.optimizer_experiment_configurations(args))
        self.assertEqual(len(configurations), 16)
        nominal = dict(beta_min=.1, beta_max=.6, pcr=.1, mahalanobis_q=.5)
        for label, name, variant in configurations:
            self.assertEqual(name, names[0])
            expected = dict(nominal)
            expected[variant.sensitivity_parameter] = variant.sensitivity_value
            actual = main.optimizer_scientific_parameters(name, variant)
            for key, value in expected.items():
                self.assertEqual(actual[key], value)
        self.assertEqual(len({main.build_cache_signature(v, n, 'F12017') for _, n, v in configurations}), 13)
        functions = ['F12017', 'F82017', 'F152017', 'F242017']
        self.assertEqual(main.select_experiment_functions(args, dict.fromkeys(functions)), functions)
        self.assertTrue(main.make_paths(args, create=False).res_dir.endswith('/sensitivity/DE-MC-CF-v2'))
        args.sensitivity_optimizer = 'MaCRO-DE'
        self.assertTrue(main.make_paths(args, create=False).res_dir.endswith('/sensitivity'))

    def test_scales_and_crossover(self):
        new = DE_MC_CF_V2(epoch=3, pop_size=10)
        new.problem = SimpleNamespace(n_dims=100)
        new.generator = np.random.default_rng(42)
        expected = np.random.default_rng(42).uniform(.1, .6, 100)
        np.testing.assert_array_equal(new._sample_scale_factors(), expected)
        draws = np.array([new._sample_scale_factors() for _ in range(100)])
        self.assertTrue(np.all((draws >= .1) & (draws <= .6)))
        self.assertEqual(np.unique(draws).size, draws.size)
        new.correct_solution = lambda x: x
        for pcr in (0., .1, .4, 1.):
            new.cr = pcr
            new.generator = np.random.default_rng(43)
            rng = np.random.default_rng(43)
            j0 = rng.integers(0, 100)
            mask = rng.random(100) <= pcr
            mask[j0] = True
            actual = new._binomial_crossover(np.zeros(100), np.ones(100))
            np.testing.assert_array_equal(actual, mask.astype(float))
            self.assertEqual(actual[j0], 1.)
            if pcr == 0:
                self.assertEqual(actual.sum(), 1)

    def test_inherited_geometry_routing_and_fallback(self):
        old, new = DE_MC_CF(epoch=3, pop_size=10, mahalanobis_q=.5), DE_MC_CF_V2(epoch=3, pop_size=10)
        pop = np.random.default_rng(11).normal(size=(10, 5))
        for method in ('_awad', '_route_for_diversity', '_mutation_pool_indices',
                       '_valid_candidates', '_fallback_candidates', '_sample_mutation_indices',
                       '_covariance_matrix', '_covariance_inverse', '_mahalanobis_dist2',
                       '_mahalanobis_threshold', '_close_far_indices', '_binomial_crossover'):
            self.assertIs(getattr(DE_MC_CF, method), getattr(DE_MC_CF_V2, method))
        for model in (old, new):
            model.problem = SimpleNamespace(n_dims=5, lb=np.full(5, -5), ub=np.full(5, 5))
            model.initialize_variables()
        self.assertEqual(old._awad(pop, None, None), new._awad(pop, None, None))
        np.testing.assert_array_equal(old._covariance_matrix(pop), new._covariance_matrix(pop))
        np.testing.assert_array_equal(old._mahalanobis_dist2(pop), new._mahalanobis_dist2(pop))
        with patch('numpy.linalg.cholesky', side_effect=np.linalg.LinAlgError):
            for model in (old, new):
                np.testing.assert_array_equal(model._covariance_inverse(np.eye(5)), np.eye(5))
                factor, kind = model.backend.covariance_factor(np.eye(5), 'cholesky')
                np.testing.assert_array_equal(factor, np.eye(5))
                self.assertEqual(kind, 'inverse')
        for close in (np.array([], dtype=int), np.array([0, 1, 2]), np.array([0, 1, 2, 3]), np.arange(10)):
            far = np.setdiff1d(np.arange(10), close)
            for div in (0., .499999, .5, 1.):
                for model in (old, new):
                    model._close_far_indices = lambda p: (close, far)
                    model.div_norm_for_update = div
                    model.generator = np.random.default_rng(99)
                for target in range(10):
                    a, b = [m._sample_mutation_indices(pop, target) for m in (old, new)]
                    np.testing.assert_array_equal(a, b)
                    self.assertNotIn(target, a)
                    self.assertEqual(len(set(a)), 3)

    def test_original_generation_equivalence_when_parameter_change_disabled(self):
        old = DE_MC_CF(epoch=3, pop_size=10, mahalanobis_q=.5)
        new = DE_MC_CF_V2(epoch=3, pop_size=10, pcr=.9)
        # Disable extra RNG draws solely to prove every other scalar operation unchanged.
        new._sample_scale_factors = lambda: .5
        old.solve(problem(), seed=1234)
        new.solve(problem(), seed=1234)
        np.testing.assert_array_equal(old._positions(old.pop), new._positions(new.pop))
        np.testing.assert_array_equal(old.div_awad_hist, new.div_awad_hist)
        np.testing.assert_array_equal(old.div_norm_hist, new.div_norm_hist)
        self.assertEqual(old.routing_counts, new.routing_counts)

    def test_scalar_batched_and_batch_size_determinism(self):
        scalar = DE_MC_CF_V2(epoch=3, pop_size=10)
        scalar.solve(problem(), seed=1234)
        _, state, _ = engine().run([0, 1], [1234, 1235], capture_trace=True)
        _, single, _ = engine().run([0], [1234], capture_trace=True)
        np.testing.assert_allclose(scalar._positions(scalar.pop), state.positions[0], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(single.positions[0], state.positions[0])
        np.testing.assert_allclose(scalar.history.list_global_best_fit, state.histories[0], rtol=0, atol=1e-13)
        trace = state.trace
        x = trace['initial_population']; d = trace['donor_indices']
        rows = np.arange(2)[:, None]
        mutant = np.clip(x[rows, d[:, :, 0]] + trace['f_vectors'] *
                         (x[rows, d[:, :, 1]] - x[rows, d[:, :, 2]]), -5., 5.)
        np.testing.assert_array_equal(trace['trial_population'], np.where(trace['crossover_masks'], mutant, x))
        self.assertTrue(trace['crossover_masks'].any(axis=2).all())
        np.testing.assert_array_equal(trace['pcr_values'], [.1, .1])

    def test_device_plan_matches_host_if_cuda_available(self):
        try:
            import cupy as cp
            cp.zeros(1)
        except Exception as exc:
            self.skipTest(f'CUDA unavailable: {exc}')
        from de_mc_cf_v2_gpu_random import CFV2RandomPlan
        for div in (0., .499999, .5, 1.):
            for count in (0, 3, 4, 10):
                e = engine(); state = e.initialize([0, 1], [1234, 1235])
                rngs = [np.random.default_rng(s) for s in (1234, 1235)]
                for dst, src in zip(rngs, state.generators):
                    dst.bit_generator.state = src.bit_generator.state
                plan = CFV2RandomPlan(cp, rngs, 10, 5)
                close = np.tile(np.arange(10) < count, (2, 1))
                state.algorithm_state['div_norm_cpu'][:] = div
                host = random_plan(e, state, close, ~close)
                device = plan.generate(cp.asarray(close), cp.full(2, div), e)
                for a, b in zip(host, device):
                    np.testing.assert_array_equal(a, cp.asnumpy(b))
                plan.export_states(rngs)
                for a, b in zip(rngs, state.generators):
                    self.assertEqual(a.bit_generator.state, b.bit_generator.state)
        for device in ('hybrid', 'gpu'):
            e = engine(device=device)
            # Tiny numerical validation uses a GPU sphere, never a benchmark experiment.
            e.gpu_objective = SimpleNamespace(
                evaluate=lambda x: cp.sum(x*x, axis=-1), function_name='validation-sphere')
            e.selected_objective_backend = 'gpu'
            state = e.initialize([0], [1234], capture_trace=True)
            for epoch in range(1, 4):
                e._advance_one_epoch(state, epoch)
            _, cpu, _ = engine().run([0], [1234], capture_trace=True)
            np.testing.assert_allclose(cp.asnumpy(state.positions), cpu.positions, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
