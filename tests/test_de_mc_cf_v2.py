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
from macro_de_optimizer import MaCRO_DE
import main


def sphere(x):
    return float(np.sum(x * x))


def problem():
    return dict(bounds=FloatVar(lb=(-5.,) * 5, ub=(5.,) * 5),
                minmax='min', obj_func=sphere, log_to=None)


def engine(name='DE-MC-CF-v2', device='cpu', **kwargs):
    return BatchedDEEngine(name, sphere, [-5.] * 5, [5.] * 5, 3, 10,
                           compute_device=device, beta_min=.1, beta_max=.6,
                           mahalanobis_q=.5, objective_strategy='serial', **kwargs)


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
                          args.seed_base, args.compute_device), (30, 50, 2000, 30, 1234, main.COMPUTE_DEVICE))
        names = ['DE-MC-CF-v2', 'DE-MC-CF', 'MaCRO-DE', 'DSADE']
        signatures = [main.build_cache_signature(args, name, 'F12017') for name in names]
        self.assertEqual(len(set(signatures)), 4)
        self.assertEqual(signatures[0], main.build_cache_signature(args, 'MaCRO-DE-t-v2', 'F12017'))
        self.assertEqual(main.optimizer_scientific_parameters(names[0], args)['implementation_revision'],
                         'macro-D-scaled-coordinate-adaptive-pcr-v5')
        args.resolved_gpu_batch_size = None
        args.estimated_gpu_batch_capacity = None
        metadata = main.checkpoint_metadata(args, signatures[0], 'F12017', names[0], 0, 1234)
        self.assertTrue(main.checkpoint_metadata_compatible(metadata, metadata))
        legacy = dict(metadata, optimizer_parameters=dict(
            beta_min=.1, beta_max=.6, pcr=.1, mahalanobis_q=.5,
            implementation_revision='awad-close-far-beta-v2'))
        self.assertFalse(main.checkpoint_metadata_compatible(legacy, metadata))
        scalar_legacy = dict(metadata, optimizer_parameters=dict(
            mahalanobis_q=.5, implementation_revision='mahalanobis-adaptive-scalar-v3'))
        self.assertFalse(main.checkpoint_metadata_compatible(scalar_legacy, metadata))
        unscaled_legacy = dict(metadata, optimizer_parameters=dict(
            metadata['optimizer_parameters'], implementation_revision='mahalanobis-adaptive-pcr-coordinate-v4'))
        self.assertFalse(main.checkpoint_metadata_compatible(unscaled_legacy, metadata))
        configurations = list(main.optimizer_experiment_configurations(args))
        self.assertEqual(len(configurations), 12)
        nominal = dict(beta_min=main.MACRO_BETA_MIN, beta_max=main.MACRO_BETA_MAX,
                       mahalanobis_q=main.MACRO_MAHAL_Q)
        for label, name, variant in configurations:
            self.assertEqual(name, names[0])
            expected = dict(nominal)
            expected[variant.sensitivity_parameter] = variant.sensitivity_value
            actual = main.optimizer_scientific_parameters(name, variant)
            for key, value in expected.items():
                self.assertEqual(actual[key], value)
            self.assertNotIn('pcr', actual)
        self.assertEqual(len({main.build_cache_signature(v, n, 'F12017') for _, n, v in configurations}), 10)
        functions = ['F12017', 'F82017', 'F152017', 'F242017']
        self.assertEqual(main.select_experiment_functions(args, dict.fromkeys(functions)), functions)
        self.assertTrue(main.make_paths(args, create=False).res_dir.endswith('/sensitivity/DE-MC-CF-v2'))
        args.sensitivity_optimizer = 'MaCRO-DE'
        self.assertTrue(main.make_paths(args, create=False).res_dir.endswith('/sensitivity'))

    def test_scales_and_crossover(self):
        new = DE_MC_CF_V2(epoch=3, pop_size=10, beta_min=.2, beta_max=.8)
        new.problem = SimpleNamespace(n_dims=100)
        new.correct_solution = lambda x: x
        for dm in (0., .25, .5, 1.):
            new.div_norm_for_update = dm
            new.generator = np.random.default_rng(42)
            rng = np.random.default_rng(42)
            factors = new._sample_scale_factors()
            scale = np.clip(1.5 - dm, .5, 1.5)
            np.testing.assert_array_equal(factors, np.clip(rng.uniform(.2, .8, 100) * scale, .1, 1.5))
            self.assertEqual(np.unique(factors).size, 100)
            self.assertTrue(np.all((factors >= .2 * scale) & (factors <= .8 * scale)))
            new.cr = .1 + .25 * (1 - dm)
            j0 = rng.integers(0, 100)
            mask = rng.random(100) <= new.cr
            mask[j0] = True
            actual = new._binomial_crossover(np.zeros(100), np.ones(100))
            np.testing.assert_array_equal(actual, mask.astype(float))
            self.assertEqual(actual[j0], 1.)
        for key in ('pcr', 'wf', 'cr'):
            with self.assertRaises(TypeError):
                DE_MC_CF_V2(epoch=3, pop_size=10, **{key: .2})
        for low, high in ((-.1, .6), (.8, .2), (np.nan, .6), (.1, np.inf)):
            with self.assertRaises(ValueError):
                DE_MC_CF_V2(epoch=3, pop_size=10, beta_min=low, beta_max=high)

    def test_host_plan_adaptive_control_and_rng_order(self):
        e = engine()
        state = e.initialize([0, 1], [1234, 1235])
        state.algorithm_state['div_norm_cpu'][:] = [0., 1.]
        dm = np.tile(np.linspace(0, 1, 10), (2, 1))
        close = np.tile(np.arange(10) < 5, (2, 1))
        rngs = [np.random.default_rng() for _ in range(2)]
        for dst, src in zip(rngs, state.generators):
            dst.bit_generator.state = src.bit_generator.state
        donors, cross, factors, pcr = random_plan(e, state, close, ~close, dm)
        np.testing.assert_array_equal(pcr, .1 + .25 * (1 - dm))
        for r, rng in enumerate(rngs):
            for i in range(10):
                pool = np.arange(5, 10) if r == 0 else np.arange(5)
                candidates = pool[pool != i]
                np.testing.assert_array_equal(donors[r, i], rng.choice(candidates, 3, replace=False))
                D = state.algorithm_state['div_norm_cpu'][r]
                expected_f = np.clip(rng.uniform(e.beta_min, e.beta_max, 5) * np.clip(1.5 - D, .5, 1.5), .1, 1.5)
                np.testing.assert_array_equal(factors[r, i], expected_f)
                j0 = rng.integers(0, 5)
                mask = rng.random(5) <= pcr[r, i]
                mask[j0] = True
                np.testing.assert_array_equal(cross[r, i], mask)
            self.assertEqual(rng.bit_generator.state, state.generators[r].bit_generator.state)

    def test_macro_D_definition_and_F_clipping(self):
        new = DE_MC_CF_V2(epoch=3, pop_size=10)
        macro = MaCRO_DE(epoch=3, pop_size=10)
        new.solve(problem(), seed=1234)
        self.assertEqual(macro.div_norm_for_update, 1.)
        positions = new._positions(new.pop)
        self.assertEqual(new._awad(positions, None, None), macro._awad(positions, None, None))
        self.assertEqual(new.div_norm_for_update, new.div_norm_hist[-1])
        for D in (-1., 0., .5, 1., 2.):
            for low, high in ((.01, .02), (.2, .8), (4., 5.)):
                new.div_norm_for_update = D
                new.beta_min, new.beta_max = low, high
                new.generator = np.random.default_rng(17)
                rng = np.random.default_rng(17)
                scale = np.clip(1.5 - np.clip(D, 0, 1), .5, 1.5)
                expected = np.clip(rng.uniform(low, high, 5) * scale, .1, 1.5)
                np.testing.assert_array_equal(new._sample_scale_factors(), expected)
                e = engine()
                e.beta_min, e.beta_max = low, high
                state = e.initialize([0], [1234])
                state.algorithm_state['div_norm_cpu'][:] = D
                _, _, factors, pcr = random_plan(
                    e, state, np.ones((1, 10), dtype=bool), np.zeros((1, 10), dtype=bool),
                    np.full((1, 10), .25))
                np.testing.assert_array_equal(pcr, np.full((1, 10), .1 + .25 * .75))
                if high == .02:
                    np.testing.assert_array_equal(factors, np.full((1, 10, 5), .1))
                elif low == 4.:
                    np.testing.assert_array_equal(factors, np.full((1, 10, 5), 1.5))

    def test_mahalanobis_normalization_and_threshold(self):
        e = engine()
        state = e.initialize([0, 1], [1234, 1235])
        dist2 = e.backend.mahalanobis_distances(state.positions, 5, 'cholesky')
        dm = e.backend.normalized_mahalanobis(dist2)
        np.testing.assert_allclose(dm, np.sqrt(dist2) / np.sqrt(dist2).max(axis=1, keepdims=True))
        np.testing.assert_array_equal(e.backend.normalized_mahalanobis(np.zeros((2, 10))), 0)
        close, far, actual = e._classification(state.positions, 'cholesky', e.threshold)
        np.testing.assert_array_equal(close, dist2 <= e.threshold)
        np.testing.assert_array_equal(far, dist2 > e.threshold)
        np.testing.assert_array_equal(actual, dist2)

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

    def test_frozen_generation_and_greedy_selection(self):
        new = DE_MC_CF_V2(epoch=3, pop_size=10)
        # Historical equivalence test: disable extra RNG draws solely to prove
        # every other scalar operation unchanged. Adaptive control is now tested
        # directly; frozen donors and greedy fitness survival remain invariants.
        new.solve(problem(), seed=1234)
        before = new._positions(new.pop).copy()
        fitness = np.array([agent.target.fitness for agent in new.pop])
        captured = []
        sample = new._sample_mutation_indices
        def record(pop, idx):
            captured.append(pop.copy())
            return sample(pop, idx)
        new._sample_mutation_indices = record
        new.evolve(1)
        for pop in captured:
            np.testing.assert_array_equal(pop, before)
        self.assertTrue(np.all([a.target.fitness <= f for a, f in zip(new.pop, fitness)]))

    def test_evolve_uses_target_mahalanobis_control(self):
        new = DE_MC_CF_V2(epoch=3, pop_size=10)
        new.solve(problem(), seed=1234)
        dm = np.linspace(0, 1, 10)
        recorded = []
        sample_f = new._sample_scale_factors
        def record_f():
            recorded.append((new.cr, new.div_norm_for_update))
            return sample_f()
        new._sample_scale_factors = record_f
        # AWAD controls routing only for group sampling; it cannot replace dM.
        # The same delayed MaCRO-DE diversity state also scales F.
        new.div_norm_for_update = .123
        with patch.object(new.backend, 'close_far_indices', return_value=(
                np.arange(10), np.array([], dtype=int), dm**2)):
            new.evolve(1)
        np.testing.assert_array_equal(np.array(recorded)[:, 0], .1 + .25 * (1 - dm))
        np.testing.assert_array_equal(np.array(recorded)[:, 1], np.full(10, .123))

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
        np.testing.assert_array_equal(trace['pcr_values'], .1 + .25 * (1 - trace['dM']))
        factors = trace['f_vectors']
        self.assertTrue(np.all((factors >= .1) & (factors <= .3)))
        unclipped = factors[factors > .1]
        self.assertEqual(np.unique(unclipped).size, unclipped.size)
        self.assertTrue(np.all(state.trace['first_generation_fitness'] <= np.sum(x*x, axis=-1)))

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
                dm = np.tile(np.linspace(0, 1, 10), (2, 1))
                host = random_plan(e, state, close, ~close, dm)
                device = plan.generate(cp.asarray(close), cp.full(2, div), e, cp.asarray(dm))
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
