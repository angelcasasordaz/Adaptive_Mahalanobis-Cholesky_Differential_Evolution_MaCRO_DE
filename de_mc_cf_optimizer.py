import numpy as np

from de_mc_optimizer import DE_MC


class DE_MC_CF(DE_MC):
    """DE-MC with MaCRO-DE's AWAD-driven close/far pool routing."""

    IMPLEMENTATION_REVISION = "awad-close-far-v2"

    def initialize_variables(self):
        self.div_awad_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_max_seen = None
        self.div_norm_for_update = 1.0
        self._awad_pair_indices = None
        self.routing_counts = {"close": 0, "far": 0, "fallback": 0}

    def before_main_loop(self):
        pop_pos = self._positions(self.pop)
        div0 = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_max_seen = max(div0, self.EPSILON)

    def _awad(self, pop_pos, lb, ub):
        """Match MaCRO-DE's AWAD calculation and safeguards exactly."""
        _ = lb, ub
        npop, n_dims = pop_pos.shape
        med_dim = np.median(pop_pos, axis=0)
        div_dim = np.mean(np.abs(pop_pos - med_dim), axis=0)
        div = float(np.sum(div_dim) / max(n_dims, 1))

        unique_count = np.unique(pop_pos, axis=0).shape[0]
        non_repeat_percent = (unique_count * 100.0) / max(npop, 1)

        std_devs = np.std(pop_pos, axis=0)
        std_devs[std_devs == 0] = 1e-5
        if npop <= 1:
            min_distance = 0.0
        else:
            pair_count = npop * (npop - 1) // 2
            if (
                self._awad_pair_indices is None
                or self._awad_pair_indices[0].size != pair_count
            ):
                self._awad_pair_indices = np.triu_indices(npop, k=1)
            left, right = self._awad_pair_indices
            diff = (pop_pos[right] - pop_pos[left]) / std_devs
            min_distance = float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
            if not np.isfinite(min_distance):
                min_distance = 0.0

        penalty_factor = ((min_distance + 0.1) ** 2) / (1.0 + min_distance**2)
        return float(div * 0.1 * non_repeat_percent * penalty_factor)

    @staticmethod
    def _route_for_diversity(div_norm):
        return "close" if float(div_norm) >= 0.5 else "far"

    @classmethod
    def routing_diagnostic(cls):
        """Tiny side-effect-free check for the two diversity routing branches."""
        routes = {
            "high_diversity": cls._route_for_diversity(0.5),
            "low_diversity": cls._route_for_diversity(0.499999),
        }
        routes["passed"] = (
            routes["high_diversity"] == "close"
            and routes["low_diversity"] == "far"
        )
        return routes

    def _mutation_pool_indices(self, pop_pos, current_idx):
        close, far = self._close_far_indices(pop_pos)
        route = self._route_for_diversity(self.div_norm_for_update)
        selected = close if route == "close" else far
        if self._valid_candidates(selected, current_idx).size >= 3:
            self.routing_counts[route] += 1
            return selected
        self.routing_counts["fallback"] += 1
        return np.arange(self.pop_size)

    def evolve(self, epoch):
        if self.div_max_seen is None:
            self.before_main_loop()
        super().evolve(epoch)

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch - 1] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(
            np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0)
        )
        self.div_norm_hist[epoch - 1] = div_norm_now
        self.div_norm_for_update = div_norm_now

    @property
    def covariance_inverse_method(self):
        # Preserve this separate ablation's existing inverse-based arithmetic.
        return "cholesky"
