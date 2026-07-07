import copy

import numpy as np
from mealpy.optimizer import Optimizer


class DBOOptimizer(Optimizer):
    """Dung Beetle Optimizer adapted to Mealpy's Optimizer API."""

    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.support_parallel_modes = False
        self.pbest = None

    def before_main_loop(self):
        self.pbest = [copy.deepcopy(agent) for agent in self.pop]

    def _positions(self, pop):
        return np.array([agent.solution for agent in pop], dtype=float)

    def _agent_from_position(self, position):
        position = self.correct_solution(position)
        agent = self.generate_empty_agent(position)
        agent.target = self.get_target(position)
        return agent

    def _best_from(self, pop):
        best = pop[0]
        for agent in pop[1:]:
            if self.compare_target(agent.target, best.target, self.problem.minmax):
                best = agent
        return best

    def _worst_from(self, pop):
        worst = pop[0]
        for agent in pop[1:]:
            if self.compare_target(worst.target, agent.target, self.problem.minmax):
                worst = agent
        return worst

    def evolve(self, epoch):
        if self.pbest is None:
            self.before_main_loop()

        pop_pos = self._positions(self.pop)
        pbest_pos = self._positions(self.pbest)
        best = self._best_from(self.pbest)
        worst = self._worst_from(self.pbest)
        best_pos = best.solution.copy()
        worst_pos = worst.solution.copy()
        previous_pos = pop_pos.copy()

        n_roll = max(1, int(round(0.20 * self.pop_size)))
        n_brood = max(n_roll + 1, int(round(0.40 * self.pop_size)))
        n_small = max(n_brood + 1, int(round(0.65 * self.pop_size)))
        n_brood = min(n_brood, self.pop_size)
        n_small = min(n_small, self.pop_size)

        ratio = 1.0 - epoch / max(self.epoch, 1)
        local_best = copy.deepcopy(best)
        local_best_pos = local_best.solution.copy()
        new_pop = []

        for idx in range(self.pop_size):
            current = pop_pos[idx]
            personal = pbest_pos[idx]

            if idx < n_roll:
                if self.generator.random() < 0.9:
                    direction = 1.0 if self.generator.random() > 0.1 else -1.0
                    step = 0.3 * np.abs(personal - worst_pos) + direction * 0.1 * previous_pos[idx]
                    new_pos = personal + step
                else:
                    angle = int(self.generator.integers(1, 180))
                    while angle in (90, 180):
                        angle = int(self.generator.integers(1, 180))
                    theta = np.deg2rad(angle)
                    new_pos = personal + np.tan(theta) * np.abs(personal - previous_pos[idx])
            elif idx < n_brood:
                low = self.correct_solution(local_best_pos * (1.0 - ratio))
                high = self.correct_solution(local_best_pos * (1.0 + ratio))
                new_pos = local_best_pos + self.generator.random(self.problem.n_dims) * (personal - low)
                new_pos += self.generator.random(self.problem.n_dims) * (personal - high)
            elif idx < n_small:
                low = self.correct_solution(best_pos * (1.0 - ratio))
                high = self.correct_solution(best_pos * (1.0 + ratio))
                new_pos = personal + self.generator.normal() * (personal - low)
                new_pos += self.generator.random(self.problem.n_dims) * (personal - high)
            else:
                spread = (np.abs(personal - local_best_pos) + np.abs(personal - best_pos)) / 2.0
                new_pos = best_pos + self.generator.normal(size=self.problem.n_dims) * spread

            candidate = self._agent_from_position(new_pos)
            current_agent = self.get_better_agent(candidate, self.pop[idx], self.problem.minmax)
            pbest_agent = self.get_better_agent(current_agent, self.pbest[idx], self.problem.minmax)
            self.pbest[idx] = copy.deepcopy(pbest_agent)
            new_pop.append(current_agent)

            if self.compare_target(candidate.target, best.target, self.problem.minmax):
                best = candidate
                best_pos = candidate.solution.copy()
            if self.compare_target(candidate.target, local_best.target, self.problem.minmax):
                local_best = candidate
                local_best_pos = candidate.solution.copy()

        self.pop = new_pop


DBO = DBOOptimizer
