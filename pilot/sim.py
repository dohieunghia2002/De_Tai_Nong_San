"""Thin wrapper around the vendored PenSimPy (IndPenSim Python port) for the pilot.

Changes relative to PenSimPy 1.2.5 (all listed in README.md):
1. Solver: PenSimPy's C++ `fastodeint` (boost::odeint `integrate`, i.e. adaptive
   Dormand-Prince RK45, abs/rel tol 1e-6) needs a C++ build, so it is emulated with
   scipy `solve_ivp(method="RK45", rtol=1e-6, atol=1e-6)`. [H+] is floored at 1e-12
   inside trial stages only. (The in-place mutation of y[6] in the Python ODE was
   also fixed, see pensimpy/ode/.)
2. Recipe semantics: a setpoint {time: t, value: v} means "v applies UNTIL t"
   (as in the original MATLAB IndPenSim). PenSimPy forward-fills instead, which
   turns the 2-hour broth discharges into 28-hour drains and empties the vessel.
3. Recipe-driven mode (Raman_spec = 0): PAA follows the recipe instead of the
   Raman-based PID loop, like IndPenSim batches 1-30.
"""
import copy

import numpy as np
from scipy.integrate import solve_ivp

import pensimpy.peni_env_setup as pes
from pensimpy.constants import NUM_STEPS, STEP_IN_HOURS
from pensimpy.data import constants as C
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.ode.indpensim_ode_py import indpensim_ode_py


class _RK45:
    """Drop-in replacement for the `fastodeint` module."""

    @staticmethod
    def integrate(x00, par, t0, t1, dt):
        def rhs(t, y):
            # Trial RK stages can push [H+] (y[6]) slightly below zero under strong
            # base dosing; floor it for the derivative evaluation only. Accepted
            # steps are still controlled by the RK45 error estimate.
            if y[6] <= 1e-12:
                y = y.copy()
                y[6] = 1e-12
            return indpensim_ode_py(t, y, par)
        sol = solve_ivp(rhs, (t0, t1), x00, method="RK45", rtol=1e-6, atol=1e-6, first_step=dt)
        return list(sol.y[:, -1])


pes.fastodeint = _RK45

# Manipulated variables in the order PenSimEnv.step expects them.
MV_NAMES = ["Fs", "Foil", "Fg", "pressure", "discharge", "Fw", "Fpaa"]

DEFAULT_PROFILES = {
    "Fs": C.FS_DEFAULT_PROFILE, "Foil": C.FOIL_DEFAULT_PROFILE, "Fg": C.FG_DEFAULT_PROFILE,
    "pressure": C.PRESS_DEFAULT_PROFILE, "discharge": C.DISCHARGE_DEFAULT_PROFILE,
    "Fw": C.WATER_DEFAULT_PROFILE, "Fpaa": C.PAA_DEFAULT_PROFILE,
}


def recipe_value(profile, t_hours):
    """Value of a setpoint profile at time t: the first setpoint whose time >= t."""
    for sp in profile:
        if t_hours <= sp["time"]:
            return sp["value"]
    return profile[-1]["value"]


def recipe_values(k):
    """Recipe values of all manipulated variables at simulation step k (1-based)."""
    t = k * STEP_IN_HOURS
    return {name: recipe_value(DEFAULT_PROFILES[name], t) for name in MV_NAMES}


def _dummy_combo():
    # PenSimEnv requires a RecipeCombo, but we always pass the MVs explicitly.
    return RecipeCombo({n: Recipe(p, n) for n, p in DEFAULT_PROFILES.items()})


class Batch:
    """One IndPenSim batch that can be advanced step by step and forked.

    `seed` selects the batch's initial conditions and disturbance profiles,
    exactly like PenSimEnv.get_batches(random_seed=seed).
    """

    def __init__(self, seed):
        self.env = pes.PenSimEnv(recipe_combo=_dummy_combo(), fast=True)
        # Recipe-driven mode (like IndPenSim batches 1-30): no simulated Raman
        # device, so PAA is not overridden by the Raman-based PID loop.
        self.env.ctrl_flags.Raman_spec = 0
        self.env.random_seed_ref = seed
        _, self.x = self.env.reset()
        self.x.Raman_Spec.Intensity = np.zeros((1, 1))  # unused; keeps forks light
        self.env.yield_pre = 0
        self.k = 0                      # last completed step
        self.total_yield = 0.0          # kg, in-vessel + discharged (PenSimPy reward sum)

    @property
    def done(self):
        return self.k >= NUM_STEPS

    @property
    def t_hours(self):
        return self.k * STEP_IN_HOURS

    def step(self, mv):
        """Advance one 12-minute step with manipulated variables `mv` (dict)."""
        self.k += 1
        _, self.x, reward, _ = self.env.step(self.k, self.x, *[mv[n] for n in MV_NAMES])
        self.total_yield += reward
        return reward

    def run_until(self, t_hours, policy=None):
        """Run until t_hours. `policy(k) -> dict of multipliers` scales the recipe."""
        k_end = min(NUM_STEPS, int(round(t_hours / STEP_IN_HOURS)))
        while self.k < k_end:
            mv = recipe_values(self.k + 1)
            if policy is not None:
                for name, mult in policy(self.k + 1).items():
                    mv[name] = mv[name] * mult
            self.step(mv)

    def fork(self):
        return copy.deepcopy(self)


def run_recipe_batch(seed):
    b = Batch(seed)
    b.run_until(NUM_STEPS * STEP_IN_HOURS)
    return b


if __name__ == "__main__":
    import time
    for seed in [1, 2, 3]:
        t0 = time.time()
        b = run_recipe_batch(seed)
        V = np.array(b.x.V.y)
        print(f"seed {seed}: yield {b.total_yield:7.1f} kg | V min/max {V.min():8.0f}/{V.max():8.0f} L"
              f" | final P {b.x.P.y[-1]:.2f} g/L | {time.time() - t0:5.1f} s")
