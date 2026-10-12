"""Shared definitions for the pilot DSS: decision points, actions, state features."""
import itertools
import math
import os

import numpy as np

from pensimpy.constants import NUM_STEPS, STEP_IN_HOURS

# Experiment settings. v1 = first Colab run (results kept separately);
# v2 = wider action range + vessel-volume constraint + fixed-rule baselines (CPU);
# v3 = v2 + level controller for every policy, simulated on GPU (jaxsim.py);
EXPERIMENTS = {
    "v1": {"train_range": (0.7, 1.3), "levels": [0.8, 0.9, 1.0, 1.1, 1.2]},
    "v2": {"train_range": (0.6, 1.6), "levels": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
    "v3": {"train_range": (0.6, 1.6), "levels": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], "level_control": True},
    # v4 = v3 + targeted data around the DSS's own recommendations (augment_gpu.py)
    "v4": {"train_range": (0.6, 1.6), "levels": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], "level_control": True},
}
EXP = os.environ.get("PILOT_EXP", "v4")
OUT_DIR = f"runs/{EXP}"          # data/, models/, results/ of this experiment live here
LEVEL_CONTROL = EXPERIMENTS[EXP].get("level_control", False)
V_LIMIT_L = 100_000              # IndPenSim bioreactor volume (100 m3); broth must stay below it

# Operator decision points (hours) and the window each decision covers, following
# the 24 h decision rhythm of Mlinaric & Glavan (ERK 2025 / Univ. Ljubljana 2025).
DECISION_TIMES = [96, 120, 144, 168, 192, 216]
BATCH_END_H = NUM_STEPS * STEP_IN_HOURS  # 230 h


def window_end(t_dec):
    return min(t_dec + 24, BATCH_END_H)


# Manipulated variables the DSS may adjust, as multipliers on the recipe value.
ACTION_MVS = ["Fs", "Foil", "Fg", "Fpaa"]
ACTION_LABELS = {"Fs": "sugar feed", "Foil": "oil feed", "Fg": "aeration", "Fpaa": "PAA feed"}
TRAIN_RANGE = EXPERIMENTS[EXP]["train_range"]      # random actions for training data
CANDIDATE_LEVELS = EXPERIMENTS[EXP]["levels"]      # DSS search grid (inside training range)
RECIPE_ACTION = {mv: 1.0 for mv in ACTION_MVS}

CANDIDATES = [dict(zip(ACTION_MVS, c)) for c in itertools.product(CANDIDATE_LEVELS, repeat=len(ACTION_MVS))]


def action_policy(action, t_start, t_end):
    """Recipe multipliers applied between t_start and t_end (hours)."""
    def policy(k):
        t = k * STEP_IN_HOURS
        return action if t_start < t <= t_end else {}
    return policy


OFFLINE_EVERY_H = 12   # IndPenSim: lab samples every 12 h ...
OFFLINE_DELAY_H = 4    # ... with results available 4 h later


def _offline(arr, k):
    """Latest lab value available at step k: the true value at the last 12-h
    sampling time whose result (4 h delay) is already back. Computed here because
    PenSimPy's fast path shifts timestamps by h_ode and never records offline data."""
    t = k * STEP_IN_HOURS
    t_sample = np.floor((t - OFFLINE_DELAY_H) / OFFLINE_EVERY_H) * OFFLINE_EVERY_H
    if t_sample <= 0:
        return np.nan
    return float(arr[int(round(t_sample / STEP_IN_HOURS)) - 1])


def state_features(batch):
    """Features an operator could actually see at the current time.

    Online sensors at the current step, 12 h means of off-gas rates, and the last
    available OFFLINE lab values (penicillin, biomass, PAA, NH3, viscosity). True
    internal states (e.g. penicillin concentration P) are NOT used.
    """
    x, k = batch.x, batch.k
    i = k - 1
    w = slice(max(0, k - int(12 / STEP_IN_HOURS)), k)
    h_plus = x.pH.y[i]
    return {
        "t_h": k * STEP_IN_HOURS,
        "V": x.V.y[i], "Wt": x.Wt.y[i], "T": x.T.y[i],
        "pH": -math.log10(h_plus) if h_plus > 0 else np.nan,
        "DO2": x.DO2.y[i], "O2_offgas": x.O2.y[i], "CO2_offgas": x.CO2outgas.y[i],
        "OUR": x.OUR.y[i], "CER": x.CER.y[i],
        "OUR_12h": float(np.mean(x.OUR.y[w])), "CER_12h": float(np.mean(x.CER.y[w])),
        "DO2_12h": float(np.mean(x.DO2.y[w])),
        "P_off": _offline(x.P.y, k), "X_off": _offline(x.X.y, k),
        "PAA_off": _offline(x.PAA.y, k), "NH3_off": _offline(x.NH3.y, k),
        "Visc_off": _offline(x.Viscosity.y, k),
    }


def action_features(action):
    return {f"a_{mv}": action[mv] for mv in ACTION_MVS}


def state_features_batch(bs):
    """state_features() for every batch of a jaxsim.BatchSet, as a DataFrame (one row per batch)."""
    import pandas as pd
    k = bs.k
    i = k - 1
    w = slice(max(0, k - int(12 / STEP_IN_HOURS)), k)
    h = {key: bs.history(key) for key in ["H", "T", "V", "Wt", "DO2", "O2", "CO2", "OUR", "CER",
                                          "P", "X", "PAA", "NH3", "Visc"]}
    t_sample = np.floor((k * STEP_IN_HOURS - OFFLINE_DELAY_H) / OFFLINE_EVERY_H) * OFFLINE_EVERY_H
    j = int(round(t_sample / STEP_IN_HOURS)) - 1 if t_sample > 0 else None

    def off(key):
        return h[key][:, j] if j is not None else np.full(len(bs), np.nan)

    hp = h["H"][:, i]
    return pd.DataFrame({
        "t_h": np.full(len(bs), k * STEP_IN_HOURS),
        "V": h["V"][:, i], "Wt": h["Wt"][:, i], "T": h["T"][:, i],
        "pH": np.where(hp > 0, -np.log10(np.where(hp > 0, hp, 1.0)), np.nan),
        "DO2": h["DO2"][:, i], "O2_offgas": h["O2"][:, i], "CO2_offgas": h["CO2"][:, i],
        "OUR": h["OUR"][:, i], "CER": h["CER"][:, i],
        "OUR_12h": h["OUR"][:, w].mean(1), "CER_12h": h["CER"][:, w].mean(1), "DO2_12h": h["DO2"][:, w].mean(1),
        "P_off": off("P"), "X_off": off("X"), "PAA_off": off("PAA"), "NH3_off": off("NH3"), "Visc_off": off("Visc"),
    })


def usage_since(batch, k0):
    """Max broth volume and sugar/oil fed (L) over steps k0+1 .. batch.k."""
    x, k = batch.x, batch.k
    return {"vmax": float(np.max(x.V.y[k0:k])),
            "sugar_L": float(np.sum(x.Fs.y[k0:k]) * STEP_IN_HOURS),
            "oil_L": float(np.sum(x.Foil.y[k0:k]) * STEP_IN_HOURS)}
