"""The DSS decision rule, shared by evaluation (eval_dss_gpu.py) and targeted data
collection (augment_gpu.py): score every candidate action with the yield model, drop
candidates the volume model predicts would overfill the vessel, recommend the best one,
or keep the recipe if no gain is predicted.
"""
import numpy as np
import pandas as pd

from dss_common import ACTION_MVS, CANDIDATES, RECIPE_ACTION, V_LIMIT_L
from train_dss import FEATURES, STATE_COLS

MIN_PRED_GAIN_KG = 0.0
CAND_MAT = np.array([[a[mv] for mv in ACTION_MVS] for a in CANDIDATES])


def frame(state, actions):
    """Model input rows for one state and a list of actions."""
    return pd.DataFrame([{**state, **{f"a_{mv}": a[mv] for mv in ACTION_MVS}} for a in actions])[FEATURES]


def candidate_matrix(states):
    """All candidates for every state: rows ordered state-major, columns = FEATURES."""
    s = np.asarray([[st[c] for c in STATE_COLS] for st in states], dtype=float)
    X = np.hstack([np.repeat(s, len(CANDIDATES), axis=0), np.tile(CAND_MAT, (len(states), 1))])
    return pd.DataFrame(X, columns=STATE_COLS + [f"a_{mv}" for mv in ACTION_MVS])[FEATURES]


def recommend(states, ymodel, vmodel):
    """One recommendation per state: dict with action, changed, pred_gain, n_feasible."""
    n = len(states)
    X = candidate_matrix(states)
    preds = ymodel.predict(X).reshape(n, -1)
    vpred = vmodel.predict(X).reshape(n, -1)
    i_recipe = int(np.flatnonzero(np.all(CAND_MAT == 1.0, axis=1))[0])
    out = []
    for i in range(n):
        feasible = vpred[i] <= V_LIMIT_L
        action, pred_gain = RECIPE_ACTION, 0.0
        if feasible.any():
            ib = int(np.argmax(np.where(feasible, preds[i], -np.inf)))
            gain = float(preds[i, ib] - preds[i, i_recipe])
            if gain > MIN_PRED_GAIN_KG:
                action, pred_gain = CANDIDATES[ib], gain
        out.append({"action": action, "changed": action != RECIPE_ACTION, "pred_gain": pred_gain,
                    "n_feasible": int(feasible.sum())})
    return out
