"""Pilot: mango harvest decision support from NIR (Mendeley v3 file).
1) Reproduce the Anderson et al. 2020 PLS baseline (SG d2 w17 p2 on full spectrum -> trim 684-990 nm, 11 LV).
2) Decision layer per fruit (mean of its duplicate spectra): ready if predicted DMC >= T.
3) Split-conformal deferral (PLS trained on Cal, calibrated on Tuning, fruit level): defer if T lies in the interval.
4) Shift handling: per new population, k fruit measured destructively -> bias correction; Q-residual OOD flag.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

F = Path(__file__).resolve().parents[1] / 'data' / 'MangoDMC_NIR_Data_v3.csv'
d = pd.read_csv(F)
wl = [c for c in d.columns if c.isdigit()]
X = savgol_filter(d[wl].values, 17, 2, deriv=2, axis=1)            # full spectrum first, then trim
keep = [i for i, c in enumerate(wl) if 684 <= int(c) <= 990]
X = X[:, keep]
y = d['DM'].values
d['fruit'] = d.groupby(['Set', 'Pop', 'Type', 'Cultivar', 'DM'], sort=False).ngroup()
S = {s: (d['Set'] == s).values for s in ['Cal', 'Tuning', 'Val Ext', 'Val Ext 2']}
tr = S['Cal'] | S['Tuning']
print(f'variables after trim: {X.shape[1]} (paper: 103) | train spectra {tr.sum()} (paper: 10,243)')

# 1) reproduction ------------------------------------------------------------
pls = PLSRegression(11, scale=False).fit(X[tr], y[tr])
for s, paper in [('Val Ext', '1.01 (Anderson 2020 PLS)'), ('Val Ext 2', '1.87 / bias 1.56 (Walsh 2024 best PLS)')]:
    p = pls.predict(X[S[s]]).ravel(); e = p - y[S[s]]
    print(f'[{s}] PLS 11LV: RMSEP {np.sqrt(np.mean(e**2)):.2f}, bias {e.mean():+.2f} | paper: {paper}')

# 2-4) decision layer ---------------------------------------------------------
m = PLSRegression(11, scale=False).fit(X[S['Cal']], y[S['Cal']])
pred = m.predict(X).ravel()
fr = d.assign(pred=pred).groupby('fruit').agg(Set=('Set', 'first'), Pop=('Pop', 'first'), y=('DM', 'first'), p=('pred', 'mean'))
cal = fr[fr.Set == 'Tuning']
q90 = np.quantile(np.abs(cal.p - cal.y), np.ceil(0.9 * (len(cal) + 1)) / len(cal))
# Q-residual OOD score (spectral reconstruction error outside the PLS x-score space)
T_, P_ = m.x_scores_, m.x_loadings_
Xc = X - m._x_mean
Q = ((Xc - (Xc @ m.x_rotations_) @ P_.T) ** 2).sum(1)
q_lim = np.quantile(Q[S['Cal']], 0.95)
rng = np.random.default_rng(0)


def report(sub, name, T):
    yt, pt = sub.y.values, sub.p.values
    ok = (pt >= T) == (yt >= T)
    dec = np.abs(pt - T) >= q90
    fr_ready_wrong = np.mean((pt >= T) & (yt < T))
    print(f'  {name:<28} acc {ok.mean():.1%} | false-ready {fr_ready_wrong:.1%} | interval coverage {np.mean(np.abs(pt-yt) <= q90):.0%} '
          f'| defer: decides {dec.mean():.0%}, acc {ok[dec].mean():.1%}, false-ready {np.mean(((pt>=T)&(yt<T))[dec]):.1%}')


for T in [float(t) for t in (sys.argv[1:] or ['15'])]:
    print(f'\n=== threshold T = {T}% DMC (illustrative) | conformal 90% half-width q = {q90:.2f} (calibrated on Tuning) ===')
    for s in ['Val Ext', 'Val Ext 2']:
        sub = fr[fr.Set == s]
        print(f'[{s}] fruit {len(sub)}, truly ready {np.mean(sub.y >= T):.0%}, spectra flagged OOD (Q > cal 95th pct): {np.mean(Q[S[s]] > q_lim):.0%}')
        report(sub, 'raw PLS', T)
        for k in (5, 10):  # bias correction with k destructive reference fruit per population
            accs = []
            for rep in range(30):
                parts = []
                for pop, g in sub.groupby('Pop'):
                    idx = rng.choice(len(g), size=min(k, len(g)), replace=False)
                    b = (g.p.values[idx] - g.y.values[idx]).mean()
                    parts.append(g.drop(g.index[idx]).assign(p=lambda z: z.p - b))
                accs.append(pd.concat(parts))
            allc = pd.concat(accs)
            report(allc, f'bias-corrected, k={k}/pop', T)
