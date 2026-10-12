"""Vectorised IndPenSim in JAX: simulates many batches at once (GPU if available).

It reproduces the CPU path (sim.py + PenSimPy PenSimEnv.step) step by step:
  - the same ODEs (pensimpy/ode/indpensim_ode_py.py, with the y[6] fix and the
    [H+] floor used in sim.py),
  - the same PID loops for pH and temperature, including the quirks of the
    original code (pH error history without the set-point, heating PID seeded with
    the cooling-water value), so results match the CPU runs,
  - the same integrator: scipy RK45 step control (rtol = atol = 1e-6, RMS error
    norm, first step h_ode), each 12-min step integrated over [t_{k-1}, t_k + h_ode],
  - the same clamps on the stored state and the same yield accounting.
Per-batch initial conditions and disturbances come from PenSimPy itself (CPU).

Optional level controller (experiment v3): when the broth volume exceeds V_HIGH the
broth is bled at LC_RATE L/h until it drops below V_LOW. Bled broth counts as
harvested product, like the scheduled discharges in IndPenSim.
validate_gpu.py checks this module against the CPU results of experiment v2.
"""
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

import pensimpy.peni_env_setup as pes  # noqa: E402
from pensimpy.constants import NUM_STEPS, STEP_IN_HOURS  # noqa: E402
from sim import MV_NAMES, _dummy_combo, recipe_values  # noqa: E402

H = STEP_IN_HOURS            # 0.2 h per control step
H_ODE = H / 40               # first RK step, as in PenSimEnv.step
RTOL = ATOL = 1e-6
PI = 3.141592653589793
V_HIGH, V_LOW, LC_RATE = 98_000.0, 95_000.0, 4000.0   # level controller (L, L, L/h)
MAX_RK_ITERS = 5000

# Recipe values for every step k = 1..NUM_STEPS (row k-1), columns in MV_NAMES order.
REC = jnp.asarray(np.array([[recipe_values(k)[n] for n in MV_NAMES] for k in range(1, NUM_STEPS + 1)]))
HIST_KEYS = ["H", "T", "V", "Wt", "DO2", "O2", "CO2", "OUR", "CER", "P", "X", "PAA", "NH3",
             "Visc", "Fs", "Foil", "Fg", "dis", "muX"]

# Dormand-Prince coefficients exactly as in scipy.integrate.RK45
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
_A = [np.array([]), np.array([1 / 5]), np.array([3 / 40, 9 / 40]), np.array([44 / 45, -56 / 15, 32 / 9]),
      np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
      np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656])]
_B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
_E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40])


def rhs(t, y, par):
    """IndPenSim right-hand side for N batches. t [N], y [N,33], par [N,31] -> [N,33]."""
    Y = [y[:, i] for i in range(33)]
    Y[6] = jnp.where(Y[6] <= 1e-12, 1e-12, Y[6])      # [H+] floor (sim.py)
    p = [par[:, i] for i in range(31)]

    ratio_mu_e_mu_b, P_std_dev, mean_P = 0.4, 0.0015, 0.002
    mu_v, mu_a, mu_diff, beta_1 = 1.71e-4, 3.5e-3, 5.36e-3, 0.006
    K_b, K_diff_L, K_e, K_v = 0.05, 0.09, 0.009, 0.05
    delta_r, k_v, D, rho_a0, rho_d = 0.75e-4, 3.22e-5, 2.66e-11, 0.35, 0.18
    r_0, delta_0 = 1.5e-4, 1e-4
    Y_sX, Y_sP, m_s, Y_O2_X, Y_O2_P, m_O2_X = 1.85, 0.9, 0.029, 650, 160, 17.5
    ka, kb, kc, kd = 0.38, 0.34, -0.38, 0.25
    Henrys_c, n_imp, r, r_imp, Po, epsilon = 0.0251, 3, 2.1, 0.85, 5, 0.1
    R = 8.314
    X_crit_DO2, P_crit_DO2, A_inhib = 0.1, 0.3, 1
    Tf, Tw, Th, Tair = 288, 288, 333, 290
    C_ps, C_pw, dealta_H_evap, U_jacket, A_c = 5.9, 4.18, 2430.7, 36, 105
    Eg, Ed, k_g, k_d, Y_QX = 14880, 173250, 450, 2.5e+29, 25
    gamma1, gamma2, m_ph, K1, K2 = 0.0325e-5, 2.5e-11, 0.0025, 1e-5, 2.5e-8
    N_conc_oil, N_conc_shot, Y_NX, Y_NP, m_N, X_crit_N = 20000, 400000, 10, 80, 0.03, 150
    Y_PAA_P, Y_PAA_X, m_PAA, X_crit_PAA, P_crit_PAA = 187.5, 45, 1.05, 2400, 200
    B_1, B_2, B_3, B_4, B_5 = -64.29, -1.825, 0.3649, 0.1280, -4.9496e-04
    delta_c_0, k3, k1, k2, t1c, t2c = 0.89, 0.005, 0.001, 0.0001, 1, 250
    q_co2, X_crit_CO2, alpha_evp, beta_T = 0.1353, 7570, 5.2400e-4, 2.88
    pho_g, pho_oil, pho_w, pho_paa = 1540, 900, 1000, 1000
    N2_in, C_CO2_in, Tv, T0, alpha_1 = 0.79, 0.033, 373, 273, 2451.8

    alpha_kla, N_conc_paa = p[2], p[3]
    Fs, Fg, RPM, Fc, Fh, Fb, Fa, step1 = p[6], p[7] / 60, p[8], p[9], p[10], p[11], p[12], p[13]
    Fw = jnp.where(p[14] < 0, 0.0, p[14])
    pressure, F_discharge, Fpaa, Foil, NH3_shots = p[15], p[17], p[18], p[19], p[20]
    viscosity = Y[9]                                    # viscosity flag = 0
    pho_b = 1100 + Y[3] + Y[11] + Y[12] + Y[13] + Y[14]

    # disturbances (dist flag = 1)
    mu_p = p[0] + p[22]
    mux_max = p[1] + p[23]
    c_s = 600 + p[24]
    c_oil = 1000 + p[25]
    abc = 0.033 + p[26]
    PAA_c = p[4] + p[27]
    Tcin = 285 + p[28]
    O_2_in = 0.21 + p[29]

    A_t1 = Y[10] / (Y[11] + Y[12] + Y[13] + Y[14])
    s, a_1, a_0, a_3 = Y[0], Y[12], Y[11], Y[13]
    total_X = Y[11] + Y[12] + Y[13] + Y[14]
    h_b = (Y[4] / 1000) / (PI * r ** 2)
    h_b = h_b * (1 - epsilon)
    pressure_bottom = 1 + pressure + pho_b * h_b * 9.81e-5
    pressure_top = 1 + pressure
    total_pressure = (pressure_bottom - pressure_top) / jnp.log(pressure_bottom / pressure_top)
    viscosity = jnp.where(viscosity < 4, 1.0, viscosity)
    DOstar_tp = total_pressure * O_2_in / Henrys_c

    # inhibition flag = 2
    pH_inhib = 1 / (1 + (Y[6] / K1) + (K2 / Y[6]))
    NH3_inhib = 0.5 * (1 - jnp.tanh(A_inhib * (X_crit_N - Y[30])))
    T_inhib = k_g * jnp.exp(-(Eg / (R * Y[7]))) - k_d * jnp.exp(-(Ed / (R * Y[7])))
    CO2_inhib = 0.5 * (1 + jnp.tanh(A_inhib * (X_crit_CO2 - Y[28] * 1000)))
    DO_2_inhib_X = 0.5 * (1 - jnp.tanh(A_inhib * (X_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - Y[1])))
    DO_2_inhib_P = 0.5 * (1 - jnp.tanh(A_inhib * (P_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - Y[1])))
    PAA_inhib_X = 0.5 * (1 + (jnp.tanh((X_crit_PAA - Y[29]))))
    PAA_inhib_P = 0.5 * (1 + (jnp.tanh((-P_crit_PAA + Y[29]))))
    pH = -jnp.log10(Y[6])
    mu_h = jnp.exp((B_1 + B_2 * pH + B_3 * Y[7] + B_4 * (pH ** 2)) + B_5 * (Y[7] ** 2))

    P_inhib = 2.5 * P_std_dev * ((P_std_dev * 2.5066282746310002) ** -1 * jnp.exp(-0.5 * ((s - mean_P) / P_std_dev) ** 2))
    mu_a0 = ratio_mu_e_mu_b * mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X
    mu_e = mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X
    K_diff = 0.75 - (A_t1 * beta_1)
    K_diff = jnp.where(K_diff < K_diff_L, K_diff_L, K_diff)
    r_b0 = mu_a0 * a_1 * s / (K_b + s)
    r_sb0 = Y_sX * r_b0
    r_e1 = (mu_e * a_0 * s) / (K_e + s)
    r_se1 = Y_sX * r_e1
    r_d1 = mu_diff * a_0 / (K_diff + s)
    r_m0 = m_s * a_0 / (K_diff + s)

    v_2 = 0 + Y[26]                                     # sum(phi), same order as the CPU code
    for j in range(9):
        v_2 = v_2 + 4.1887902047863905 * (1.5e-4 + j * delta_r) ** 3 * Y[16 + j] * delta_r
    rho_a1 = (a_1 / ((a_1 / rho_a0) + v_2))
    v_a1 = a_1 / (2 * rho_a1) - v_2
    r_p = mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P - mu_h * Y[3]
    r_m1 = (m_s * rho_a0 * v_a1 * s) / (K_v + s)
    r_d4 = mu_a * a_3
    dn0_dt = ((mu_v * v_a1) / (K_v + s)) * (1.909859317102744 * ((r_0 + delta_0) ** -3)) - k_v * Y[15]
    dn = [-k_v * ((Y[n + 1] - Y[n - 1]) / (2 * delta_r)) + D * (Y[n + 1] - 2 * Y[n] + Y[n - 1]) / delta_r ** 2
          for n in range(16, 25)]                        # dn1 .. dn9
    r_k = r_0 + 8 * delta_r
    r_m = (r_0 + 10 * delta_r)
    dn_m_dt = k_v * dn[8] / (r_m - r_k) - mu_a * Y[25]
    n_k = Y[24]
    dphi_0_dt = ((mu_v * v_a1) / (K_v + s)) - k_v * Y[15] * (PI * (r_0 + delta_0) ** 3) / 6

    F_evp = Y[4] * alpha_evp * (jnp.exp(2.5 * (Y[7] - T0) / (Tv - T0)) - 1)
    pho_feed = (c_s / 1000 * pho_g + (1 - c_s / 1000) * pho_w)
    dilution = Fs + Fb + Fa + Fw - F_evp + Fpaa
    dV1 = Fs + Fb + Fa + Fw + F_discharge / (pho_b / 1000) - F_evp + Fpaa
    dWt = Fs * pho_feed / 1000 + pho_oil / 1000 * Foil + Fb + Fa + Fw + F_discharge - F_evp + Fpaa * pho_paa / 1000
    da_0_dt = r_b0 - r_d1 - Y[11] * dilution / Y[4]
    da_1_dt = r_e1 - r_b0 + r_d1 - (PI * ((r_k + r_m) ** 3) / 6) * rho_d * k_v * n_k - Y[12] * dilution / Y[4]
    da_3_dt = (PI * ((r_k + r_m) ** 3) / 6) * rho_d * k_v * n_k - r_d4 - Y[13] * dilution / Y[4]
    da_4_dt = r_d4 - Y[14] * dilution / Y[4]
    dP_dt = r_p - Y[3] * dilution / Y[4]
    X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt
    X_t = Y[11] + Y[12] + Y[13] + Y[14]
    Qrxn_t = X_1 * Y_QX * Y[4] * Y_O2_X / 1000 + dP_dt * Y_QX * Y[4] * Y_O2_P / 1000
    Qrxn_t = jnp.where(Qrxn_t < 0, 0.0, Qrxn_t)
    N = RPM / 60
    D_imp = 2 * r_imp
    unaerated_power = (n_imp * Po * pho_b * (N ** 3) * (D_imp ** 5))
    P_g = 0.706 * (((unaerated_power ** 2) * N * D_imp ** 3) / (Fg ** 0.56)) ** 0.45
    P_n = P_g / unaerated_power
    variable_power = (n_imp * Po * pho_b * (N ** 3) * (D_imp ** 5) * P_n) / 1000

    d0 = -r_se1 - r_sb0 - r_m0 - r_m1 - (Y_sP * mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P) \
        + Fs * c_s / Y[4] + Foil * c_oil / Y[4] - Y[0] * dilution / Y[4]
    V_s = Fg / (PI * r ** 2)
    T, V, V_m = Y[7], Y[4], Y[4] / 1000
    P_air = ((V_s * R * T * V_m / (22.4 * h_b)) * jnp.log(1 + pho_b * 9.81 * h_b / (pressure_top * 1e5)))
    P_t1 = (variable_power + P_air)
    viscosity = jnp.where(viscosity <= 4, 1.0, viscosity)
    vis_scaled = viscosity / 100
    oil_f = Foil / V
    kla = alpha_kla * ((V_s ** ka) * ((P_t1 / V_m) ** kb) * vis_scaled ** kc) * (1 - oil_f ** kd)
    OUR = -X_1 * Y_O2_X - m_O2_X * X_t - dP_dt * Y_O2_P
    OTR = kla * (DOstar_tp - Y[1])
    d1 = OUR + OTR - (Y[1] * dilution / Y[4])
    Vg = epsilon * V_m
    Qfg_in = 85714.28571428572 * Fg
    Qfg_out = Fg * (N2_in / (1 - Y[2] - Y[27] / 100)) * 85714.28571428572
    d2 = (Qfg_in * O_2_in - Qfg_out * Y[2] - 0.06 * OTR * V_m) / (Vg * 1293.3035714285716)
    d3 = r_p - Y[3] * dilution / Y[4]

    pH_dis = Fs + Foil + Fb + Fa + F_discharge + Fw
    bal1 = ~(-jnp.log10(Y[6]) < 7)                     # pH_balance == 1
    cb = jnp.where(bal1, abc, -abc)
    ca = jnp.where(bal1, -abc, abc)
    x6 = jnp.where(bal1, 1e-14 / Y[6] - Y[6], Y[6])
    Bq = -(x6 * Y[4] + ca * Fa * step1 + cb * Fb * step1) / (Y[4] + Fb * step1 + Fa * step1)
    growth = (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X)
    d6 = jnp.where(bal1,
                   -gamma1 * growth - gamma1 * r_p - gamma2 * pH_dis + ((-Bq - (Bq ** 2 + 4e-14) ** .5) / 2 - x6),
                   gamma1 * growth + gamma1 * r_p + gamma2 * pH_dis + ((-Bq + (Bq ** 2 + 4e-14) ** .5) / 2 - x6))

    Ws = P_t1
    Qcon = U_jacket * A_c * (Y[7] - Tair)
    dQ_dt = Fs * pho_feed * C_ps * (Tf - Y[7]) / 1000 + Fw * pho_w * C_pw * (Tw - Y[7]) / 1000 \
        - F_evp * pho_b * C_pw / 1000 - dealta_H_evap * F_evp * pho_w / 1000 + Qrxn_t + Ws \
        - (alpha_1 / 1000) * Fc ** (beta_T + 1) * ((Y[7] - Tcin) / (Fc / 1000 + (alpha_1 * (Fc / 1000) ** beta_T) / 2 * pho_b * C_ps)) \
        - (alpha_1 / 1000) * Fh ** (beta_T + 1) * ((Y[7] - Th) / (Fh / 1000 + (alpha_1 * (Fh / 1000) ** beta_T) / 2 * pho_b * C_ps)) \
        - Qcon
    d7 = dQ_dt / ((Y[4] / 1000) * C_pw * pho_b)
    d9 = 3 * (a_0 ** (1 / 3)) * (1 / (1 + jnp.exp(-k1 * (t - t1c)))) * (1 / (1 + jnp.exp(-k2 * (t - t2c)))) - k3 * Fw
    d10 = Y[11] + Y[12] + Y[13] + Y[14]
    CER = (Y[11] + Y[12]) * q_co2 * V
    d27 = (117857.14285714287 * Fg * C_CO2_in + CER - 117857.14285714287 * Fg * Y[27]) / (Vg * 1293.3035714285716)
    Henrys_c_co2 = (jnp.exp(11.25 - 395.9 / (Y[7] - 175.9))) / 4400
    C_star_CO2 = (total_pressure * Y[27]) / Henrys_c_co2
    d28 = kla * delta_c_0 * (C_star_CO2 - Y[28]) - Y[28] * dilution / Y[4]
    d29 = Fpaa * PAA_c / V - Y_PAA_P * dP_dt - Y_PAA_X * X_1 - m_PAA * Y[3] - Y[29] * dilution / Y[4]
    X_C_nitrogen = (-r_b0 - r_e1 - r_d1 - r_d4) * Y_NX
    P_C_nitrogen = -dP_dt * Y_NP
    d30 = (NH3_shots * N_conc_shot) / Y[4] + X_C_nitrogen + P_C_nitrogen - m_N * total_X \
        + (1 * N_conc_paa * Fpaa / Y[4]) + N_conc_oil * Foil / Y[4] - Y[30] * dilution / Y[4]
    out = [d0, d1, d2, d3, dV1, dWt, d6, d7, dQ_dt, d9, d10, da_0_dt, da_1_dt, da_3_dt, da_4_dt,
           dn0_dt, *dn, dn_m_dt, dphi_0_dt, d27, d28, d29, d30, mu_p, mu_e]
    return jnp.stack([jnp.broadcast_to(o, Y[0].shape) for o in out], axis=1)


def _rk45(y0, par, t0, t1):
    """scipy RK45 (rtol = atol = 1e-6, first_step = H_ODE) from t0 to t1 for every batch."""
    n = y0.shape[0]
    f0 = rhs(jnp.full(n, t0), y0, par)

    def cond(c):
        return (~jnp.all(c["done"])) & (c["it"] < MAX_RK_ITERS)

    def body(c):
        t, y, f, h = c["t"], c["y"], c["f"], c["h"]
        t_new = jnp.minimum(t + h, t1)
        hh = t_new - t
        K = [f]
        for s in range(1, 6):
            dy = sum(K[j] * _A[s][j] for j in range(s)) * hh[:, None]
            K.append(rhs(t + _C[s] * hh, y + dy, par))
        y_new = y + hh[:, None] * sum(K[j] * _B[j] for j in range(6))
        f_new = rhs(t_new, y_new, par)
        K.append(f_new)
        scale = ATOL + jnp.maximum(jnp.abs(y), jnp.abs(y_new)) * RTOL
        err = sum(K[j] * _E[j] for j in range(7)) * hh[:, None] / scale
        err_norm = jnp.sqrt(jnp.mean(err ** 2, axis=1))
        ok = (err_norm < 1) & ~c["done"]
        grow = jnp.where(err_norm == 0, 10.0, jnp.minimum(10.0, 0.9 * err_norm ** -0.2))
        grow = jnp.where(c["rej"], jnp.minimum(1.0, grow), grow)
        shrink = jnp.where(jnp.isnan(err_norm), 0.2, jnp.maximum(0.2, 0.9 * err_norm ** -0.2))
        return {"t": jnp.where(ok, t_new, t), "y": jnp.where(ok[:, None], y_new, y),
                "f": jnp.where(ok[:, None], f_new, f),
                "h": jnp.where(c["done"], h, jnp.where(ok, hh * grow, hh * shrink)),
                "rej": jnp.where(ok, False, jnp.where(c["done"], c["rej"], True)),
                "done": c["done"] | (ok & (t_new >= t1)), "it": c["it"] + 1}

    c0 = {"t": jnp.full(n, t0), "y": y0, "f": f0, "h": jnp.full(n, H_ODE), "rej": jnp.zeros(n, bool),
          "done": jnp.zeros(n, bool), "it": jnp.array(0)}
    return lax.while_loop(cond, body, c0)["y"]


def _pid(uk1, ek, ek1, yk, yk1, yk2, u_min, u_max, Kp, Ti, Td):
    """pensimpy.utils.pid_controller (Ti > 1e-7 and Td > 0.001 for every loop used here)."""
    uu = uk1 + Kp * ((ek - ek1) + ek * H / Ti + (-Td / H * (yk - 2 * yk1 + yk2)))
    uu = jnp.where(uu > u_max, u_max, uu)
    return jnp.where(uu < u_min, u_min, uu)


def _col(a, i):
    return lax.dynamic_index_in_dim(a, i, axis=1, keepdims=False)


def _step(k, st, mult, level_control):
    """One 12-min step for all batches (PenSimEnv.step + integrate_control_strategy)."""
    hist = st["hist"]
    e, e1, e2 = jnp.maximum(k - 2, 0), jnp.maximum(k - 3, 0), jnp.maximum(k - 4, 0)
    ph_of = lambda h: -jnp.log(h) / jnp.log(10.0)          # noqa: E731
    ph, ph1, ph2 = ph_of(_col(hist["H"], e)), ph_of(_col(hist["H"], e1)), ph_of(_col(hist["H"], e2))
    ph_err = 6.5 - ph
    ph_err1 = jnp.where(k <= 2, 6.5 - ph_of(_col(hist["H"], 0)), -ph1)   # original omits the set-point
    base = ph_err >= -0.05
    Fb = jnp.where(base, _pid(st["Fb"], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 4.0e-05, 8), st["Fb"] * 0.5)
    Fa = jnp.where(base, 0.0, _pid(st["Fa"], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 12.5, 0.125))

    T0_, T1_, T2_ = _col(hist["T"], e), _col(hist["T"], e1), _col(hist["T"], e2)
    terr, terr1 = 298 - T0_, 298 - T1_
    cold = terr <= 0.05
    Fc = jnp.where(cold, _pid(st["Fc"], terr, terr1, T0_, T1_, T2_, 0, 1.5e3, -300, 1.6, 0.005), st["Fc"] * 0.3)
    Fh = jnp.where(cold, st["Fh"] * 0.1, _pid(st["Fc"], terr, terr1, T0_, T1_, T2_, 0, 1.5e3, 50, 0.050, 1))
    Fc = jnp.where(Fc < 1e-4, 1e-4, Fc)
    Fh = jnp.where(Fh < 1e-4, 1e-4, Fh)

    rec = lax.dynamic_index_in_dim(REC, k - 1, axis=0, keepdims=False)
    Fs, Foil, Fg = rec[0] * mult[:, 0], rec[1] * mult[:, 1], rec[2] * mult[:, 2]
    pressure, dis_in, Fw, Fpaa = rec[3], rec[4], rec[5], rec[6] * mult[:, 3]
    ys = st["y"]
    lc = st["lc"]
    if level_control:
        lc = jnp.where(ys[:, 4] > V_HIGH, True, jnp.where(ys[:, 4] < V_LOW, False, lc))
        dis_in = dis_in + jnp.where(lc, LC_RATE, 0.0)
    dis_in = jnp.broadcast_to(dis_in, Fs.shape)
    discharge = -dis_in

    # growth-rate reduction after a long decline of mu_X (PenSimEnv.step, Inhib flag 2)
    win = lax.dynamic_slice_in_dim(hist["muX"], jnp.maximum(k - 66, 0), 65, axis=1)
    trig = (k > 65) & (jnp.sum(jnp.diff(win, axis=1) < 0, axis=1) >= 63)
    mux = jnp.where(trig, _col(hist["muX"], jnp.maximum(k - 2, 0)) * 5, st["mux"])

    n = Fs.shape[0]
    one = jnp.ones(n)
    dist = lax.dynamic_index_in_dim(st["dist"], k - 1, axis=1, keepdims=False)
    par = jnp.stack([st["mup"], mux, st["kla"], st["ncpaa"], st["paac"], 2 * one, Fs, Fg * one, 100 * one, Fc, Fh,
                     Fb, Fa, H_ODE * one, Fw * one, pressure * one, 4 * one, discharge, Fpaa, Foil, 0 * one, one,
                     *[dist[:, i] for i in range(8)], 0 * one], axis=1)
    kf = k.astype(jnp.float64)
    y_sol = _rk45(ys, par, (kf - 1) * H, kf * H + H_ODE)

    y31 = jnp.where(y_sol[:, :31] <= 0, 0.001, y_sol[:, :31])
    y31 = y31.at[:, 1].set(jnp.where(y31[:, 1] < 2, 1.0, y31[:, 1]))
    y31 = y31.at[:, 4].set(jnp.nan_to_num(y31[:, 4])).at[:, 3].set(jnp.nan_to_num(y31[:, 3]))
    V, P, O2, CO2 = y31[:, 4], y31[:, 3], y31[:, 2], y31[:, 27]
    O2_in = 0.204
    OUR = (1.4285714285714286 * Fg) * (O2_in - O2 * (0.7902 / (1 - O2 - CO2 / 100)))
    CER = (1.9642857142857144 * Fg) * ((0.0065 * CO2) * (0.7902 / (1 - O2_in - CO2 / 100) - 0.0330))
    peni = V * P / 1000
    total = st["total"] + (peni - st["ypre"] - discharge * P * H / 1000)

    rec_vals = {"H": y31[:, 6], "T": y31[:, 7], "V": V, "Wt": y31[:, 5], "DO2": y31[:, 1], "O2": O2, "CO2": CO2,
                "OUR": OUR * one, "CER": CER * one, "P": P, "X": y31[:, 11] + y31[:, 12] + y31[:, 13] + y31[:, 14],
                "PAA": y31[:, 29], "NH3": y31[:, 30], "Visc": y31[:, 9], "Fs": Fs, "Foil": Foil, "Fg": Fg * one, "dis": dis_in,
                "muX": y_sol[:, 32]}
    hist = {key: lax.dynamic_update_index_in_dim(hist[key], rec_vals[key], k - 1, axis=1) for key in HIST_KEYS}
    return {**st, "y": jnp.concatenate([y31, jnp.zeros((n, 2))], axis=1), "Fa": Fa, "Fb": Fb, "Fc": Fc,
            "Fh": Fh, "mux": mux, "lc": lc, "ypre": peni, "total": total, "hist": hist}


@partial(jax.jit, static_argnames=("level_control",))
def _run(st, k_from, k_to, mult, level_control):
    return lax.fori_loop(k_from + 1, k_to + 1, lambda k, s: _step(k, s, mult, level_control), st)


def _seed_init(seed):
    """Initial state, parameters and disturbances of one batch, straight from PenSimPy."""
    env = pes.PenSimEnv(recipe_combo=_dummy_combo(), fast=True)
    env.ctrl_flags.Raman_spec = 0
    env.random_seed_ref = seed
    env.reset()
    x0, prm, xi = env.x0, env.param_list, env.xinterp
    y0 = [x0.S, x0.DO2, x0.O2, x0.P, x0.V, x0.Wt, x0.pH, x0.T, 0, 4, x0.Culture_age, x0.a0, x0.a1, x0.a3,
          x0.a4] + [0] * 12 + [x0.CO2outgas, 0, x0.PAA, x0.NH3, 0, 0]
    dist = np.stack([np.asarray(getattr(xi, nm).y, dtype=float) for nm in
                     ["distMuP", "distMuX", "distcs", "distcoil", "distabc", "distPAA", "distTcin", "distO_2in"]],
                    axis=1)
    return np.array(y0, dtype=float), np.array(prm, dtype=float), dist, x0.pH, x0.T


class BatchSet:
    """N batches advanced together; mirrors sim.Batch but vectorised."""

    def __init__(self, seeds, level_control=False, _state=None, _k=0):
        self.seeds = np.asarray(seeds)
        self.level_control = level_control
        self.k = _k
        if _state is None:
            inits = [_seed_init(int(s)) for s in self.seeds]
            y0 = np.stack([i[0] for i in inits])
            prm = np.stack([i[1] for i in inits])
            n = len(inits)
            hist = {key: np.zeros((n, NUM_STEPS)) for key in HIST_KEYS}
            hist["H"][:, 0] = [i[3] for i in inits]      # controllers read the initial pH/T at k = 1
            hist["T"][:, 0] = [i[4] for i in inits]
            _state = {"y": y0, "Fa": np.zeros(n), "Fb": np.zeros(n), "Fc": np.zeros(n), "Fh": np.zeros(n),
                      "mup": prm[:, 0], "mux": prm[:, 1], "kla": prm[:, 2], "ncpaa": prm[:, 3], "paac": prm[:, 4],
                      "dist": np.stack([i[2] for i in inits]), "lc": np.zeros(n, bool),
                      "ypre": np.zeros(n), "total": np.zeros(n), "hist": hist}
            _state = jax.tree_util.tree_map(jnp.asarray, _state)
        self.state = _state

    def __len__(self):
        return len(self.seeds)

    @property
    def total_yield(self):
        return np.asarray(self.state["total"])

    def run_until(self, t_hours, mult=None):
        """Advance to t_hours. `mult` [N,4] multiplies the recipe Fs, Foil, Fg, Fpaa."""
        k_to = min(NUM_STEPS, int(round(t_hours / H)))
        if k_to <= self.k:
            return
        mult = jnp.ones((len(self), 4)) if mult is None else jnp.asarray(mult, dtype=jnp.float64)
        self.state = _run(self.state, jnp.asarray(self.k), jnp.asarray(k_to), mult, self.level_control)
        self.k = k_to

    def fork(self, idx):
        idx = np.asarray(idx)
        st = jax.tree_util.tree_map(lambda a: a[idx], self.state)
        return BatchSet(self.seeds[idx], self.level_control, _state=st, _k=self.k)

    def history(self, key):
        return np.asarray(self.state["hist"][key])

    def usage_since(self, k0):
        """Max volume, sugar and oil fed, broth discharged (L) and aeration used (Fg x h,
        in the simulator's Fg units) over steps k0+1 .. k."""
        sl = slice(k0, self.k)
        return {"vmax": self.history("V")[:, sl].max(1), "sugar_L": self.history("Fs")[:, sl].sum(1) * H,
                "oil_L": self.history("Foil")[:, sl].sum(1) * H,
                "discharged_L": self.history("dis")[:, sl].sum(1) * H,
                "air": self.history("Fg")[:, sl].sum(1) * H}
