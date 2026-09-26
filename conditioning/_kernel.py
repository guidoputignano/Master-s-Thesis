"""Compiled integrator for model.simulate (numba). Mirrors the numpy implementation step by step;
tests/test_conditioning.py checks that both give the same trajectories."""
import numpy as np

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    njit = None


def _kernel(shear, P, idx, bias, f, prot_grat, coc, preorder, pp1, dt, tau_act, n_single,
            tau_ref, k_relax):
    n_steps, n = shear.shape
    rec = np.zeros((7, n_steps, n))
    (i_tau_x, i_w_x, i_k_ord, i_rho_c, i_kappa, i_beta_j, i_k_x, i_k_col, i_j50, i_q, i_t_p,
     i_k_sp, i_seed, i_tau_j, i_t_j, i_th_a, i_th_c, i_delta_x, i_k_heal, i_k_lim, i_lim, i_lim_coc,
     i_d_grat, i_pi_grat, i_b_topo, i_a_grat, i_f_coc, i_k_rx, i_k_rd, i_k_rt, i_tau_r) = idx
    for j in range(n):
        tau_x = P[i_tau_x, j]; w_x = P[i_w_x, j]; k_ord = P[i_k_ord, j]; rho_c = P[i_rho_c, j]
        kappa = P[i_kappa, j]; beta_j = P[i_beta_j, j]; k_x = P[i_k_x, j]; k_col = P[i_k_col, j]
        qh = P[i_q, j]; q50 = P[i_j50, j] ** qh; t_p = P[i_t_p, j]
        k_sp = P[i_k_sp, j]; seed = 10.0 ** P[i_seed, j]; tau_j = P[i_tau_j, j]; t_j = P[i_t_j, j]
        delta_x = P[i_delta_x, j]; k_heal = P[i_k_heal, j]; k_lim = P[i_k_lim, j]
        lim = (P[i_lim_coc, j] if coc[j] else P[i_lim, j]) + (P[i_d_grat, j] if prot_grat[j] else 0.0)
        b_topo = P[i_b_topo, j]; k_rx = P[i_k_rx, j]; k_rd = P[i_k_rd, j]; k_rt = P[i_k_rt, j]
        tau_r = P[i_tau_r, j]
        prot = P[i_pi_grat, j] if prot_grat[j] else 1.0
        kfac = P[i_f_coc, j] if coc[j] else 1.0
        pre = preorder[j] * P[i_a_grat, j]
        A = pre if pre > 0 else 0.0
        C = -pre if pre < 0 else 0.0
        X = 0.0
        Nn = 1.0 - A - C
        J = 0.0
        D = 0.0
        R = 1.0
        fj = f[j]
        th_a = 45.0 - fj * (45.0 - P[i_th_a, j])
        th_c = 90.0 - th_a if prot_grat[j] else 45.0 + fj * (P[i_th_c, j] - 45.0)
        s_a = abs(np.cos(2.0 * np.deg2rad(th_a)))
        s_c = abs(np.cos(2.0 * np.deg2rad(th_c)))
        prime = 0.0
        prev = 0.0
        for k in range(n_steps):
            tau = shear[k, j]
            dtau = abs(tau - prev)
            J = 1.0 - (1.0 - J) * np.exp(-dtau / tau_j)
            J = J * np.exp(-dt / t_j)
            if pp1[k, j]:
                J = 0.0
            prev = tau
            act = tau * tau / (tau * tau + tau_act * tau_act)
            single = D ** n_single
            pref = (1.0 - single) * act * np.tanh((tau_x - tau) / w_x) + single * act + bias[j] * b_topo
            pp = pref if pref > 0 else 0.0
            pm = -pref if pref < 0 else 0.0
            hj = J ** qh / (J ** qh + q50)
            prime = prime + (pm - prime) * (1.0 - np.exp(-dt / t_p))
            boost = (1.0 + beta_j * J) * (1.0 - D) * kfac
            r_na = k_ord * pp * (1.0 + kappa * A) * boost
            r_nc = k_ord * rho_c * min(pm, prime) * (1.0 + kappa * C) * boost
            r_xa = k_x * pp * (1.0 + kappa * A) * kfac
            r_xc = k_x * pm * (1.0 + kappa * C) * kfac
            spread = k_sp * (seed + X)
            col = k_col * hj
            r_ax = pm * (col + spread)
            r_cx = pp * (col + spread)
            w_hj = col / (col + spread) if col + spread > 0 else 0.0
            r_rel = k_relax * (1.0 - act)
            out_n = r_na + r_nc
            fn = 1.0 - np.exp(-out_n * dt)
            to_a_n = Nn * fn * r_na / out_n if out_n > 0 else 0.0
            to_c_n = Nn * fn * r_nc / out_n if out_n > 0 else 0.0
            out_a = r_ax + r_rel
            fa = 1.0 - np.exp(-out_a * dt)
            a_to_x = A * fa * (r_ax / out_a) if out_a > 0 else 0.0
            a_to_n = A * fa - a_to_x
            out_c = r_cx + r_rel
            fc = 1.0 - np.exp(-out_c * dt)
            c_to_x = C * fc * (r_cx / out_c) if out_c > 0 else 0.0
            c_to_n = C * fc - c_to_x
            out_x = r_xa + r_xc
            fx = 1.0 - np.exp(-out_x * dt)
            x_to_a = X * fx * r_xa / out_x if out_x > 0 else 0.0
            x_to_c = X * fx * r_xc / out_x if out_x > 0 else 0.0
            A = A - a_to_x - a_to_n + to_a_n + x_to_a
            C = C - c_to_x - c_to_n + to_c_n + x_to_c
            X = X + a_to_x + c_to_x - x_to_a - x_to_c
            Nn = 1.0 - A - C - X
            if Nn < 0.0:
                Nn = 0.0
            elif Nn > 1.0:
                Nn = 1.0
            collapse = w_hj * (a_to_x * s_a + c_to_x * s_c) / dt
            mism = Nn + X + (C if pref >= 0 else A)
            ov = tau - lim
            src = prot * delta_x * collapse + k_lim * (ov if ov > 0 else 0.0) / tau_ref * mism
            D = 1.0 - (1.0 - D) * np.exp(-src * dt)
            D = D * np.exp(-k_heal * (1.0 - X) * dt)
            tr = tau - tau_r
            loss = prot * k_rx * collapse + k_rd * D * tau / tau_ref + k_rt * (tr if tr > 0 else 0.0) / tau_ref
            R = R * np.exp(-loss * dt)
            rec[0, k, j] = Nn; rec[1, k, j] = A; rec[2, k, j] = C; rec[3, k, j] = X
            rec[4, k, j] = J; rec[5, k, j] = D; rec[6, k, j] = R
    return rec


kernel = njit(cache=True, fastmath=False)(_kernel) if njit is not None else None
