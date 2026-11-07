"""Batched torch ports of the firmware's sensing path (fw/core), so training
sees what the policy sees on the vehicle (INTERFACES.md §9-§10).

* `GyroFilterChain` - fw/core ControlLoop step 1: 150 Hz biquad low-pass
  (RBJ, Q=1/sqrt2, direct form II transposed), then the RPM notch bank (one
  notch per motor per axis at the motor rotation frequency, Q=3, bypassed
  below 80 Hz, fresh state on (re)activation, clamped to 0.45 fs). Float32,
  same op order as fw/core/src/filters.cpp; parity with the C++ chain is
  tested in test/sitl (`nf_gfilt_*`). Its output minus the gyro-bias estimate
  is the firmware's `state.omega`: the rate PID measurement AND policy obs
  12-14.
* `TranslationKf` - fw/core TranslationKf: three decoupled [p, v] Kalman
  filters driven at 1 kHz by the 40 Hz PT1-filtered accelerometer rotated to
  W; z corrected by ToF height, x/y by optical-flow velocity at 100 Hz, with
  the same gates. Measurements are generated exactly like fw/sitl/sitl.py
  (`SitlRunner._sensors`): range = z / R22 + N(0, 0.4 pos_noise_std),
  flow = v_B / range - omega + N(0, vel_noise_std / range).

The SITL measurements (fw/sitl/diagnose_gap.py) that motivated this: with
the training-env inputs (true attitude/velocity, white noise) swapped back
into the firmware the v1 policy hovers at 5 cm with 4 deg max tilt; with the
firmware's flow-KF velocity alone it tilts to 22-37 deg and trips
TILT_PREDICTED. The KF velocity error is coloured (autocorrelation 0.66 at
10 ms, 0.41 at 20 ms, ~0 at 50 ms) with rms ~0.6 vel_noise_std, so it has ~5x
the low-frequency power of the white per-step noise the policy was trained on.
"""
from __future__ import annotations

import math

import torch

F32 = torch.float32


def _lowpass_coeffs(fc: float, fs: float, q: float = 0.70710678):
    fc = min(max(fc, 1e-3), 0.49 * fs)
    w0 = torch.tensor(2.0 * math.pi, dtype=F32) * torch.tensor(fc, dtype=F32) / torch.tensor(fs, dtype=F32)
    sn, cs = torch.sin(w0), torch.cos(w0)
    alpha = sn / (2.0 * torch.tensor(q, dtype=F32))
    a0 = 1.0 + alpha
    b0 = (1.0 - cs) * 0.5 / a0
    b1 = (1.0 - cs) / a0
    return b0, b1, b0.clone(), -2.0 * cs / a0, (1.0 - alpha) / a0


def _notch_coeffs(fc: torch.Tensor, fs: float, q: float):
    """fc: tensor (any shape) already clamped. Returns b0, b1, b2, a1, a2."""
    w0 = (2.0 * torch.tensor(math.pi, dtype=F32)) * fc / torch.tensor(fs, dtype=F32)
    sn, cs = torch.sin(w0), torch.cos(w0)
    alpha = sn / (2.0 * torch.tensor(q, dtype=F32))
    a0 = 1.0 + alpha
    b0 = 1.0 / a0
    b1 = -2.0 * cs / a0
    return b0, b1, b0, b1, (1.0 - alpha) / a0


class GyroFilterChain:
    """Batched fw gyro LPF + RPM notch bank. State (N,3) / (N,4,3)."""

    def __init__(self, n: int, fs: float = 1000.0, lpf_hz: float = 150.0, rpm: bool = True,
                 notch_q: float = 3.0, notch_min_hz: float = 80.0):
        self.n, self.fs, self.rpm = n, float(fs), rpm
        self.q, self.min_hz = notch_q, notch_min_hz
        self.lb0, self.lb1, self.lb2, self.la1, self.la2 = _lowpass_coeffs(lpf_hz, fs)
        self.ls1 = torch.zeros(n, 3)
        self.ls2 = torch.zeros(n, 3)
        self.active = torch.zeros(n, 4, dtype=torch.bool)
        init = torch.full((n, 4), float(notch_min_hz))
        self.nb = list(_notch_coeffs(init, fs, notch_q))  # each (N,4)
        self.ns1 = torch.zeros(n, 4, 3)
        self.ns2 = torch.zeros(n, 4, 3)

    def reset(self, idx: torch.Tensor, value: torch.Tensor, motor_hz: torch.Tensor | None = None) -> None:
        """Steady state at constant input `value` (len(idx),3), as if the
        filters had been running (the firmware never restarts them in flight)."""
        v = value.to(F32)
        self.ls2[idx] = (self.lb2 - self.la2) * v
        self.ls1[idx] = (self.lb1 - self.la1) * v + self.ls2[idx]
        if self.rpm and motor_hz is not None:
            f = motor_hz.to(F32)
            ok = torch.isfinite(f) & (f >= self.min_hz)
            fc = f.clamp(self.min_hz, 0.45 * self.fs)
            c = _notch_coeffs(fc, self.fs, self.q)
            for k in range(5):
                self.nb[k][idx] = torch.where(ok, c[k], self.nb[k][idx])
            self.active[idx] = ok
            b0, b1, b2, a1, a2 = (x[idx].unsqueeze(-1) for x in self.nb)
            # cascaded notches all have DC gain 1 -> every stage sees `v`
            vv = v.unsqueeze(1).expand(-1, 4, 3)
            s2 = (b2 - a2) * vv
            s1 = (b1 - a1) * vv + s2
            self.ns1[idx] = torch.where(ok.unsqueeze(-1), s1, torch.zeros_like(s1))
            self.ns2[idx] = torch.where(ok.unsqueeze(-1), s2, torch.zeros_like(s2))
        else:
            self.active[idx] = False
            self.ns1[idx] = 0.0
            self.ns2[idx] = 0.0

    def update_rpm(self, motor_hz: torch.Tensor) -> None:
        """fw RpmNotchBank::update (retune without state reset; reset on activation)."""
        f = motor_hz.to(F32)
        ok = torch.isfinite(f) & (f >= self.min_hz)
        fc = f.clamp(self.min_hz, 0.45 * self.fs)
        c = _notch_coeffs(fc, self.fs, self.q)
        for k in range(5):
            self.nb[k] = torch.where(ok, c[k], self.nb[k])
        fresh = (ok & ~self.active).unsqueeze(-1)
        self.ns1 = torch.where(fresh, torch.zeros_like(self.ns1), self.ns1)
        self.ns2 = torch.where(fresh, torch.zeros_like(self.ns2), self.ns2)
        self.active = ok

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(F32)
        y = self.lb0 * x + self.ls1
        self.ls1 = self.lb1 * x - self.la1 * y + self.ls2
        self.ls2 = self.lb2 * x - self.la2 * y
        if not self.rpm:
            return y
        o = y
        for m in range(4):
            b0, b1, b2, a1, a2 = (c[:, m:m + 1] for c in self.nb)
            act = self.active[:, m:m + 1]
            yy = b0 * o + self.ns1[:, m]
            s1 = b1 * o - a1 * yy + self.ns2[:, m]
            s2 = b2 * o - a2 * yy
            self.ns1[:, m] = torch.where(act, s1, self.ns1[:, m])
            self.ns2[:, m] = torch.where(act, s2, self.ns2[:, m])
            o = torch.where(act, yy, o)
        return o

    def step(self, gyro: torch.Tensor, motor_hz: torch.Tensor | None) -> torch.Tensor:
        """One 1 kHz sample: update notch frequencies, then LPF -> notches."""
        if self.rpm and motor_hz is not None:
            self.update_rpm(motor_hz)
        return self.apply(gyro)


class TranslationKf:
    """Batched port of nf::TranslationKf (default Config) + the 40 Hz PT1 accel
    filter that feeds it, plus the SITL range/flow sensor models."""

    def __init__(self, n: int, fs: float = 1000.0, gravity: float = 9.81, accel_lpf_hz: float = 40.0,
                 flow_div: int = 10, accel_noise: float = 0.5, flow_vel_noise: float = 0.1,
                 range_noise: float = 0.02, min_quality: float = 0.3, min_height_flow: float = 0.08,
                 max_tilt_range: float = 0.7, max_range: float = 4.0, nis_gate: float = 25.0):
        self.n, self.dt, self.g = n, 1.0 / fs, gravity
        rc = 1.0 / (2.0 * math.pi * accel_lpf_hz)
        self.acc_k = self.dt / (rc + self.dt)
        self.flow_div = flow_div
        self.q = accel_noise * accel_noise
        self.r2_flow = flow_vel_noise * flow_vel_noise
        self.r2_range = range_noise * range_noise
        self.min_h, self.max_tilt, self.max_range, self.gate = min_height_flow, max_tilt_range, max_range, nis_gate
        self.min_quality = min_quality
        self.p = torch.zeros(n, 3)
        self.v = torch.zeros(n, 3)
        self.P00 = torch.full((n, 3), 0.01)
        self.P01 = torch.zeros(n, 3)
        self.P11 = torch.full((n, 3), 0.01)
        self.acc = torch.zeros(n, 3)
        self.height_age = torch.zeros(n)
        self.flow_age = torch.zeros(n)
        self.tick = 0

    def reset(self, idx, p, v, acc_b) -> None:
        self.p[idx], self.v[idx] = p, v
        self.P00[idx], self.P01[idx], self.P11[idx] = 0.01, 0.0, 0.01
        self.acc[idx] = acc_b
        self.height_age[idx] = 0.0
        self.flow_age[idx] = 0.0

    def predict(self, accel_b: torch.Tensor, R: torch.Tensor) -> None:
        dt = self.dt
        self.acc = self.acc + self.acc_k * (accel_b - self.acc)
        aw = torch.einsum("nij,nj->ni", R, self.acc)
        aw[:, 2] -= self.g
        aw = torch.where(torch.isfinite(aw), aw, torch.zeros_like(aw))
        self.p = self.p + self.v * dt + 0.5 * aw * dt * dt
        self.v = self.v + aw * dt
        p00, p01, p11, q = self.P00, self.P01, self.P11, self.q
        self.P00 = p00 + dt * (2 * p01) + dt * dt * p11 + q * dt * dt * dt / 3.0
        self.P01 = p01 + dt * p11 + q * dt * dt / 2.0
        self.P11 = p11 + q * dt
        self.height_age = self.height_age + dt
        self.flow_age = self.flow_age + dt

    def _update(self, ax: int, idx: int, z: torch.Tensor, r2: float, mask: torch.Tensor) -> torch.Tensor:
        """Scalar update of axis `ax` with measurement of p (idx 0) or v (idx 1)."""
        p, v = self.p[:, ax], self.v[:, ax]
        P00, P01, P11 = self.P00[:, ax], self.P01[:, ax], self.P11[:, ax]
        hx = p if idx == 0 else v
        y = z - hx
        Pii = P00 if idx == 0 else P11
        s = Pii + r2
        ok = mask & ((y * y / s) <= self.gate)
        P0i = P00 if idx == 0 else P01
        P1i = P01 if idx == 0 else P11
        k0, k1 = P0i / s, P1i / s
        n00 = P00 - k0 * P0i
        n01 = P01 - k0 * P1i
        n11 = P11 - k1 * P1i
        n10 = P01 - k1 * P0i
        n01 = 0.5 * (n01 + n10)
        self.p[:, ax] = torch.where(ok, p + k0 * y, p)
        self.v[:, ax] = torch.where(ok, v + k1 * y, v)
        self.P00[:, ax] = torch.where(ok, n00, P00)
        self.P01[:, ax] = torch.where(ok, n01, P01)
        self.P11[:, ax] = torch.where(ok, n11, P11)
        return ok

    def correct(self, R: torch.Tensor, omega_est: torch.Tensor, true_pos: torch.Tensor, true_vel: torch.Tensor,
                true_R: torch.Tensor, true_omega: torch.Tensor, pos_noise_std: torch.Tensor,
                vel_noise_std: torch.Tensor, gen: torch.Generator) -> None:
        """One 1 kHz tick: on every `flow_div`-th tick generate range + flow
        exactly like SitlRunner._sensors and apply the firmware update."""
        t = self.tick
        self.tick += 1
        if t % self.flow_div != 0:
            return
        n = self.n
        r22_t = true_R[:, 2, 2]
        ok_s = r22_t > 0.3
        rng = torch.where(ok_s, true_pos[:, 2] / r22_t.clamp_min(1e-3), torch.zeros_like(r22_t))
        rng = (rng + torch.randn(n, generator=gen) * 0.4 * pos_noise_std).clamp_min(0.0)
        vb = torch.einsum("nji,nj->ni", true_R, true_vel)
        hb_s = rng.clamp_min(1e-3)
        fx = vb[:, 0] / hb_s - true_omega[:, 1] + torch.randn(n, generator=gen) * vel_noise_std / hb_s
        fy = vb[:, 1] / hb_s + true_omega[:, 0] + torch.randn(n, generator=gen) * vel_noise_std / hb_s
        range_valid = ok_s
        flow_valid = ok_s & (rng > 0.05)
        # ---- firmware TranslationKf::correct
        r22 = R[:, 2, 2]
        tilt = torch.acos(r22.clamp(-1.0, 1.0))
        tilt_ok = tilt < self.max_tilt
        m_r = range_valid & (rng >= 0.0) & (rng < self.max_range) & tilt_ok
        ok_h = self._update(2, 0, rng * r22, self.r2_range, m_r)
        self.height_age = torch.where(ok_h, torch.zeros_like(self.height_age), self.height_age)
        h = self.p[:, 2]
        m_f = flow_valid & (1.0 >= self.min_quality) & (h > self.min_h) & (self.height_age < 0.2) & tilt_ok
        hb = h / r22
        vb_est = torch.einsum("nji,nj->ni", R, self.v)
        vbm = torch.stack((hb * (fx + omega_est[:, 1]), hb * (fy - omega_est[:, 0]), vb_est[:, 2]), -1)
        vw = torch.einsum("nij,nj->ni", R, vbm)
        ok0 = self._update(0, 1, vw[:, 0], self.r2_flow, m_f)
        ok1 = self._update(1, 1, vw[:, 1], self.r2_flow, m_f)
        self.flow_age = torch.where(ok0 & ok1, torch.zeros_like(self.flow_age), self.flow_age)
