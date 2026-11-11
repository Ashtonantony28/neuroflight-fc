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
        self.nis_lpf = torch.zeros(n)   # nf::TranslationKf::nis() (supervisor ESTIMATOR: > 12 for 200 ms)
        rc_n = torch.tensor(1.0, dtype=F32) / (torch.tensor(2.0, dtype=F32) * torch.tensor(3.14159265358979, dtype=F32)
                                              * torch.tensor(2.0, dtype=F32))
        dtt = torch.tensor(self.dt, dtype=F32)
        self.nis_k = dtt / (rc_n + dtt)
        self.tick = 0

    def reset(self, idx, p, v, acc_b) -> None:
        self.p[idx], self.v[idx] = p, v
        self.P00[idx], self.P01[idx], self.P11[idx] = 0.01, 0.0, 0.01
        self.acc[idx] = acc_b
        # firmware reset() sets the ages to 1e3 (invalid until the first update);
        # the sim resets in flight, as if the sensors had just been updated
        self.height_age[idx] = 0.0
        self.flow_age[idx] = 0.0
        self.nis_lpf[idx] = 0.0

    def filter_accel(self, accel_b: torch.Tensor) -> torch.Tensor:
        """The firmware's 40 Hz PT1 accel filter (ControlLoop acc_lpf_); also the Mahony input."""
        self.acc = self.acc + self.acc_k * (accel_b - self.acc)
        return self.acc

    def valid(self) -> torch.Tensor:
        """height_valid() && flow_valid() (VehicleState::pos_valid)."""
        return (self.height_age < 0.2) & (self.flow_age < 0.2)

    def predict(self, accel_b: torch.Tensor, R: torch.Tensor) -> None:
        self.filter_accel(accel_b)
        self.predict_filtered(R)

    def predict_filtered(self, R: torch.Tensor) -> None:
        """Propagate with the already filtered accel (`filter_accel`)."""
        dt = self.dt
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
        nis = y * y / s
        self.nis_lpf = torch.where(mask, self.nis_lpf + self.nis_k * (nis - self.nis_lpf), self.nis_lpf)
        ok = mask & (nis <= self.gate)
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


class Pt1:
    """Batched nf::Pt1 (first-order low-pass, k = dt / (rc + dt), float32)."""

    def __init__(self, shape, cutoff_hz: float, fs: float):
        rc = torch.tensor(1.0, dtype=F32) / (torch.tensor(2.0, dtype=F32) * torch.tensor(3.14159265358979, dtype=F32)
                                            * torch.tensor(cutoff_hz, dtype=F32))
        dt = torch.tensor(1.0, dtype=F32) / torch.tensor(fs, dtype=F32)
        self.k = dt / (rc + dt)
        self.y = torch.zeros(shape)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        self.y = self.y + self.k * (x.to(F32) - self.y)
        return self.y


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """nf::Quat::to_mat (B->W), same op order. q (N,4) w,x,y,z -> (N,3,3)."""
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    r = torch.stack((
        ww + xx - yy - zz, 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), ww - xx + yy - zz, 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), ww - xx - yy + zz), -1)
    return r.reshape(q.shape[0], 3, 3)


def _norm3(v: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), -1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w, x, y, z = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((w * bw - x * bx - y * by - z * bz, w * bx + x * bw + y * bz - z * by,
                        w * by - x * bz + y * bw + z * bx, w * bz + x * by - y * bx + z * bw), -1)


def _quat_normalized(q: torch.Tensor) -> torch.Tensor:
    n = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    ok = (n > 1e-12).unsqueeze(-1)
    ident = torch.zeros_like(q)
    ident[:, 0] = 1.0
    return torch.where(ok, q / n.clamp_min(1e-30).unsqueeze(-1), ident)


def quat_integrated(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """nf::Quat::integrated: q * exp(0.5 omega dt), normalized."""
    dtt = torch.tensor(dt, dtype=F32)
    n = _norm3(omega)
    ang = n * dtt
    big = ang > 1e-9
    a = torch.where((n > 1e-12).unsqueeze(-1), omega / n.clamp_min(1e-30).unsqueeze(-1), torch.zeros_like(omega))
    half = 0.5 * ang
    s = torch.sin(half)
    dq_big = torch.cat((torch.cos(half).unsqueeze(-1), a * s.unsqueeze(-1)), -1)
    dq_small = torch.cat((torch.ones_like(n).unsqueeze(-1), 0.5 * omega * dtt), -1)
    dq = torch.where(big.unsqueeze(-1), dq_big, dq_small)
    return _quat_normalized(_quat_mul(q, dq))


class Mahony:
    """Batched port of nf::Mahony (default Config; fw/core/src/estimator.cpp),
    float32, same op order: accel correction (and bias integral) only while
    |gyro - bias| < max_rate_for_accel (2 rad/s) and |a| in [0.5, 1.5] g;
    `innov` = 1 Hz low-pass of the angle between the measured accel and the
    predicted direction (the supervisor's ESTIMATOR input, > 0.35 rad for
    200 ms trips it). With `sf_ref` (world-frame expected specific force, the
    firmware's KF acceleration + g reference, FOLLOWUPS 11) the prediction is
    normalized(R^T sf_ref) instead of R^T e_z.

    Input gyro is the firmware's filtered gyro (GyroFilterChain output, NOT
    bias-corrected); accel the 40 Hz PT1-filtered accelerometer. Parity with
    the C++ is tested in test/sitl (`nf_mahony_*`)."""

    def __init__(self, n: int, kp: float = 1.0, ki: float = 0.05, kp_init: float = 10.0, init_time_s: float = 1.0,
                 accel_min_g: float = 0.5, accel_max_g: float = 1.5, max_rate_for_accel: float = 2.0,
                 bias_limit: float = 0.2, gravity: float = 9.81, innov_lpf_hz: float = 1.0):
        self.n = n
        self.kp, self.ki, self.kp_init, self.init_time = kp, ki, kp_init, init_time_s
        self.amin, self.amax, self.max_rate = accel_min_g, accel_max_g, max_rate_for_accel
        self.bias_limit, self.g, self.innov_hz = bias_limit, gravity, innov_lpf_hz
        self.q = torch.zeros(n, 4)
        self.q[:, 0] = 1.0
        self.bias = torch.zeros(n, 3)
        self.innov = torch.zeros(n)
        self.t = torch.zeros(n)
        self._kcache = {}

    def reset(self, idx, q: torch.Tensor, bias: torch.Tensor | None = None, t_s: float = 10.0) -> None:
        """Like nf::Mahony::reset(q, t_s) + set_bias(bias). Default age 10 s:
        a vehicle in flight is long past the boosted-gain init phase."""
        self.q[idx] = _quat_normalized(q.to(F32))
        self.bias[idx] = 0.0 if bias is None else bias.to(F32).clamp(-self.bias_limit, self.bias_limit)
        self.innov[idx] = 0.0
        self.t[idx] = float(t_s)

    def R(self) -> torch.Tensor:
        return _quat_to_mat(self.q)

    def _k(self, dt: float):
        k = self._kcache.get(dt)
        if k is None:
            one, two = torch.tensor(1.0, dtype=F32), torch.tensor(2.0, dtype=F32)
            pi = torch.tensor(3.14159265358979, dtype=F32)
            d = torch.tensor(dt, dtype=F32)
            k = d / (one / (two * pi * torch.tensor(self.innov_hz, dtype=F32)) + d)
            self._kcache[dt] = k
        return k

    def update(self, gyro: torch.Tensor, accel: torch.Tensor, dt: float, sf_ref: torch.Tensor | None = None) -> None:
        dtt = torch.tensor(dt, dtype=F32)
        self.t = self.t + dtt
        omega = gyro.to(F32) - self.bias
        accel = accel.to(F32)
        an = _norm3(accel)
        g = torch.tensor(self.g, dtype=F32)
        init = self.t < self.init_time
        calm = (_norm3(omega) < self.max_rate) | init
        gate = calm & torch.isfinite(an) & (an > self.amin * g) & (an < self.amax * g)
        a = accel / torch.where(gate, an, torch.ones_like(an)).unsqueeze(-1)
        R = _quat_to_mat(self.q)
        v = R[:, 2, :]
        if sf_ref is not None:
            r = sf_ref.to(F32)
            # R^T r: dot(col(i), r)
            vr = torch.stack([R[:, 0, i] * r[:, 0] + R[:, 1, i] * r[:, 1] + R[:, 2, i] * r[:, 2] for i in range(3)], -1)
            vn = _norm3(vr)
            ok = (torch.isfinite(vn) & (vn > 1e-3)).unsqueeze(-1)
            v = torch.where(ok, vr / torch.where(ok.squeeze(-1), vn, torch.ones_like(vn)).unsqueeze(-1), v)
        e = _cross(a, v)
        dav = a[:, 0] * v[:, 0] + a[:, 1] * v[:, 1] + a[:, 2] * v[:, 2]
        innov_inst = torch.atan2(_norm3(e), dav)
        k = self._k(dt)
        self.innov = torch.where(gate, self.innov + k * (innov_inst - self.innov), self.innov)
        kp = torch.where(init, torch.tensor(self.kp_init, dtype=F32), torch.tensor(self.kp, dtype=F32))
        if self.ki > 0.0:
            nb = (self.bias - e * (torch.tensor(self.ki, dtype=F32) * dtt)).clamp(-self.bias_limit, self.bias_limit)
            self.bias = torch.where((gate & ~init).unsqueeze(-1), nb, self.bias)
        omega = torch.where(gate.unsqueeze(-1), omega + e * kp.unsqueeze(-1), omega)
        qn = quat_integrated(self.q, omega, dt)
        self.q = torch.where(torch.isfinite(qn).all(-1, keepdim=True), qn, self.q)


class KfAccelRef:
    """The firmware's Mahony specific-force reference (ControlLoop,
    `mahony_kf_ref`): 5 Hz PT1 of the KF velocity derivative while armed with
    valid height + flow (else the input is 0), horizontal part clamped to
    6 m/s^2, vertical to +-g/2. `value()` is the KF acceleration estimate
    after the soft deadband (ControlLoop::accel_ref, `mahony_ref_deadband`,
    horizontal magnitude and z separately); the Mahony reference is
    value() + [0, 0, g]."""

    def __init__(self, n: int, fs: float = 1000.0, lpf_hz: float = 5.0, max_acc: float = 6.0, gravity: float = 9.81,
                 deadband: float = 0.3):
        self.fs, self.max_acc, self.g, self.db = fs, max_acc, gravity, deadband
        self.dt = torch.tensor(1.0, dtype=F32) / torch.tensor(fs, dtype=F32)
        self.lpf = Pt1((n, 3), lpf_hz, fs)
        self.prev_v = torch.zeros(n, 3)

    def reset(self, idx, v: torch.Tensor) -> None:
        self.prev_v[idx] = v.to(F32)
        self.lpf.y[idx] = 0.0

    def update(self, v: torch.Tensor, valid: torch.Tensor) -> None:
        v = v.to(F32)
        ok = valid & torch.isfinite(v).all(-1) & torch.isfinite(self.prev_v).all(-1)
        a = torch.where(ok.unsqueeze(-1), (v - self.prev_v) / self.dt, torch.zeros_like(v))
        ah = torch.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1])
        big = ah > self.max_acc
        ahs = torch.where(big, ah, torch.ones_like(ah))
        a = torch.stack((torch.where(big, a[:, 0] * self.max_acc / ahs, a[:, 0]),
                         torch.where(big, a[:, 1] * self.max_acc / ahs, a[:, 1]),
                         a[:, 2].clamp(-0.5 * self.g, 0.5 * self.g)), -1)
        a = torch.where(torch.isfinite(a).all(-1, keepdim=True), a, torch.zeros_like(a))
        self.lpf.apply(a)
        self.prev_v = v.clone()

    def value(self) -> torch.Tensor:
        a = self.lpf.y
        if self.db <= 0.0:
            return a.clone()
        db = torch.tensor(self.db, dtype=F32)
        ah = torch.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1])
        sh = torch.where(ah > db, (ah - db) / torch.where(ah > db, ah, torch.ones_like(ah)), torch.zeros_like(ah))
        az = torch.where(a[:, 2].abs() > db, a[:, 2] - torch.sign(a[:, 2]) * db, torch.zeros_like(ah))
        return torch.stack((a[:, 0] * sh, a[:, 1] * sh, az), -1)

    def sf_ref(self) -> torch.Tensor:
        r = self.value()
        r[:, 2] = r[:, 2] + torch.tensor(self.g, dtype=F32)
        return r
