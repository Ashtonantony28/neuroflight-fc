"""Software-in-the-loop: the real C++ firmware core flies inside the Python sim.

    python fw/sitl/sitl.py               # quick demo: fallback takeoff + hover, prints a summary
    python fw/sitl/report.py             # full scenario suite -> docs/sitl-report.md

Pieces
------
* `Firmware`  - ctypes bindings to fw/sitl/build/libnf_sitl.so (N independent
  nf::ControlLoop instances stepped together; see fw/sitl/nf_sitl.h).
* `FirmwareCtbr` - the firmware rate PID + mixer in isolation (INTERFACES.md §9 parity).
* `SitlRunner` - closed loop: sim `Quadrotor` (batched, nominal or randomized)
  -> sensor models at their rates -> firmware step at 1 kHz -> motor commands
  back into the sim (the sim applies its own per-env latency + motor lag).
* `RcScript` - scripted RC (arm, mode switch to POLICY via re-engage, kill, link loss).
* `Log` - blackbox-like per-step trajectory (numpy; `save_npz`, `save_csv`).

Sensor models (all in sim time, 1 kHz physics):
* IMU A: `Quadrotor.sample_gyro/accel` (per-env noise + gyro bias from the
  randomization config) every step.
* IMU B: true rates/specific force + its own noise and a small independent
  bias (|b| <= 0.01 rad/s), every step.
* ToF range (`range = z / R22`) and optical flow (FlowSample convention in
  fw/core/include/nf/estimator.hpp: flow_x = v_Bx/h - w_y, flow_y = v_By/h + w_x)
  at `flow_hz` (100 Hz). Flow noise = vel_noise_std / h, range noise =
  0.4 * pos_noise_std (<= 2 cm, VL53L1X class).
* Motor frequency for the RPM notch: 400 Hz * motor state.
* Ground contact (added here, the sim library has none): z is clamped to 0,
  vertical velocity zeroed, horizontal velocity damped, rates zeroed, and the
  accelerometer sees the normal force. Touchdowns faster than
  `CRASH_IMPACT_MS` or more tilted than `CRASH_TILT_RAD` count as a crash.
"""
from __future__ import annotations

import ctypes
import csv
import json
import math
import os
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.params import load_nominal, load_randomization  # noqa: E402
from sim.quadrotor import Quadrotor  # noqa: E402

BUILD_DIR = os.path.join(HERE, "build")
LIB_PATH = os.path.join(BUILD_DIR, "libnf_sitl.so")
ABI_VERSION = 1

# ---- layouts (mirror fw/sitl/nf_sitl.h)
IN_GYRO, IN_ACCEL, IN_GYRO_B, IN_ACCEL_B = 0, 3, 6, 9
IN_FLOW_X, IN_FLOW_Y, IN_FLOW_Q, IN_RANGE, IN_MOTOR_HZ, IN_VBAT, IN_THROTTLE, IN_F = 12, 13, 14, 15, 16, 20, 21, 22
(II_B_VALID, II_FLOW_VALID, II_RANGE_VALID, II_RPM_VALID, II_VBAT_VALID, II_LINK_OK, II_ARM, II_KILL,
 II_POLICY_ENABLE, II_REENGAGE, IN_I) = range(11)
OUT_CMD, OUT_Q, OUT_OMEGA, OUT_POS, OUT_VEL, OUT_THRUST, OUT_RATE_SP, OUT_ACTION, OUT_F = 0, 4, 8, 11, 14, 17, 18, 21, 25
(OU_ARMED, OU_MODE, OU_ACTIVE, OU_HISTORY, OU_FIRST, OU_ARM_FAIL, OU_INFER_US, OU_FLAGS, OUT_U) = range(9)

# nf::FlightMode
DISARMED, POLICY, FALLBACK_HOLD, CONTROLLED_DESCENT, MOTOR_STOP = range(5)
MODE_NAMES = ["DISARMED", "POLICY", "FALLBACK_HOLD", "CONTROLLED_DESCENT", "MOTOR_STOP"]
# nf::Fault bits
FAULTS = ["TILT", "RATE", "GEOFENCE", "HEIGHT_FLOOR", "POLICY_NONFINITE", "POLICY_RANGE", "POLICY_SATURATION",
          "POLICY_OSCILLATION", "DEADLINE", "LOOP_JITTER", "IMU_DISAGREE", "ESTIMATOR", "RC_LOSS",
          "POLICY_MISSING", "KILL", "DESCENT_ENVELOPE", "TILT_PREDICTED"]
FAULT = {name: 1 << i for i, name in enumerate(FAULTS)}

CRASH_IMPACT_MS = 1.0   # touchdown vertical speed counted as a crash (m/s)
CRASH_TILT_RAD = 0.6    # touchdown tilt counted as a crash


def fault_names(bits: int) -> list[str]:
    return [n for i, n in enumerate(FAULTS) if bits & (1 << i)]


# =========================================================================== build / load
def _sources():
    srcs = [os.path.join(HERE, f) for f in ("nf_sitl.cpp", "nf_sitl.h", "CMakeLists.txt")]
    core = os.path.join(ROOT, "fw", "core")
    for d in ("src", "include/nf"):
        p = os.path.join(core, d)
        srcs += [os.path.join(p, f) for f in os.listdir(p)]
    return srcs


def build_lib(force: bool = False, quiet: bool = True) -> str:
    """(Re)build libnf_sitl.so with CMake if missing or older than any source."""
    if not force and os.path.exists(LIB_PATH):
        t = os.path.getmtime(LIB_PATH)
        if all(os.path.getmtime(s) <= t for s in _sources()):
            return LIB_PATH
    out = subprocess.DEVNULL if quiet else None
    subprocess.run(["cmake", "-S", HERE, "-B", BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release"], check=True, stdout=out)
    subprocess.run(["cmake", "--build", BUILD_DIR, "-j", str(os.cpu_count() or 2)], check=True, stdout=out)
    return LIB_PATH


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(build_lib())
        vp, i32, u32, u64, f32 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_float
        P = ctypes.POINTER
        L.nf_sitl_abi_version.restype = ctypes.c_int
        L.nf_sitl_create.restype = vp
        L.nf_sitl_create.argtypes = [i32, P(SitlConfig)]
        L.nf_sitl_destroy.argtypes = [vp]
        L.nf_sitl_n_envs.argtypes = [vp]
        L.nf_sitl_n_envs.restype = i32
        L.nf_sitl_set_policy_f32.argtypes = [vp, i32, i32, vp, vp, vp, vp, vp, vp, P(u32)]
        L.nf_sitl_set_policy_f32.restype = i32
        L.nf_sitl_clear_policy.argtypes = [vp, i32]
        L.nf_sitl_poke_policy.argtypes = [vp, i32, i32, i32, i32, f32]
        L.nf_sitl_poke_policy.restype = i32
        L.nf_sitl_policy_forward.argtypes = [vp, i32, vp, vp]
        L.nf_sitl_policy_forward.restype = i32
        L.nf_sitl_set_target.argtypes = [vp, i32, f32, f32, f32, f32]
        L.nf_sitl_set_inference_us.argtypes = [vp, i32, u32]
        L.nf_sitl_step.argtypes = [vp, u64, vp, vp, vp, vp]
        L.nf_sitl_mode_name.argtypes = [i32]
        L.nf_sitl_mode_name.restype = ctypes.c_char_p
        L.nf_sitl_fault_name.argtypes = [u32]
        L.nf_sitl_fault_name.restype = ctypes.c_char_p
        L.nf_ctbr_create.argtypes = [i32, i32]
        L.nf_ctbr_create.restype = vp
        L.nf_ctbr_destroy.argtypes = [vp]
        L.nf_ctbr_reset.argtypes = [vp]
        L.nf_ctbr_step.argtypes = [vp, vp, vp, vp, vp, vp]
        if L.nf_sitl_abi_version() != ABI_VERSION:
            raise RuntimeError("libnf_sitl ABI mismatch; rebuild with build_lib(force=True)")
        _LIB = L
    return _LIB


class SitlConfig(ctypes.Structure):
    _fields_ = [("rate_hz", ctypes.c_int32), ("control_hz", ctypes.c_int32), ("hold_height_m", ctypes.c_float),
                ("rpm_filter", ctypes.c_int32), ("gyro_lpf_hz", ctypes.c_float), ("accel_lpf_hz", ctypes.c_float)]


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


# =========================================================================== policies (*.nfpol.json)
def load_nfpol(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def nfpol_arrays(pol: dict):
    """-> (dims int32, mean, std, W concat, b concat, acts int32), float32."""
    layers = pol["layers"]
    Ws = [np.asarray(L["W"], np.float32) for L in layers]
    bs = [np.asarray(L["b"], np.float32).reshape(-1) for L in layers]
    dims = [Ws[0].shape[1]] + [W.shape[0] for W in Ws]
    acts = [1 if L["act"] == "tanh" else 0 for L in layers]
    for L in layers:
        if L["act"] not in ("tanh", "none"):
            raise ValueError(L["act"])
    return (np.asarray(dims, np.int32), np.ascontiguousarray(pol["obs_mean"], np.float32),
            np.ascontiguousarray(pol["obs_std"], np.float32),
            np.ascontiguousarray(np.concatenate([W.reshape(-1) for W in Ws])),
            np.ascontiguousarray(np.concatenate(bs)), np.asarray(acts, np.int32))


def nfpol_crc32(pol: dict) -> int:
    """CRC rule of fw/core/include/nf/policy.hpp (float model), in Python."""
    c = 0
    c = zlib.crc32(np.asarray(pol["obs_mean"], "<f4").tobytes(), c)
    c = zlib.crc32(np.asarray(pol["obs_std"], "<f4").tobytes(), c)
    for L in pol["layers"]:
        c = zlib.crc32(np.asarray(L["W"], "<f4").tobytes(), c)
        c = zlib.crc32(np.asarray(L["b"], "<f4").tobytes(), c)
    return c


def numpy_forward(pol: dict, obs: np.ndarray) -> np.ndarray:
    x = (np.asarray(obs, np.float32) - np.asarray(pol["obs_mean"], np.float32)) / np.asarray(pol["obs_std"], np.float32)
    for L in pol["layers"]:
        x = x @ np.asarray(L["W"], np.float32).T + np.asarray(L["b"], np.float32)
        if L["act"] == "tanh":
            x = np.tanh(x)
    return x.astype(np.float32)


def _single_layer_policy(W: np.ndarray, b: np.ndarray, meta: dict) -> dict:
    return {"version": 1, "mode": "ctbr", "obs_mean": [0.0] * 19, "obs_std": [1.0] * 19,
            "layers": [{"W": W.astype(np.float32).tolist(), "b": b.astype(np.float32).tolist(), "act": "tanh"}],
            "meta": meta}


def linear_hover_policy(nominal: dict | None = None, kp_z=4.0, kd_z=3.0, kp_xy=1.0, kd_xy=1.6, k_att=6.0,
                        k_yaw=3.0) -> dict:
    """Hand-built single-layer CTBR 'policy' (tanh of a linear map of the §4
    observation) that hovers: small-angle PD position/velocity -> tilt ->
    rate setpoint. Synthetic stand-in so policy-mode / fault tests do not
    depend on training. Obs: 0-2 pos err, 3-5 vel, 6-8 R[:,0], 9-11 R[:,1]."""
    nom = nominal or load_nominal()
    m, g, tmax = nom["mass"], nom["gravity"], nom["thrust_max"]
    rmax = nom["max_body_rate"]
    W = np.zeros((4, 19))
    b = np.zeros(4)
    a0 = m * g / (2 * tmax) - 1
    slope = 1 - a0 * a0                       # d tanh at the hover operating point
    b[0] = math.atanh(a0)
    c = m / (2 * tmax) / slope                # (m/s^2) -> pre-tanh a0
    W[0, 2], W[0, 5] = -kp_z * c, -kd_z * c
    # roll: theta_x = R[2,1] = obs[11]; desired theta_x = (kp e_y + kd v_y)/g
    W[1, 1], W[1, 4], W[1, 11] = k_att * kp_xy / g / rmax[0], k_att * kd_xy / g / rmax[0], -k_att / rmax[0]
    # pitch: theta_y = -R[2,0] = -obs[8]; desired theta_y = -(kp e_x + kd v_x)/g
    W[2, 0], W[2, 3], W[2, 8] = -k_att * kp_xy / g / rmax[1], -k_att * kd_xy / g / rmax[1], k_att / rmax[1]
    # yaw: theta_z = R[1,0] = obs[7]
    W[3, 7] = -k_yaw / rmax[2]
    return _single_layer_policy(W, b, {"name": "linear_hover"})


def calibrated_nominal(runner: "SitlRunner", env: int) -> dict:
    """Nominal dict with this env's TRUE mass / mean thrust_max (for the
    synthetic linear policy only; firmware keeps the nominal parameters)."""
    nom = dict(runner.nom)
    nom["mass"] = float(runner.quad.params.mass[env])
    nom["thrust_max"] = float(runner.quad.params.thrust_max[env].mean())
    return nom


def oscillating_policy(nominal: dict | None = None, gain: float = 1.6) -> dict:
    """linear_hover + a destabilizing feedback of the previous roll action
    (obs[16]): a1[k+1] = tanh(-gain a1[k] + ...) -> period-2 (250 Hz) limit
    cycle of amplitude ~0.9 (never reaches the 0.98 saturation level)."""
    p = linear_hover_policy(nominal)
    W = np.asarray(p["layers"][0]["W"])
    b = np.asarray(p["layers"][0]["b"])
    W[1, 16] = -gain
    b[1] += 0.05   # break the symmetric fixed point at 0
    return _single_layer_policy(W, b, {"name": "oscillating"})


def random_policy(seed: int = 0, hidden=(64, 64), w_std: float = 1.0) -> dict:
    """Untrained random-weight MLP (deliberately bad policy)."""
    rng = np.random.default_rng(seed)
    dims = [19, *hidden, 4]
    layers = []
    for i in range(len(dims) - 1):
        W = rng.normal(0, w_std, (dims[i + 1], dims[i]))
        layers.append({"W": W.astype(np.float32).tolist(), "b": rng.normal(0, 0.5, dims[i + 1]).astype(np.float32).tolist(),
                       "act": "tanh"})
    return {"version": 1, "mode": "ctbr", "obs_mean": [0.0] * 19, "obs_std": [1.0] * 19, "layers": layers,
            "meta": {"name": f"random_seed{seed}"}}


# =========================================================================== firmware bindings
class Firmware:
    """N firmware instances (nf::ControlLoop) behind the C ABI."""

    def __init__(self, n: int, rate_hz=1000, control_hz=500, hold_height_m=1.0, rpm_filter=True,
                 gyro_lpf_hz=0.0, accel_lpf_hz=0.0):
        self.L = lib()
        self.n = n
        cfg = SitlConfig(rate_hz, control_hz, hold_height_m, int(rpm_filter), gyro_lpf_hz, accel_lpf_hz)
        self.h = self.L.nf_sitl_create(n, ctypes.byref(cfg))
        if not self.h:
            raise RuntimeError("nf_sitl_create failed")
        self.in_f = np.zeros((n, IN_F), np.float32)
        self.in_i = np.zeros((n, IN_I), np.int32)
        self.out_f = np.zeros((n, OUT_F), np.float32)
        self.out_u = np.zeros((n, OUT_U), np.uint32)
        self._keep = []

    def close(self):
        if self.h:
            self.L.nf_sitl_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_policy(self, pol: dict, env: int = -1) -> tuple[bool, int]:
        dims, mean, std, W, b, acts = nfpol_arrays(pol)
        crc = ctypes.c_uint32(0)
        ok = self.L.nf_sitl_set_policy_f32(self.h, env, len(dims) - 1, _ptr(dims), _ptr(mean), _ptr(std), _ptr(W),
                                           _ptr(b), _ptr(acts), ctypes.byref(crc))
        return bool(ok), crc.value

    def clear_policy(self, env: int = -1):
        self.L.nf_sitl_clear_policy(self.h, env)

    def poke_policy(self, layer: int, which: str, index: int, value: float, env: int = -1) -> bool:
        return bool(self.L.nf_sitl_poke_policy(self.h, env, layer, 0 if which == "W" else 1, index, value))

    def policy_forward(self, obs: np.ndarray, env: int = 0) -> np.ndarray:
        o = np.ascontiguousarray(obs, np.float32)
        out = np.zeros(4, np.float32)
        if not self.L.nf_sitl_policy_forward(self.h, env, _ptr(o), _ptr(out)):
            raise RuntimeError("no/invalid policy")
        return out

    def set_target(self, pos, yaw=0.0, env: int = -1):
        self.L.nf_sitl_set_target(self.h, env, float(pos[0]), float(pos[1]), float(pos[2]), float(yaw))

    def set_inference_us(self, us: int, env: int = -1):
        self.L.nf_sitl_set_inference_us(self.h, env, int(us))

    def step(self, now_us: int):
        self.L.nf_sitl_step(self.h, int(now_us), _ptr(self.in_f), _ptr(self.in_i), _ptr(self.out_f), _ptr(self.out_u))
        return self.out_f, self.out_u


class FirmwareCtbr:
    """Firmware decode_ctbr -> RatePid -> Mixer, batched (INTERFACES.md §9)."""

    def __init__(self, n: int, rate_hz: int = 1000):
        self.L = lib()
        self.n = n
        self.h = self.L.nf_ctbr_create(n, rate_hz)
        self.cmd = np.zeros((n, 4), np.float32)
        self.sat = np.zeros(n, np.int32)
        self.tau = np.zeros((n, 3), np.float32)

    def __del__(self):
        try:
            self.L.nf_ctbr_destroy(self.h)
        except Exception:
            pass

    def reset(self):
        self.L.nf_ctbr_reset(self.h)

    def step(self, action: np.ndarray, gyro: np.ndarray):
        a = np.ascontiguousarray(action, np.float32)
        g = np.ascontiguousarray(gyro, np.float32)
        self.L.nf_ctbr_step(self.h, _ptr(a), _ptr(g), _ptr(self.cmd), _ptr(self.sat), _ptr(self.tau))
        return self.cmd.copy(), self.sat.astype(bool)


# =========================================================================== RC script
@dataclass
class RcScript:
    """Piecewise-constant RC channels. events: (t_s, {channel: value}).
    Channels: arm, kill, policy_enable, reengage, link_ok, throttle.
    `reengage` set to 1 is automatically released after `pulse_s`."""
    events: list = field(default_factory=list)
    pulse_s: float = 0.05
    arm_retry_s: float = 0.2   # re-flick arm every this many s until armed (0 = never)

    def first_arm_t(self) -> float:
        return min([t for t, ch in self.events if ch.get("arm")], default=0.0)

    def at(self, t: float, **ch):
        self.events.append((t, ch))
        if ch.get("reengage"):
            self.events.append((t + self.pulse_s, {"reengage": 0}))
        self.events.sort(key=lambda e: e[0])
        return self

    def state(self, t: float) -> dict:
        s = {"arm": 0, "kill": 0, "policy_enable": 0, "reengage": 0, "link_ok": 1, "throttle": 0.0}
        for te, ch in self.events:
            if te <= t + 1e-9:
                s.update(ch)
        return s


def standard_script(arm_t=1.5, policy_t: float | None = None, kill_t: float | None = None) -> RcScript:
    """Arm into FALLBACK_HOLD (takes off to the 1 m hold point), optionally
    switch to POLICY (re-engage pulse) and optionally hit kill."""
    s = RcScript().at(arm_t, arm=1, policy_enable=0)
    if policy_t is not None:
        s.at(policy_t, reengage=1)
    if kill_t is not None:
        s.at(kill_t, kill=1)
    return s


# =========================================================================== log
LOG_FIELDS = {  # name -> per-env width
    "pos": 3, "vel": 3, "quat": 4, "omega": 3, "motor": 4, "cmd": 4,
    "est_pos": 3, "est_vel": 3, "est_quat": 4, "est_omega": 3,
    "thrust_sp": 1, "rate_sp": 3, "action": 4, "gyro": 3,
}
LOG_UFIELDS = ["armed", "mode", "active_faults", "fault_history", "first_fault", "arm_fail", "inference_us"]


class Log:
    def __init__(self, steps: int, n: int, dt: float, decimate: int = 1):
        self.dt, self.decimate, self.n = dt, decimate, n
        m = (steps + decimate - 1) // decimate
        self.t = np.zeros(m)
        self.f = {k: np.zeros((m, n, w), np.float32) for k, w in LOG_FIELDS.items()}
        self.u = {k: np.zeros((m, n), np.uint32) for k in LOG_UFIELDS}
        self.k = 0

    def record(self, t, quad: Quadrotor, cmd, out_f, out_u, in_f=None):
        k = self.k
        self.t[k] = t
        F = self.f
        F["pos"][k], F["vel"][k], F["quat"][k] = quad.pos.numpy(), quad.vel.numpy(), quad.quat.numpy()
        F["omega"][k], F["motor"][k], F["cmd"][k] = quad.omega.numpy(), quad.motor.numpy(), cmd
        F["est_pos"][k] = out_f[:, OUT_POS:OUT_POS + 3]
        F["est_vel"][k] = out_f[:, OUT_VEL:OUT_VEL + 3]
        F["est_quat"][k] = out_f[:, OUT_Q:OUT_Q + 4]
        F["est_omega"][k] = out_f[:, OUT_OMEGA:OUT_OMEGA + 3]
        F["thrust_sp"][k, :, 0] = out_f[:, OUT_THRUST]
        F["rate_sp"][k] = out_f[:, OUT_RATE_SP:OUT_RATE_SP + 3]
        F["action"][k] = out_f[:, OUT_ACTION:OUT_ACTION + 4]
        if in_f is not None:
            F["gyro"][k] = in_f[:, IN_GYRO:IN_GYRO + 3]   # IMU A as fed to the firmware
        for j, name in enumerate(LOG_UFIELDS):
            self.u[name][k] = out_u[:, j]
        self.k += 1

    def trim(self):
        self.t = self.t[:self.k]
        self.f = {k: v[:self.k] for k, v in self.f.items()}
        self.u = {k: v[:self.k] for k, v in self.u.items()}
        return self

    def __getitem__(self, name):
        return self.f[name] if name in self.f else self.u[name]

    def save_npz(self, path: str):
        np.savez_compressed(path, t=self.t, **self.f, **self.u)

    def save_csv(self, path: str, env: int = 0):
        cols = ["t"]
        for k, w in LOG_FIELDS.items():
            cols += [f"{k}{i}" if w > 1 else k for i in range(w)]
        cols += LOG_UFIELDS
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(cols)
            for i in range(len(self.t)):
                row = [f"{self.t[i]:.4f}"]
                for k in LOG_FIELDS:
                    row += [f"{v:.6g}" for v in self.f[k][i, env]]
                row += [int(self.u[k][i, env]) for k in LOG_UFIELDS]
                wr.writerow(row)


# =========================================================================== runner
def tilt_of_quat(q: np.ndarray) -> np.ndarray:
    """Angle between body z and world z, from q=[w,x,y,z] (...,4)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r22 = 1 - 2 * (x * x + y * y)
    return np.arccos(np.clip(r22, -1, 1))


class SitlRunner:
    """Closed loop sim <-> firmware, one firmware instance per sim env."""

    def __init__(self, n_envs: int = 1, randomize: bool = False, seed: int = 0, flow_hz: int = 100,
                 imu_b: bool = True, start_pos=(0.0, 0.0, 0.0), nominal: dict | None = None,
                 randomization: dict | None = None, policy: dict | None = None, log_decimate: int = 1,
                 **fw_kwargs):
        self.nom = nominal or load_nominal()
        self.rand = randomization or load_randomization()
        self.n = n_envs
        self.quad = Quadrotor(n_envs, self.nom, self.rand, use_nominal=not randomize, seed=seed, floor_z=-1e9)
        self.dt = self.quad.dt
        self.rate_hz = int(round(1.0 / self.dt))
        self.fw = Firmware(n_envs, rate_hz=self.rate_hz, control_hz=int(self.nom["control_rate_hz"]), **fw_kwargs)
        self.policy = policy
        self.policy_ok = None
        if policy is not None:
            self.policy_ok, self.policy_crc = self.fw.set_policy(policy)
        self.rng = np.random.default_rng(seed + 12345)
        self.flow_div = max(1, int(round(self.rate_hz / flow_hz)))
        self.imu_b = imu_b
        self.imu_b_bias = self.rng.uniform(-0.01, 0.01, (n_envs, 3)).astype(np.float32)
        self.g = float(self.nom["gravity"])
        self.log_decimate = log_decimate
        sp = torch.tensor(start_pos, dtype=torch.float32).expand(n_envs, 3)
        idx = torch.arange(n_envs)
        z4 = torch.zeros(n_envs, 4)
        self.quad.set_state(idx, pos=sp.clone(), vel=torch.zeros(n_envs, 3), quat=torch.tensor([1.0, 0, 0, 0]).expand(n_envs, 4),
                            omega=torch.zeros(n_envs, 3), motor=z4, cmd=z4)
        self.on_ground = self.quad.pos[:, 2].numpy() <= 1e-6
        self._ground_sf()
        self.step_k = 0
        self.t = 0.0
        self.hooks: list[Callable] = []       # hook(runner, t) before each firmware step (fault injection)
        self.crashed = np.zeros(n_envs, bool)
        self.crash_reason = [""] * n_envs
        self.touchdowns: list[list[tuple]] = [[] for _ in range(n_envs)]  # (t, vz, tilt)
        self.cmd = np.zeros((n_envs, 4), np.float32)
        self.t_boot = 0.0                     # firmware time = t_boot + t (power-on period before run())
        self.ever_armed = np.zeros(n_envs, bool)
        self.arm_attempts = np.zeros(n_envs, int)
        self._prev_arm_cmd = np.zeros(n_envs, np.int32)

    # ---------------------------------------------------------------- power-on
    def power_on(self, seconds: float = 60.0):
        """Vehicle sits still on the ground, disarmed, for `seconds` before
        run(): the firmware runs at 1 kHz (Mahony settles, gyro bias is
        learned). Fast path: physics is not integrated (nothing moves), sensors
        are drawn with the same noise models."""
        q, f, i = self.quad, self.fw.in_f, self.fw.in_i
        n, steps = self.n, int(round(seconds / self.dt))
        p = q.params
        bias = p.gyro_bias.numpy()
        gstd = p.gyro_noise_std.numpy()[:, None]
        astd = p.accel_noise_std.numpy()[:, None]
        R = q.R.numpy()
        sf = R[:, 2, :] * self.g
        h = q.pos[:, 2].numpy()
        i[:] = 0
        i[:, II_B_VALID] = int(self.imu_b)
        i[:, II_LINK_OK] = 1
        i[:, II_RPM_VALID] = 1
        f[:] = 0
        for k in range(steps):
            f[:, IN_GYRO:IN_GYRO + 3] = bias + self.rng.standard_normal((n, 3)) * gstd
            f[:, IN_ACCEL:IN_ACCEL + 3] = sf + self.rng.standard_normal((n, 3)) * astd
            f[:, IN_GYRO_B:IN_GYRO_B + 3] = self.imu_b_bias + self.rng.standard_normal((n, 3)) * gstd
            f[:, IN_ACCEL_B:IN_ACCEL_B + 3] = sf + self.rng.standard_normal((n, 3)) * astd
            new = k % self.flow_div == 0
            i[:, II_RANGE_VALID] = int(new)
            f[:, IN_RANGE] = np.maximum(h + self.rng.standard_normal(n) * 0.4 * p.pos_noise_std.numpy(), 0.0)
            self.fw.step(int(round((self.t_boot + k * self.dt) * 1e6)))
        self.t_boot += steps * self.dt

    # ---------------------------------------------------------------- sensors
    def _ground_sf(self):
        g = self.on_ground
        if g.any():
            idx = torch.as_tensor(np.nonzero(g)[0])
            R = self.quad.R[idx]
            self.quad.specific_force_b[idx] = R[:, 2, :] * self.g    # R^T [0,0,g]

    def _sensors(self):
        q, f, i = self.quad, self.fw.in_f, self.fw.in_i
        n = self.n
        f[:, IN_GYRO:IN_GYRO + 3] = q.sample_gyro().numpy()
        f[:, IN_ACCEL:IN_ACCEL + 3] = q.sample_accel().numpy()
        p = q.params
        omega = q.omega.numpy()
        sf = q.specific_force_b.numpy()
        gstd = p.gyro_noise_std.numpy()[:, None]
        astd = p.accel_noise_std.numpy()[:, None]
        f[:, IN_GYRO_B:IN_GYRO_B + 3] = omega + self.imu_b_bias + self.rng.standard_normal((n, 3)) * gstd
        f[:, IN_ACCEL_B:IN_ACCEL_B + 3] = sf + self.rng.standard_normal((n, 3)) * astd
        i[:, II_B_VALID] = int(self.imu_b)
        new = (self.step_k % self.flow_div) == 0
        i[:, II_FLOW_VALID] = 0
        i[:, II_RANGE_VALID] = 0
        if new:
            R = q.R.numpy()
            r22 = R[:, 2, 2]
            pos, vel = q.pos.numpy(), q.vel.numpy()
            ok = r22 > 0.3
            rng_m = np.where(ok, pos[:, 2] / np.maximum(r22, 1e-3), 0.0)
            rng_m = rng_m + self.rng.standard_normal(n) * 0.4 * p.pos_noise_std.numpy()
            rng_m = np.maximum(rng_m, 0.0)
            vb = np.einsum("nji,nj->ni", R, vel)
            hb = np.maximum(rng_m, 1e-3)
            vstd = p.vel_noise_std.numpy()
            fx = vb[:, 0] / hb - omega[:, 1] + self.rng.standard_normal(n) * vstd / hb
            fy = vb[:, 1] / hb + omega[:, 0] + self.rng.standard_normal(n) * vstd / hb
            f[:, IN_RANGE] = rng_m
            f[:, IN_FLOW_X], f[:, IN_FLOW_Y], f[:, IN_FLOW_Q] = fx, fy, 1.0
            i[:, II_RANGE_VALID] = ok
            i[:, II_FLOW_VALID] = ok & (rng_m > 0.05)
        f[:, IN_MOTOR_HZ:IN_MOTOR_HZ + 4] = 400.0 * q.motor.numpy()
        i[:, II_RPM_VALID] = 1

    def _rc(self, script: RcScript):
        s = script.state(self.t)
        i, f = self.fw.in_i, self.fw.in_f
        arm = np.full(self.n, s["arm"], np.int32)
        if script.arm_retry_s > 0 and s["arm"]:
            # Pilot behaviour: arming refused -> flick the arm switch again
            # (low for half a period) until the vehicle arms. Only before the
            # first successful arm (no re-arming after kill / disarm).
            period = max(2, int(round(script.arm_retry_s / self.dt)))
            retry = ~self.ever_armed & (self.step_k % period >= period // 2) & (self.t > script.first_arm_t() + script.arm_retry_s)
            arm[retry] = 0
        self.arm_attempts += (arm == 1) & (self._prev_arm_cmd == 0)
        self._prev_arm_cmd = arm
        i[:, II_ARM], i[:, II_KILL] = arm, s["kill"]
        i[:, II_POLICY_ENABLE], i[:, II_REENGAGE], i[:, II_LINK_OK] = s["policy_enable"], s["reengage"], s["link_ok"]
        f[:, IN_THROTTLE] = s["throttle"]

    # ---------------------------------------------------------------- physics
    def _physics(self, cmd: np.ndarray):
        q = self.quad
        z_prev = q.pos[:, 2].numpy().copy()
        q.step(torch.from_numpy(cmd))
        pos = q.pos.numpy()
        vel = q.vel.numpy()
        below = pos[:, 2] < 0.0
        if not np.isfinite(pos).all():
            bad = ~np.isfinite(pos).all(1)
            for e in np.nonzero(bad & ~self.crashed)[0]:
                self.crash_reason[e] = "non-finite state"
            self.crashed |= bad
        if below.any():
            idx = np.nonzero(below)[0]
            tilt = tilt_of_quat(q.quat.numpy()[idx])
            for j, e in enumerate(idx):
                if z_prev[e] > 1e-6 or not self.on_ground[e]:   # new touchdown
                    vz = float(vel[e, 2])
                    self.touchdowns[e].append((self.t, vz, float(tilt[j])))
                    if (-vz > CRASH_IMPACT_MS or tilt[j] > CRASH_TILT_RAD) and not self.crashed[e]:
                        self.crashed[e] = True
                        self.crash_reason[e] = f"impact vz={vz:.2f} m/s tilt={math.degrees(tilt[j]):.0f} deg at t={self.t:.3f}"
            ti = torch.as_tensor(idx)
            q.pos[ti, 2] = 0.0
            q.vel[ti, 2] = torch.clamp(q.vel[ti, 2], min=0.0)
            q.vel[ti, 0:2] *= 0.9
            q.omega[ti] = 0.0
        self.on_ground = below
        self._ground_sf()

    # ---------------------------------------------------------------- main loop
    def run(self, duration_s: float, script: RcScript, log: Log | None = None,
            stop: Callable | None = None) -> Log:
        steps = int(round(duration_s / self.dt))
        log = log or Log(steps, self.n, self.dt, self.log_decimate)
        for _ in range(steps):
            self._sensors()
            self._rc(script)
            for h in self.hooks:
                h(self, self.t)
            out_f, out_u = self.fw.step(int(round((self.t_boot + self.t) * 1e6)))
            self.ever_armed |= out_u[:, OU_ARMED] == 1
            self.cmd = np.ascontiguousarray(out_f[:, OUT_CMD:OUT_CMD + 4])
            if self.step_k % self.log_decimate == 0:
                log.record(self.t, self.quad, self.cmd, out_f, out_u, self.fw.in_f)
            self._physics(self.cmd)   # sim applies its latency buffer + motor lag
            self.step_k += 1
            self.t = self.step_k * self.dt
            if stop is not None and stop(self):
                break
        return log.trim()


# =========================================================================== demo
def _demo():
    r = SitlRunner(1)
    log = r.run(8.0, standard_script())
    p = log["pos"][:, 0]
    k = log.t > 5
    print(f"final pos {p[-1]}, mode {MODE_NAMES[log['mode'][-1, 0]]}, z err rms (t>5s) "
          f"{np.sqrt(np.mean((p[k, 2] - 1) ** 2)):.4f} m, crashed={r.crashed[0]}")


if __name__ == "__main__":
    _demo()
