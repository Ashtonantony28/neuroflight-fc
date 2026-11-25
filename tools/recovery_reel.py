"""Recovery reel of the v3 neural policy (simulation) for the README.

    python -m tools.recovery_reel [--out docs/media] [--seed 7]

Flies train/policies/hover_ctbr_v3.nfpol.json on the nominal vehicle in the training simulator
with the firmware sensing path (fw3: filtered gyro, flow/ToF KF, firmware Mahony), from the eval
suite's own initial states: a throw, a 60 deg tilt and an upside-down start. The target is (0, 0, 2) m.
The reel is drawn at half speed and is labelled as simulation. It also writes recovery-envelope.png:
position error and tilt over time for 64 randomized vehicles per scenario, v3 vs the PID fallback.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from train import eval as ev

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, GRID, TXT, DIM = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
C_V3, C_PID, C_TGT = "#58a6ff", "#f0883e", "#3fb950"
SCEN = ("throw", "tilt_60", "upside_down")
NAMES = {"throw": "Thrown: 3 m/s, 10 rad/s spin", "tilt_60": "Released at 60° tilt", "upside_down": "Released upside down"}
POLICY = ROOT / "train" / "policies" / "hover_ctbr_v3.nfpol.json"


@torch.no_grad()
def fly(ctrl, scenarios, n, seed, randomized, secs=3.0):
    env, groups, nominal = ev.build(n, 0, seed, False, scenarios, "fw3")
    sel = [g for g in groups if g.mode == ("randomized" if randomized else "nominal")]
    steps = int(round(secs / env.control_dt))
    P, R, A = [], [], []
    obs = env.obs_buf.clone()
    alive = torch.ones(env.n, dtype=torch.bool)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    for _ in range(steps):
        a = ctrl(obs).clamp(-1, 1)
        obs, _, done, info = env.step(a)
        alive &= ~(info["crash"] | info["oob"])
        P.append(env.quad.pos.clone())
        R.append(env.quad.R.clone())
        A.append(a.clone())
    P, R, A = torch.stack(P).numpy(), torch.stack(R).numpy(), torch.stack(A).numpy()
    t = np.arange(1, steps + 1) * env.control_dt
    return t, {g.scenario: (P[:, g.sl], R[:, g.sl], A[:, g.sl], alive[g.sl].numpy()) for g in sel}


def quad_geom(p, Rm, arm=0.24, rotor=0.09):
    """Arms (2 lines) and 4 rotor circles of an X quad at p with attitude Rm (body FLU)."""
    d = arm / math.sqrt(2)
    tips = np.array([[d, d, 0], [-d, -d, 0], [d, -d, 0], [-d, d, 0]])
    w = (Rm @ tips.T).T + p
    th = np.linspace(0, 2 * math.pi, 24)
    circ = np.stack([rotor * np.cos(th), rotor * np.sin(th), np.zeros_like(th)], 1)
    rings = [(Rm @ (circ + tips[i]).T).T + p for i in range(4)]
    up = p + Rm @ np.array([0, 0, 0.18])
    nose = p + Rm @ np.array([0.12, 0, 0])
    return w, rings, up, nose


def reel(t, runs, out, slow=0.5, fps=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TXT, "axes.labelcolor": DIM,
                         "xtick.color": DIM, "ytick.color": DIM})
    tgt = np.array(ev.TARGET)
    dt = t[1] - t[0]
    T = t[-1]
    nfr = int(T / slow * fps)
    tmp = Path(tempfile.mkdtemp())
    fig = plt.figure(figsize=(12.8, 5.6), dpi=100, facecolor=BG)
    axes = [fig.add_axes([0.01 + 0.33 * i, 0.06, 0.32, 0.74], projection="3d") for i in range(3)]
    fig.text(0.02, 0.93, "neuroflight-fc v3 policy: recovery from bad starts", fontsize=17, weight="bold", color=TXT)
    fig.text(0.02, 0.885, "SIMULATION  ·  nominal 3\" quad, firmware sensing in the loop (filtered gyro, flow/ToF KF, "
             "firmware Mahony)  ·  target ● (0, 0, 2) m  ·  played at half speed", fontsize=9.5, color=DIM)
    arts = []
    for ax, sc in zip(axes, SCEN):
        P, R, A, ok = runs[sc]
        p = P[:, 0]
        ax.set_facecolor(BG)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.09, 0.11, 0.13, 1.0))
            axis._axinfo["grid"]["color"] = GRID
            axis.set_tick_params(labelsize=7, colors=DIM)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zlabel("height m", fontsize=8, color=DIM, labelpad=-2)
        c = np.vstack([p, tgt[None]])
        mid = (c.max(0) + c.min(0)) / 2
        h = max((c.max(0) - c.min(0)).max() / 2, 0.45) + 0.2
        ax.set_xlim(mid[0] - h, mid[0] + h)
        ax.set_ylim(mid[1] - h, mid[1] + h)
        ax.set_zlim(max(0, mid[2] - h), max(0, mid[2] - h) + 2 * h)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-60)
        ax.set_title(NAMES[sc], color=TXT, fontsize=12, pad=4)
        ax.scatter(*tgt, color=C_TGT, s=30, depthshade=False)
        trail, = ax.plot([], [], [], color=C_V3, lw=1.6, alpha=0.9)
        arms = [ax.plot([], [], [], color=TXT, lw=2.4)[0] for _ in range(2)]
        rings = [ax.plot([], [], [], color=c_, lw=1.4)[0] for c_ in (C_V3, C_V3, "#ff7b72", "#ff7b72")]
        upv, = ax.plot([], [], [], color=C_TGT, lw=1.2)
        txt = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, ha="center", color=TXT, fontsize=10, family="DejaVu Sans Mono")
        arts.append((p, R[:, 0], trail, arms, rings, upv, txt))
    for f in range(nfr + int(1.0 * fps)):
        tf = min(f * slow / fps, T)
        k = min(int(tf / dt), len(t) - 1)
        for p, Rr, trail, arms, rings, upv, txt in arts:
            trail.set_data(p[:k + 1, 0], p[:k + 1, 1])
            trail.set_3d_properties(p[:k + 1, 2])
            w, rg, up, nose = quad_geom(p[k], Rr[k])
            for i, ln in enumerate(arms):
                ln.set_data(w[2 * i:2 * i + 2, 0], w[2 * i:2 * i + 2, 1])
                ln.set_3d_properties(w[2 * i:2 * i + 2, 2])
            for ln, r in zip(rings, rg):
                ln.set_data(r[:, 0], r[:, 1])
                ln.set_3d_properties(r[:, 2])
            upv.set_data([p[k, 0], up[0]], [p[k, 1], up[1]])
            upv.set_3d_properties([p[k, 2], up[2]])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, Rr[k][2, 2]))))
            err = np.linalg.norm(p[k] - tgt)
            txt.set_text(f"t {t[k]:4.2f} s   tilt {tilt:5.1f}°   err {100 * err:5.1f} cm")
        fig.savefig(tmp / f"f{f:04d}.png", facecolor=BG)
    plt.close(fig)
    mp4, gif = out / "recovery-reel.mp4", out / "recovery-reel.gif"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%04d.png"), "-c:v", "libx264",
                    "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=none:diff_mode=rectangle", str(gif)], check=True)
    return mp4, gif


def envelope(out, seed, n=64):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TXT, "axes.labelcolor": DIM,
                         "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": GRID})
    tgt = np.array(ev.TARGET)
    nominal = ev.load_nominal()
    res = {}
    for name, mk in (("v3 policy", lambda m: ev.TorchPolicy(str(POLICY))), ("PID fallback", lambda m: ev.make_pid(m, nominal))):
        env, _, _ = ev.build(n, 0, seed, False, SCEN, "fw3")
        ctrl = mk(env.n)
        t, r = fly(ctrl, SCEN, n, seed, True, secs=4.0)
        res[name] = r
    fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.2), dpi=110, facecolor=BG, sharex=True)
    for j, sc in enumerate(SCEN):
        for name, col in (("PID fallback", C_PID), ("v3 policy", C_V3)):
            P, R, A, ok = res[name][sc]
            err = np.linalg.norm(P - tgt, axis=-1)
            tilt = np.degrees(np.arccos(np.clip(R[..., 2, 2], -1, 1)))
            for a, y in ((axs[0, j], err), (axs[1, j], tilt)):
                a.plot(t, y[:, ok], color=col, lw=0.6, alpha=0.18)
                a.plot(t, np.median(y[:, ok], axis=1), color=col, lw=2.2,
                       label=f"{name} (median; {ok.mean() * 100:.0f}% no crash)")
        axs[0, j].set_title(NAMES[sc], color=TXT, fontsize=11)
        axs[0, j].set_ylim(0, 1.6)
        axs[1, j].set_ylim(0, 185)
        axs[1, j].set_xlabel("time  s")
        axs[0, j].legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, loc="upper right")
    axs[0, 0].set_ylabel("position error  m")
    axs[1, 0].set_ylabel("tilt  °")
    for a in axs.flat:
        a.set_facecolor(PANEL)
        a.grid(color=GRID, lw=0.6)
    fig.suptitle(f"SIMULATION · {n} randomized vehicles per scenario (mass, inertia, motors, latency, sensor noise), "
                 "fw3 sensing", color=DIM, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "recovery-envelope.png", facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pol = ev.TorchPolicy(str(POLICY))
    t, runs = fly(pol, SCEN, 1, a.seed, False)
    tgt = np.array(ev.TARGET)
    for sc in SCEN:
        P, R, A, ok = runs[sc]
        print(sc, "alive" if ok[0] else "CRASH", f"final err {np.linalg.norm(P[-1, 0] - tgt):.3f} m")
    for f in reel(t, runs, out):
        print(f, f"{f.stat().st_size / 1e6:.2f} MB")
    envelope(out, a.seed)
    print(out / "recovery-envelope.png")


if __name__ == "__main__":
    main()
