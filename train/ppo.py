"""PPO with an asymmetric actor-critic for the CTBR hover/recovery task.

* Actor: obs(19) -> MLP 2x64 tanh -> linear -> tanh = mean action in [-1,1]
  (exactly the network exported per INTERFACES §6). Exploration: Gaussian
  around the mean with a state-independent log-std; the env clips to [-1,1].
* Critic: cat(actor obs, privileged true state + params) (train/priv_env.py).
* Observation normalization with running mean/std (separate for actor and
  critic inputs). Stats are frozen during each iteration's rollout + update
  and refreshed afterwards; at export they are frozen for good.
* GAE(γ, λ); timeouts (info["timeout"]) are bootstrapped with
  V(terminal_obs, terminal_priv): r_t += γ V(s_terminal).
* Curriculum over the env's initial-state distribution (config levels),
  advanced on the EMA of the episode survival rate.
* Optional action-smoothness regularization on the deterministic (mean)
  action, CAPS-style (Mysore et al. 2021), config `caps`: a temporal term on
  the second difference mu(s_t) - 2 mu(s_t-1) + mu(s_t-2) along the rollout
  (within an episode; the previous-action inputs are replaced by the unrolled
  mean actions from t-3, since the rollout obs carry the sampled ones) and a spatial term |mu(s) - mu(s + eps)|^2 with eps ~
  N(0, sigma) in normalized-observation units (dims in `spatial_exclude`
  untouched, default the previous action), both weighted per action channel. A smoothness penalty in the
  env reward would act on the sampled action and be swamped by exploration
  noise; this acts on the policy that is deployed.

Usage: python -m train.ppo --config train/config/hover_ctbr.yaml [--name NAME] [--iterations N]
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.env import ACT_DIM, OBS_DIM, DEFAULT_REWARD_WEIGHTS, EnvConfig  # noqa: E402
from sim.params import load_nominal, load_randomization  # noqa: E402
from train.priv_env import PRIV_DIM, PrivilegedQuadEnv  # noqa: E402

CRITIC_DIM = OBS_DIM + PRIV_DIM


# ---------------------------------------------------------------------------- utils
class RunningMeanStd:
    """Parallel (Chan et al.) running mean/var. std = sqrt(var + eps)."""

    def __init__(self, dim: int, eps: float = 1e-4):
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.var = torch.ones(dim, dtype=torch.float64)
        self.count = 1e-4
        self.eps = eps

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(torch.float64)
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps)

    def frozen(self):
        """(mean, std) float32 copies used for normalization."""
        return self.mean.float().clone(), self.std.float().clone()

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count, "eps": self.eps}

    def load_state_dict(self, d):
        self.mean, self.var, self.count, self.eps = d["mean"], d["var"], d["count"], d["eps"]


def mlp(sizes, act=nn.Tanh, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic part of the policy: normalized obs -> tanh mean action."""

    def __init__(self, hidden=(64, 64), init_log_std=-0.5, hover_a0: float | None = None):
        super().__init__()
        self.net = mlp([OBS_DIM, *hidden, ACT_DIM], act=nn.Tanh, out_act=nn.Tanh)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(init_log_std)))
        last = self.net[-2]
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.zeros_(m.bias)
            nn.init.orthogonal_(last.weight, gain=0.01)
            if hover_a0 is not None:   # start near hover thrust instead of 50 % collective
                last.bias[0] = math.atanh(hover_a0)

    def forward(self, xn):
        return self.net(xn)


class Critic(nn.Module):
    def __init__(self, hidden=(128, 128)):
        super().__init__()
        self.net = mlp([CRITIC_DIM, *hidden, 1], act=nn.ELU)
        with torch.no_grad():
            nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, xn):
        return self.net(xn).squeeze(-1)


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """rewards/values/dones: (T, N); dones[t]=1 if the episode ended after step t
    (timeouts must already be bootstrapped into rewards). Returns (adv, returns)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def hover_a0(nominal: dict) -> float:
    return nominal["mass"] * nominal["gravity"] / (4 * nominal["thrust_max"]) * 2 - 1


# ---------------------------------------------------------------------------- curriculum
class Curriculum:
    def __init__(self, ccfg: dict | None):
        ccfg = ccfg or {}
        self.levels = ccfg.get("levels", [{}])
        self.ema_a = float(ccfg.get("ema", 0.9))
        self.thresh = float(ccfg.get("advance_success", 0.9))
        self.min_iters = int(ccfg.get("min_iters", 30))
        self.level = 0
        self.iters_at_level = 0
        self.ema = None

    def apply(self, env_cfg: EnvConfig) -> None:
        for k, v in self.levels[self.level].items():
            if not hasattr(env_cfg, k):
                raise AttributeError(f"curriculum: unknown EnvConfig field {k}")
            setattr(env_cfg, k, tuple(v) if isinstance(v, list) else v)

    def update(self, n_timeout: int, n_term: int) -> bool:
        self.iters_at_level += 1
        n = n_timeout + n_term
        if n > 0:
            s = n_timeout / n
            self.ema = s if self.ema is None else self.ema_a * self.ema + (1 - self.ema_a) * s
        if (self.level < len(self.levels) - 1 and self.iters_at_level >= self.min_iters
                and self.ema is not None and self.ema >= self.thresh):
            self.level += 1
            self.iters_at_level = 0
            self.ema = None
            return True
        return False


# ---------------------------------------------------------------------------- env
def make_env(cfg: dict, num_envs: int | None = None, seed: int | None = None) -> PrivilegedQuadEnv:
    e = dict(cfg.get("env", {}))
    if num_envs is not None:
        e["num_envs"] = num_envs
    e["seed"] = cfg.get("seed", 0) if seed is None else seed
    for k in ("target_z",):
        if k in e and isinstance(e[k], list):
            e[k] = tuple(e[k])
    w = dict(DEFAULT_REWARD_WEIGHTS)
    w.update(cfg.get("reward_weights", {}) or {})
    ec = EnvConfig(**e, reward_weights=w)
    rand_path = cfg.get("randomization")
    rand = load_randomization(os.path.join(ROOT, rand_path)) if rand_path else None
    return PrivilegedQuadEnv(ec, randomization=rand, extra_reward=cfg.get("extra_reward"))


# ---------------------------------------------------------------------------- trainer
class PPO:
    def __init__(self, cfg: dict, out_dir: str | None = None, log=print):
        self.cfg = cfg
        self.out_dir = out_dir
        self.log = log
        seed = int(cfg.get("seed", 0))
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed + 1)
        p = cfg["ppo"]
        self.p = p
        self.nom = load_nominal()
        self.env = make_env(cfg)
        self.n = self.env.n
        m = cfg.get("model", {})
        self.actor = Actor(tuple(m.get("actor_hidden", (64, 64))), p.get("init_log_std", -0.5),
                           hover_a0(self.nom) if m.get("hover_bias_init", True) else None)
        self.critic = Critic(tuple(m.get("critic_hidden", (128, 128))))
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                    lr=float(p["lr"]))
        self.lr = float(p["lr"])
        self.obs_rms = RunningMeanStd(OBS_DIM)
        self.crit_rms = RunningMeanStd(CRITIC_DIM)
        self.curr = Curriculum(cfg.get("curriculum"))
        init = cfg.get("init_from")
        if init:   # warm start (fine-tuning): weights, normalizers, curriculum level
            st = torch.load(os.path.join(ROOT, init) if not os.path.isabs(init) else init,
                            map_location="cpu", weights_only=False)
            self.actor.load_state_dict(st["actor"])
            self.critic.load_state_dict(st["critic"])
            self.obs_rms.load_state_dict(st["obs_rms"])
            self.crit_rms.load_state_dict(st["crit_rms"])
            self.curr.level = min(int(cfg.get("init_level", st.get("curriculum_level", 0))),
                                  len(self.curr.levels) - 1)
            if "init_log_std" in cfg.get("ppo", {}) and cfg.get("reset_log_std", False):
                with torch.no_grad():
                    self.actor.log_std.fill_(float(cfg["ppo"]["init_log_std"]))
        self.curr.apply(self.env.cfg)
        self.it = 0
        self._apply_reward_ramp()
        self.obs = self.env.reset()
        self.priv = self.env.privileged()
        # warm-up normalization stats with the initial state distribution
        self.obs_rms.update(self.obs)
        self.crit_rms.update(torch.cat((self.obs, self.priv), -1))
        self.it = 0
        self.ep_ret = torch.zeros(self.n)
        self.ep_len = torch.zeros(self.n)
        self.done_stats = []

    def _apply_reward_ramp(self) -> None:
        """reward_ramp: {weight: {start, end, iters}} -> linear schedule over iterations."""
        for k, r in (self.cfg.get("reward_ramp") or {}).items():
            frac = min(self.it / max(float(r["iters"]), 1.0), 1.0)
            self.env.w[k] = float(r["start"]) + frac * (float(r["end"]) - float(r["start"]))

    # -- policy helpers
    def _dist(self, obs_n):
        mean = self.actor(obs_n)
        std = self.actor.log_std.clamp(self.p.get("min_log_std", -3.0), 1.0).exp()
        return torch.distributions.Normal(mean, std.expand_as(mean))

    def export_state(self) -> dict:
        mean, std = self.obs_rms.frozen()
        return {"actor": copy.deepcopy(self.actor.state_dict()), "obs_mean": mean, "obs_std": std,
                "actor_hidden": list(self.cfg.get("model", {}).get("actor_hidden", (64, 64))),
                "iteration": self.it, "curriculum_level": self.curr.level, "config": self.cfg}

    def save(self, path: str) -> None:
        st = self.export_state()
        st.update({"critic": self.critic.state_dict(), "obs_rms": self.obs_rms.state_dict(),
                   "crit_rms": self.crit_rms.state_dict(), "opt": self.opt.state_dict()})
        torch.save(st, path)

    # -- one iteration
    def iterate(self) -> dict:
        p, env = self.p, self.env
        T = int(p["rollout_steps"])
        gamma, lam = float(p["gamma"]), float(p["lam"])
        rscale = float(self.cfg.get("reward_scale", 1.0))
        om, os_ = self.obs_rms.frozen()
        cm, cs = self.crit_rms.frozen()
        n = self.n
        b_obs = torch.zeros(T, n, OBS_DIM)
        b_crit = torch.zeros(T, n, CRITIC_DIM)
        b_act = torch.zeros(T, n, ACT_DIM)
        b_logp = torch.zeros(T, n)
        b_val = torch.zeros(T, n)
        b_rew = torch.zeros(T, n)
        b_done = torch.zeros(T, n, dtype=torch.bool)
        n_timeout = n_term = 0
        ep_rets, ep_lens = [], []
        t0 = time.time()
        with torch.no_grad():
            for t in range(T):
                crit_in = torch.cat((self.obs, self.priv), -1)
                obs_n = (self.obs - om) / os_
                d = self._dist(obs_n)
                a = d.sample()
                v = self.critic((crit_in - cm) / cs)
                nobs, r, done, info = env.step(a)
                r = r * rscale
                self.ep_ret += r
                self.ep_len += 1
                if done.any():
                    to = info["timeout"]
                    if to.any():
                        tcrit = torch.cat((info["terminal_obs"], info["terminal_priv"]), -1)
                        tv = self.critic((tcrit[to] - cm) / cs)
                        r = r.clone()
                        r[to] += gamma * tv
                    n_timeout += int(to.sum())
                    n_term += int((done & ~to).sum())
                    ep_rets.append(self.ep_ret[done].clone())
                    ep_lens.append(self.ep_len[done].clone())
                    self.ep_ret[done] = 0
                    self.ep_len[done] = 0
                b_obs[t], b_crit[t], b_act[t] = self.obs, crit_in, a
                b_logp[t], b_val[t], b_rew[t], b_done[t] = d.log_prob(a).sum(-1), v, r, done
                self.obs = nobs
                self.priv = env.privileged()
            last_v = self.critic((torch.cat((self.obs, self.priv), -1) - cm) / cs)
            adv, ret = compute_gae(b_rew, b_val, b_done, last_v, gamma, lam)
        t_roll = time.time() - t0

        # -- update
        t1 = time.time()
        N = T * n
        f_obs_n = ((b_obs - om) / os_).reshape(N, -1)
        f_crit_n = ((b_crit - cm) / cs).reshape(N, -1)
        f_act, f_logp = b_act.reshape(N, -1), b_logp.reshape(N)
        f_adv, f_ret, f_val = adv.reshape(N), ret.reshape(N), b_val.reshape(N)
        mb = N // int(p["minibatches"])
        clip = float(p["clip"])
        stats = {"pg": 0.0, "vf": 0.0, "kl": 0.0, "ent": 0.0, "clipfrac": 0.0, "caps_t": 0.0, "caps_s": 0.0}
        caps = self.cfg.get("caps") or {}
        w_t, w_s = float(caps.get("temporal", 0.0)), float(caps.get("spatial", 0.0))
        if w_t or w_s:
            ch_w = torch.tensor(caps.get("channel_weights", [1.0, 0.25, 0.25, 0.25]), dtype=torch.float32)
            sigma = float(caps.get("sigma", 0.05))
            eps_mask = torch.ones(OBS_DIM)
            eps_mask[list(caps.get("spatial_exclude", [15, 16, 17, 18]))] = 0.0   # previous action is exact
            t_idx = torch.arange(N) // n
            done_f = b_done.reshape(N)
            prev_ok = torch.zeros(N, dtype=torch.bool)
            okix = torch.nonzero(t_idx >= 3).squeeze(-1)
            prev_ok[okix] = ~done_f[okix - n] & ~done_f[okix - 2 * n] & ~done_f[okix - 3 * n]
            a_m, a_s = om[15:19], os_[15:19]

            def with_prev(x, mu):   # obs with the previous-action dims set to the mean action mu
                return torch.cat((x[:, :15], (mu - a_m) / a_s), -1)
        nupd = 0
        for ep in range(int(p["epochs"])):
            perm = torch.randperm(N, generator=self.gen)
            for k in range(int(p["minibatches"])):
                ix = perm[k * mb:(k + 1) * mb]
                d = self._dist(f_obs_n[ix])
                logp = d.log_prob(f_act[ix]).sum(-1)
                ratio = torch.exp(logp - f_logp[ix])
                a_ = f_adv[ix]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(1 - clip, 1 + clip) * a_).mean()
                v = self.critic(f_crit_n[ix])
                if p.get("value_clip", True):
                    v_cl = f_val[ix] + (v - f_val[ix]).clamp(-clip, clip)
                    vf = torch.max((v - f_ret[ix]) ** 2, (v_cl - f_ret[ix]) ** 2).mean()
                else:
                    vf = ((v - f_ret[ix]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + float(p["vf_coef"]) * vf - float(p.get("ent_coef", 0.0)) * ent
                if w_t:
                    jx = ix[prev_ok[ix]]
                    if jx.numel() > 0:
                        # the rollout obs carry the SAMPLED previous action; on the vehicle it is the mean
                        # action, so unroll the mean-action feedback from t-3 (states fixed)
                        mu3 = self.actor(f_obs_n[jx - 3 * n])
                        mu2 = self.actor(with_prev(f_obs_n[jx - 2 * n], mu3))
                        mu1 = self.actor(with_prev(f_obs_n[jx - n], mu2))
                        mu0 = self.actor(with_prev(f_obs_n[jx], mu1))
                        lt = (((mu0 - 2 * mu1 + mu2) ** 2) * ch_w).sum(-1).mean()
                        loss = loss + w_t * lt
                        stats["caps_t"] += lt.item()
                if w_s:
                    xo = f_obs_n[ix]
                    eps = torch.randn(xo.shape, generator=self.gen) * sigma * eps_mask
                    ls = (((self.actor(xo + eps) - d.mean) ** 2) * ch_w).sum(-1).mean()
                    loss = loss + w_s * ls
                    stats["caps_s"] += ls.item()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         float(p["max_grad_norm"]))
                self.opt.step()
                with torch.no_grad():
                    lr_ratio = logp - f_logp[ix]
                    kl = ((torch.exp(lr_ratio) - 1) - lr_ratio).mean().item()
                    # adaptive LR on KL (rsl_rl style)
                    tgt = p.get("kl_target")
                    if tgt:
                        if kl > 2.0 * tgt:
                            self.lr = max(self.lr / 1.5, float(p.get("lr_min", 1e-5)))
                        elif kl < 0.5 * tgt:
                            self.lr = min(self.lr * 1.5, float(p.get("lr_max", 1e-2)))
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.item()
                stats["kl"] += kl; stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
                nupd += 1
        for k in stats:
            stats[k] /= max(nupd, 1)
        # refresh normalization stats with this rollout's raw data
        self.obs_rms.update(b_obs)
        self.crit_rms.update(b_crit)
        t_upd = time.time() - t1

        self._apply_reward_ramp()
        advanced = self.curr.update(n_timeout, n_term)
        self.curr.apply(env.cfg)
        self.it += 1
        er = torch.cat(ep_rets) if ep_rets else torch.zeros(0)
        el = torch.cat(ep_lens) if ep_lens else torch.zeros(0)
        stats.update({
            "iter": self.it, "steps": self.it * N, "level": self.curr.level, "advanced": int(advanced),
            "survival": n_timeout / max(n_timeout + n_term, 1), "episodes": n_timeout + n_term,
            "ep_ret": er.mean().item() if er.numel() else float("nan"),
            "ep_len": el.mean().item() if el.numel() else float("nan"),
            "rew_step": b_rew.mean().item(), "std": self.actor.log_std.exp().mean().item(),
            "lr": self.lr, "t_roll": t_roll, "t_upd": t_upd,
        })
        return stats


def train(cfg: dict, out_dir: str, iterations: int | None = None, log=None) -> PPO:
    log = log or (lambda *a: print(*a, flush=True))
    torch.set_num_threads(int(cfg.get("threads", 4)))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    algo = PPO(cfg, out_dir, log)
    iters = int(iterations if iterations is not None else cfg["ppo"]["iterations"])
    ckpt_every = int(cfg.get("checkpoint_every", 100))
    csv_path = os.path.join(out_dir, "progress.csv")
    writer = None
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        for i in range(iters):
            s = algo.iterate()
            s["wall"] = time.time() - t0
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(s.keys()))
                writer.writeheader()
            writer.writerow(s)
            f.flush()
            if s["advanced"]:
                log(f"[it {s['iter']}] curriculum -> level {s['level']}")
                algo.save(os.path.join(out_dir, f"level{s['level'] - 1}_done.pt"))
            if i % int(cfg.get("log_every", 10)) == 0 or i == iters - 1:
                log(f"it {s['iter']:5d} steps {s['steps']/1e6:7.1f}M lvl {s['level']} surv {s['survival']:.3f} "
                    f"ret {s['ep_ret']:7.2f} len {s['ep_len']:6.0f} r/st {s['rew_step']:.4f} "
                    f"std {s['std']:.3f} kl {s['kl']:.4f} vf {s['vf']:.4f} lr {s['lr']:.1e} "
                    f"t {s['t_roll']:.2f}+{s['t_upd']:.2f}s wall {s['wall']/60:.1f}m")
            if (i + 1) % ckpt_every == 0 or i == iters - 1:
                algo.save(os.path.join(out_dir, "last.pt"))
    return algo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "train", "config", "hover_ctbr.yaml"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--init-from", default=None, help="checkpoint to warm-start from (fine-tuning)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.num_envs is not None:
        cfg["env"]["num_envs"] = a.num_envs
    if a.init_from is not None:
        cfg["init_from"] = a.init_from
    name = a.name or cfg.get("run_name", "run")
    out = os.path.join(ROOT, "runs", name)
    train(cfg, out, a.iterations)
    print(f"saved {out}/last.pt")


if __name__ == "__main__":
    main()
