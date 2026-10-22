"""
Step 5 -- GUARDIAN safety shield (discrete-action graph CBF) + risk & XAI utilities.

This module implements the certified action filter and the risk/explainability
utilities described in the journal paper, so the numbers reported there can be
regenerated from real runs:

    Eq. (6)   barrier margin        h_j(x_i) = ||p_i - p_j|| - (r_i + r_j + delta)
    Eq. (7)   discrete CBF cond.    h_j(x_i^+(a)) >= (1 - gamma_s) * h_j(x_i)
    Eq. (8)   action selection      argmin_{a in A_safe} ||v(a) - v(a_nom)||^2,
                                    fallback a_fb = the discrete climb command
    Eq. (9)   quantile Huber loss   (critic training)
    Eq. (10)  CVaR_alpha from the lowest alpha fraction of quantile outputs
    Eq. (11)  blended value         (1 - beta) * mean + beta * CVaR
    Eq. (14)  Spearman alignment    attention mass vs. barrier DANGER (negated
                                    margins): positive rho = attention on the
                                    most safety-critical neighbor

Design choices (match the paper):
    * Discrete action set A_i with K = 11 commands:
      4 compass sectors x 2 magnitudes + climb + descend + hold.
    * Fallback a_fb (assumption A4) = the discrete climb command (index 8);
      when even a_fb is inadmissible the caller sees `fallback_used` and the
      certificate-suspension flag semantics of the paper apply.
    * One-step propagation p+ = p + v_cmd * dt (assumption A2); peer agents
      are held at their current positions (ego-only prediction), while moving
      obstacles are propagated one step ahead with their estimated velocities;
      the A1 disturbance bound is absorbed inside Eq. (6) by the inflation
      delta, so Eq. (7) is applied exactly as stated in the paper.
    * Vectorized NumPy over agents: the filter cost grows with the number of
      evaluated margins, not with the joint action space.

The shield runs standalone without the simulator (pure NumPy), so you can
benchmark the per-step cost with `python 05_guardian_shield.py` before touching
AirSim. The per-step overhead figures reported in Table 7 of the paper are the
output of this benchmark on the development host (commodity dual-core x86-64
server CPU, NumPy 2.1); re-running it on your hardware reproduces the protocol
and should give values within a few percent of the published ones. The module
plugs into the MARL environment through
`04_marl_gym_env.AirSimMultiUAVEnv.attach_guardian_shield()`.

Install:  pip install numpy
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------- constants ---
MAX_SPEED = 4.0          # m/s, must match 04_marl_gym_env.py
STEP_TIME = 0.1          # s per control step (SteppableClock), matches the paper

# Discrete action set A_i (NED frame: x=North, y=East, z=Down -> climb is -z).
# K = 11: four compass sectors x two magnitudes, climb, descend, hold.
_SECTOR_DIRS = np.array([
    [1.0, 0.0, 0.0],     # North
    [0.0, 1.0, 0.0],     # East
    [-1.0, 0.0, 0.0],    # South
    [0.0, -1.0, 0.0],    # West
], dtype=np.float64)
_HORIZONTAL_MAGS = (0.5 * MAX_SPEED, 1.0 * MAX_SPEED)
CLIMB = np.array([0.0, 0.0, -2.0])       # brake-and-climb fallback, m/s (NED)
DESCEND = np.array([0.0, 0.0, 2.0])
HOLD = np.array([0.0, 0.0, 0.0])

DISCRETE_ACTIONS = np.array(
    [d * m for m in _HORIZONTAL_MAGS for d in _SECTOR_DIRS]
    + [CLIMB, DESCEND, HOLD], dtype=np.float64
)                                        # shape (K, 3)
K = len(DISCRETE_ACTIONS)
FALLBACK_INDEX = K - 3                   # first of the special commands = climb


@dataclass
class ShieldConfig:
    """Barrier and shield parameters (paper Section 4.4 / Table 4)."""
    delta: float = 0.5          # inflation margin (m) -- absorbs the A1 bound
    gamma_s: float = 0.35       # allowed relative margin decrease per step
    agent_radius: float = 0.35  # collision radius of each UAV (m)
    obstacle_radius: float = 1.0  # default threat radius of obstacles (m)


@dataclass
class ShieldReport:
    """Per-step diagnostics (the paper's shield flags + margins)."""
    chosen_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    nominal_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    fallback_used: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    n_admissible: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    min_margin: float = float("inf")
    project_fraction: float = 0.0      # share of actions the shield replaced


# ------------------------------------------------------------- the shield ---
class GuardianShield:
    """Discrete-action graph CBF shield, vectorized over agents.

    Use `filter()` with explicit states (offline benchmarking, unit tests) or
    the `observe_env()` + `filter_from_env()` pair inside the AirSim loop.
    """

    def __init__(self, cfg: ShieldConfig | None = None, step_time: float = STEP_TIME,
                 max_speed: float = MAX_SPEED):
        self.cfg = cfg or ShieldConfig()
        self.dt = step_time
        self.vmax = max_speed
        self.actions = DISCRETE_ACTIONS.copy()          # (K, 3) m/s in NED
        self.last_report: ShieldReport | None = None

    # ------------------------------------------------------ margin helpers ---
    def margins(self, P: np.ndarray, Q: np.ndarray, r_q: np.ndarray) -> np.ndarray:
        """Eq. (6): signed distance margins between agents P (N, 3) and a
        second point set Q (M, 3) with radii r_q (M,). The inflation delta is
        added here exactly once. Returns (N, M). Callers mask self-pairs."""
        diff = P[:, None, :] - np.asarray(Q, float)[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        rad = self.cfg.agent_radius + np.asarray(r_q, float)[None, :] + self.cfg.delta
        return dist - rad

    # ---------------------------------------------------------- main filter ---
    def filter(self, positions: np.ndarray, nominal_actions: np.ndarray,
               obstacle_pos: np.ndarray | None = None,
               obstacle_rad: np.ndarray | None = None,
               obstacle_vel: np.ndarray | None = None) -> tuple[np.ndarray, ShieldReport]:
        """Shield a batch of continuous nominal actions (N, 3), m/s in NED.

        Returns (safe_velocities (N, 3) m/s, ShieldReport). Implements Eqs.
        (7)-(8): evaluate every discrete candidate, keep admissible ones, pick
        the one closest to the nominal command, else the A4 fallback.
        """
        cfg = self.cfg
        P = np.asarray(positions, dtype=np.float64)
        N = P.shape[0]
        rep = ShieldReport()

        # nominal discrete index per agent: nearest command in velocity space
        v_nom = np.clip(np.asarray(nominal_actions, dtype=np.float64),
                        -self.vmax, self.vmax)
        d2 = ((self.actions[None, :, :] - v_nom[:, None, :]) ** 2).sum(-1)   # (N, K)
        rep.nominal_index = np.argmin(d2, axis=1)

        idx_n = np.arange(N)
        r_agents = np.full(N, cfg.agent_radius)

        # current margins h_j(x_i): agent-agent (diagonal masked) + obstacles
        h_now = self.margins(P, P, r_agents)
        h_now[idx_n, idx_n] = np.inf
        r_obs = None
        if obstacle_pos is not None and len(obstacle_pos):
            r_obs = (np.asarray(obstacle_rad, float) if obstacle_rad is not None
                     else np.full(len(obstacle_pos), cfg.obstacle_radius))
            h_now = np.hstack([h_now, self.margins(P, np.asarray(obstacle_pos, float), r_obs)])

        # candidate successors for all K actions: (N*K, 3)
        succ_flat = (P[:, None, :] + self.actions[None, :, :] * self.dt).reshape(N * K, 3)
        h_succ = self.margins(succ_flat, P, r_agents).reshape(N, K, N)
        h_succ[idx_n[:, None], np.arange(K)[None, :], idx_n[:, None]] = np.inf  # self-pairs
        if obstacle_pos is not None and len(obstacle_pos):
            # moving obstacles are propagated one step ahead with their
            # estimated velocities (paper Section 4.4); zero velocity = static
            Q_obs = np.asarray(obstacle_pos, float)
            if obstacle_vel is not None and len(obstacle_vel):
                Q_obs = Q_obs + np.asarray(obstacle_vel, float) * self.dt
            h_succ_obs = self.margins(succ_flat, Q_obs, r_obs).reshape(N, K, len(obstacle_pos))
            h_succ = np.concatenate([h_succ, h_succ_obs], axis=2)

        # Eq. (7), exactly as in the paper (delta already absorbed the A1 bound)
        thresh = (1.0 - cfg.gamma_s) * h_now[:, None, :]                     # (N, 1, M)
        admissible = (h_succ >= thresh).all(axis=2)                          # (N, K)

        # Eq. (8): admissible action closest to the nominal, else fallback
        dev = d2.copy()
        dev[~admissible] = np.inf
        chosen = np.argmin(dev, axis=1)
        infeasible = ~admissible.any(axis=1)
        chosen[infeasible] = FALLBACK_INDEX

        rep.chosen_index = chosen
        rep.fallback_used = infeasible
        rep.n_admissible = admissible.sum(axis=1)
        rep.min_margin = float(h_now.min()) if np.isfinite(h_now).any() else float("inf")
        rep.project_fraction = float(np.mean(chosen != rep.nominal_index))
        return self.actions[chosen].copy(), rep

    # ------------------------------------------------- env-facing adapter ---
    def observe_env(self, env) -> None:
        """Fetch current agent positions (and optional obstacles) from the
        AirSim environment of step 04. Call once before filter_from_env()."""
        self._pos = np.array([env._position(d) for d in env.agents], dtype=np.float64)
        obs = getattr(env, "dynamic_obstacles", None)
        if obs is not None and len(obs):
            arr = np.asarray(obs, dtype=np.float64)
            self._obs_pos, self._obs_rad = arr[:, :3], arr[:, 3]
        else:
            self._obs_pos, self._obs_rad = None, None
        # optional (M, 3) velocity estimates; zeros when the env provides none
        vel = getattr(env, "dynamic_obstacle_velocities", None)
        self._obs_vel = (np.asarray(vel, dtype=np.float64)
                         if vel is not None and len(vel) else None)

    def filter_from_env(self, actions: dict) -> dict:
        """Shield a {drone: Box(-1,1,3)} action dict from the step-04 env.

        The env scales normalized actions by MAX_SPEED internally, so the
        shield rescales them, certifies, and returns normalized commands.
        """
        names = list(actions.keys())
        if not hasattr(self, "_pos") or len(self._pos) != len(names):
            raise RuntimeError("call observe_env(env) before filter_from_env()")
        v_nom = np.array([np.asarray(actions[n], np.float64) * self.vmax
                          for n in names])
        safe_v, rep = self.filter(self._pos, v_nom,
                                  obstacle_pos=self._obs_pos,
                                  obstacle_rad=self._obs_rad,
                                  obstacle_vel=getattr(self, "_obs_vel", None))
        self.last_report = rep
        return {n: (safe_v[i] / self.vmax).astype(np.float32)
                for i, n in enumerate(names)}


# ------------------------------------------------- risk & XAI utilities ---
def quantile_huber_loss(theta_pred: np.ndarray, theta_target: np.ndarray,
                        kappa: float = 1.0) -> float:
    """Eq. (9): quantile Huber loss between predicted and target quantiles.

    theta_pred, theta_target: (N_q,) quantile outputs/targets. The uniform
    quantile fraction is tau_j = (j + 0.5) / N_q.
    """
    N_q = len(theta_pred)
    tau = (np.arange(N_q) + 0.5) / N_q
    u = np.asarray(theta_target, float)[None, :] - np.asarray(theta_pred, float)[:, None]
    huber = np.where(np.abs(u) <= kappa, 0.5 * u ** 2, kappa * (np.abs(u) - 0.5 * kappa))
    return float(np.mean(np.abs(tau[:, None] - (u < 0).astype(float)) * huber))


def cvar_from_quantiles(theta_q: np.ndarray, alpha: float = 0.1) -> float:
    """Eq. (10): CVaR_alpha as the mean of the lowest alpha fraction of quantiles."""
    k = max(1, int(np.ceil(alpha * len(theta_q))))
    return float(np.mean(np.sort(np.asarray(theta_q, float))[:k]))


def blended_value(mean_v: float, cvar_v: float, beta: float) -> float:
    """Eq. (11): (1 - beta) * mean + beta * CVaR with the scheduled risk weight."""
    return float((1.0 - beta) * mean_v + beta * cvar_v)


def spearman_alignment(attention: np.ndarray, margins: np.ndarray,
                       danger: bool = True) -> float:
    """Eq. (14): Spearman rank correlation between the GAT attention mass and
    the barrier DANGER of the same neighbors, per agent and step. With
    danger=True (default) the margins are negated first, so a POSITIVE rho
    means the agent attends most to the neighbor with the SMALLEST margin,
    i.e., the most safety-critical one. Returns rho in [-1, 1]; NaN if fewer
    than 2 neighbors."""
    attention = np.asarray(attention, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    if len(attention) < 2:
        return float("nan")
    if danger:
        margins = -margins
    ra = _ranks(attention)
    rm = _ranks(margins)
    ra -= ra.mean()
    rm -= rm.mean()
    denom = np.sqrt((ra ** 2).sum() * (rm ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rm).sum() / denom)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling."""
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0     # average of 1-based ranks
        i = j + 1
    return ranks


# --------------------------------------------------------------- benchmark ---
def benchmark(agent_counts=(6, 8, 12, 20, 30), steps: int = 400, seed: int = 0,
              dense: bool = True) -> None:
    """Measure the per-step filter cost (paper Section 6.5, Table 7 protocol).

    Agents are scattered in a 60 m x 60 m x 8 m box. Two graph regimes are
    measured: with dense=True every agent evaluates all N-1 peer margins
    (dense all-to-all worst case); with dense=False the same dense graph is
    evaluated under six moving obstacles (radii 0.8-1.6 m, velocities up to
    2 m/s, propagated one step ahead), matching the obstacle load of the
    stress scenarios. Neighbor pruning to the k nearest peers is an
    environment-layer option for swarms far beyond the tested range and is
    not exercised here.
    """
    rng = np.random.default_rng(seed)
    shield = GuardianShield()
    print(f"{'agents':>7} {'graph':>7} {'ms/step':>9} {'share of 100ms':>15}")
    for n in agent_counts:
        P = rng.uniform([-30.0, -30.0, -10.0], [30.0, 30.0, -2.0], size=(n, 3))
        if dense:
            obs_pos, obs_rad, obs_vel = None, None, None
        else:  # dense graph + six moving obstacles (urban stress load)
            obs_pos = rng.uniform([-30.0, -30.0, -10.0], [30.0, 30.0, -2.0], size=(6, 3))
            obs_rad = rng.uniform(0.8, 1.6, size=6)
            obs_vel = rng.uniform(-2.0, 2.0, size=(6, 3))
        t0 = time.perf_counter()
        for _ in range(steps):
            a_nom = rng.uniform(-MAX_SPEED, MAX_SPEED, size=(n, 3))
            _, _ = shield.filter(P, a_nom, obstacle_pos=obs_pos,
                                 obstacle_rad=obs_rad, obstacle_vel=obs_vel)
        dt_ms = (time.perf_counter() - t0) / steps * 1000.0
        print(f"{n:>7} {'dense' if dense else 'urban':>7} {dt_ms:>9.2f} {dt_ms / 100.0:>14.1f}%")
    print("(control interval: 100 ms -- the last column is the used fraction)")


# ------------------------------------------------------------- self-test ---
if __name__ == "__main__":
    print(f"K = {K} discrete actions: 4 sectors x 2 magnitudes + climb/descend/hold")
    print("fallback index:", FALLBACK_INDEX, "=", DISCRETE_ACTIONS[FALLBACK_INDEX])

    # functional check: two agents closing head-on must be projected apart
    P = np.array([[0.0, 0.0, -5.0], [1.6, 0.0, -5.0]])
    v_nom = np.array([[MAX_SPEED, 0.0, 0.0], [-MAX_SPEED, 0.0, 0.0]])
    shield = GuardianShield()
    safe, rep = shield.filter(P, v_nom)
    print("chosen:", rep.chosen_index, "| fallback:", rep.fallback_used,
          "| min margin: %.2f m" % rep.min_margin)
    sep_after = np.linalg.norm((P + safe * STEP_TIME)[0] - (P + safe * STEP_TIME)[1])
    print("one-step separation after filter: %.2f m (before: %.2f m)"
          % (sep_after, np.linalg.norm(P[0] - P[1])))
    # forward invariance: the separation must not shrink (hold is admissible)
    assert sep_after >= np.linalg.norm(P[0] - P[1]) or rep.fallback_used.any()

    # a clear obstacle straight ahead must push the agent off the nominal path
    P1 = np.array([[0.0, 0.0, -5.0]])
    v1 = np.array([[MAX_SPEED, 0.0, 0.0]])
    obst = np.array([[2.0, 0.0, -5.0]])
    safe1, rep1 = shield.filter(P1, v1, obstacle_pos=obst, obstacle_rad=np.array([1.0]))
    print("obstacle test -> chosen velocity:", safe1[0], "(must differ from nominal)")
    assert not np.allclose(safe1[0], v1[0]), "shield failed to project around obstacle"

    # risk utilities sanity
    q = np.linspace(-1.0, 1.0, 51)
    print("CVaR_0.1(q) =", round(cvar_from_quantiles(q), 3),
          "| blended(beta=0.4) =", round(blended_value(0.0, cvar_from_quantiles(q), 0.4), 3))
    print("alignment(att=[0.6,0.3,0.1], danger of margins=[1,2,3]) =",
          round(spearman_alignment([0.6, 0.3, 0.1], [1, 2, 3]), 3),
          "(positive = attention on the most critical neighbor)")
    print("quantile Huber (perfect fit) =",
          round(quantile_huber_loss(np.linspace(0, 1, 51), np.linspace(0, 1, 51)), 4))

    benchmark()
    print("[OK] GUARDIAN shield self-test finished.")
