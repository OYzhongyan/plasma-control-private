"""Per-step contributions to the unchanged official F2b competition score.

The undiscounted, unnormalized episode reward sum equals its offline score when
both use the same trajectory and target. PPO may separately normalize/discount
these rewards. No additional shaping or reward-induced termination is applied.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from evaluation.eval.score import config, metrics, penalties, scoring
from eval_files.trajectory_utils import extract_semifinal_trajectory_from_obs_list


def _scalar(value: Any, default: float = 0.0) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(array[0]) if array.size else default


def _target_from_observation(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Geometry references are targets, never the actual observed geometry."""
    required = ("reference_lcfs_points", "reference_rX", "reference_zX", "reference_strike_r", "reference_strike_z")
    missing = [key for key in required if key not in obs]
    if missing:
        raise ValueError(f"Competition reward requires target references: {', '.join(missing)}")
    r = np.asarray(obs["reference_rX"], dtype=np.float64).reshape(4)
    z = np.asarray(obs["reference_zX"], dtype=np.float64).reshape(4)
    return {
        "lcfs_points": np.asarray(obs["reference_lcfs_points"], dtype=np.float64).copy(),
        "Xpt_main": np.column_stack([r[[1, 2]], z[[1, 2]]]),
        "Xpt_sec": np.column_stack([r[[0, 3]], z[[0, 3]]]),
        "strike": np.column_stack([np.asarray(obs["reference_strike_r"]).reshape(8), np.asarray(obs["reference_strike_z"]).reshape(8)]).astype(np.float64),
    }


def _reference_observation(target: dict[str, Any]) -> dict[str, np.ndarray]:
    main = np.asarray(target["Xpt_main"], dtype=np.float64)
    secondary = np.asarray(target["Xpt_sec"], dtype=np.float64)
    strike = np.asarray(target["strike"], dtype=np.float64)
    points = np.asarray([secondary[0], main[0], main[1], secondary[1]])
    return {"reference_rX": points[:, 0], "reference_zX": points[:, 1],
            "reference_strike_r": strike[:, 0], "reference_strike_z": strike[:, 1]}


@dataclass
class CompetitionReward:
    """Streaming official score, normally returning 0..30/500 per step.

    ``target`` optionally supplies an official target block (or a F2b/F2a dict).
    Otherwise freeze geometry from the active environment references on step 1.
    Legacy constructor keywords remain accepted for saved-config compatibility;
    reward_weights and score_lower_bound do not override official scoring rules.

    ``info.inference_time_ms`` is optional and is never synthesized. If provided,
    the original whole-episode gamma rule applies. ``timeout``/``hfm_timeout`` in
    info permanently invalidates the episode. Such global invalidation reverses
    previous credit in the current reward, which can therefore be negative.
    """
    reward_weights: Any = None
    score_lower_bound: bool = True
    target: dict[str, Any] | None = None
    last_components: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.horizon = config.get_total_steps()
        self._ip_reference = config.build_Ip_ref()
        self._weights = config.metric_weights()
        self._fixed_target = deepcopy(self.target)
        if self._fixed_target is not None:
            self._fixed_target = self._fixed_target.get("F2b", self._fixed_target.get("F2a", self._fixed_target))
        self.reset_state()

    def reset_state(self) -> None:
        self._step = 0
        self._finished = False
        self._target = deepcopy(self._fixed_target)
        self._reference = _reference_observation(self._target) if self._target is not None else None
        self._coil_mask = 1.0
        self._timing_over_steps = 0
        self._timeout = False
        self._metric_totals = {key: 0.0 for key in self._weights}
        self._credited_metrics = {key: 0.0 for key in self._weights}
        self.last_components = {}

    def _step_scores(self, frame, index):
        target = self._target
        ip_ref = self._ip_reference[index:index+1]
        eps = {
            config.METRIC_IP: metrics.epsilon_Ip(frame["Ip"], ip_ref),
            config.METRIC_LCFS: metrics.epsilon_LCFS(target["lcfs_points"], frame["lcfs_per_step"]),
            config.METRIC_X: metrics.epsilon_xpoints(frame["Xpt_main"], target["Xpt_main"]),
            config.METRIC_STRIKE: metrics.epsilon_strikes(frame["strike"], target["strike"]),
            config.METRIC_PSIX: metrics.epsilon_psix(frame["psiX_main"], frame["psib"], frame["psia"]),
            config.METRIC_X2: metrics.epsilon_xpoints(frame["Xpt_sec"], target["Xpt_sec"]),
            config.METRIC_PSIX2: metrics.epsilon_psix(frame["psiX_sec"], frame["psib"], frame["psia"]),
        }
        eta = penalties.compute_eta(frame["lX"], 1, 1)
        mu = penalties.compute_mu(frame["Ip"], ip_ref, 1, 1)
        self._coil_mask *= float(penalties.compute_rho(frame["Icoil"], 1)[0])
        rho = np.asarray([self._coil_mask])
        topo = penalties.compute_topo_zero_mask(frame["nX"], 1, 1)
        # N=1 computes one step using the exact official weighting/masks.
        # The fixed full-episode 1/500 factor is applied by __call__.
        total, contributions, _ = scoring.task_score(eps, self._weights, config.EPSILON_MAX, 1, 1,
                                                     eta, mu, rho, topo)
        return total, contributions, eps, float(eta[0]), float(mu[0])

    def __call__(self, observation, action, terminated=False, truncated=False, info=None) -> tuple[float, bool]:
        info = info or {}
        if info.get("step") is not None and int(info["step"]) == 1:
            self.reset_state()
        if self._finished:
            raise RuntimeError("Reset the reward/environment before starting another episode")
        if self._target is None:
            self._target = _target_from_observation(observation)
            self._reference = _reference_observation(self._target)
        index = self._step
        self._step += 1
        self._timeout |= bool(info.get("timeout", False) or info.get("hfm_timeout", False))
        if info.get("inference_time_ms") is not None:
            duration = float(info["inference_time_ms"])
            if not np.isfinite(duration) or duration < 0:
                raise ValueError("inference_time_ms must be a finite nonnegative number")
            self._timing_over_steps += int(duration > config.INFER_TIME_LIMIT_MS)
        # Passing one value per over-limit step preserves the original gamma's
        # fixed-horizon denominator, including explicit measurements after N.
        gamma = penalties.compute_gamma(np.full(self._timing_over_steps, config.INFER_TIME_LIMIT_MS + 1.0), self.horizon)
        if self._timeout:
            gamma = 0.0
        step_score, eps, eta, mu = 0.0, {}, 1.0, 1.0
        if index < self.horizon:
            frame = extract_semifinal_trajectory_from_obs_list([observation], target_obs=self._reference)
            step_score, contributions, eps, eta, mu = self._step_scores(frame, index)
            for name, value in contributions.items():
                self._metric_totals[name] += value / self.horizon
        credited = {name: gamma * value for name, value in self._metric_totals.items()}
        deltas = {name: credited[name] - self._credited_metrics[name] for name in credited}
        reward = float(sum(deltas.values()))
        self._credited_metrics = credited
        self._finished = bool(terminated or truncated)
        ip_error = abs(_scalar(observation.get("Ip")) - float(self._ip_reference[min(index, self.horizon-1)]))
        coil = np.asarray(observation.get("I_PF", np.zeros(12)), dtype=np.float64).reshape(12)
        self.last_components = {
            "reward": reward,
            "competition_score": float(gamma * step_score / config.TOTAL_MAX_SCORE),
            "competition_step_score": float(step_score),
            "competition_total_score": float(sum(credited.values())),
            "competition_terminal_correction": reward - gamma * step_score / self.horizon,
            "gamma": float(gamma), "eta": eta, "rho": self._coil_mask,
            "ip_abs_error": ip_error,
            "ip_relative_error": float(eps[config.METRIC_IP][0]) if eps else 0.0,
            "lcfs_error_cm": float(eps[config.METRIC_LCFS][0]) if eps else 0.0,
            "coil_margin": float(np.max(np.abs(coil) / config.COIL_LIMITS_A)),
            "coil_current_exceeded": 1.0 - self._coil_mask,
            "ip_error_mask": mu, "soft_ip_error_mask": mu,
            "topology_score": float(_scalar(observation.get("lX")) == 1),
            "failure_penalty": 0.0,
            **{name: float(value) for name, value in deltas.items()},
        }
        component_names = {config.METRIC_IP: "ip_score", config.METRIC_LCFS: "lcfs_score",
                           config.METRIC_X: "main_x_dist_score", config.METRIC_PSIX: "main_x_psi_score",
                           config.METRIC_X2: "second_x_dist_score", config.METRIC_PSIX2: "second_x_psi_score",
                           config.METRIC_STRIKE: "strike_dist_score"}
        for name, component in component_names.items():
            self.last_components[component] = float(scoring.score_single_step(eps[name], config.EPSILON_MAX[name])[0]) if eps else 0.0
        return reward, bool(terminated)


# Older checkpoints/configs/tools may still import this public class name.
SemifinalDenseReward = CompetitionReward
