"""F2 XPT direct-target dense reward with stable X point matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from environment.xpt_utils import extract_xpt_observation_pack

from train.references import TargetReference

COIL_LIMITS_A = np.asarray([45_000.0] + [14_000.0] * 10 + [4_000.0], dtype=np.float64)


def _scalar(obs: dict[str, Any], key: str, default: float = 0.0) -> float:
    arr = np.asarray(obs.get(key, default), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float(default)
    value = float(arr[0])
    return value if np.isfinite(value) else float(default)


def _array(obs: dict[str, Any], key: str, shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
    out = np.full(int(np.prod(shape)), default, dtype=np.float64)
    value = obs.get(key)
    if value is not None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        n = min(out.size, flat.size)
        if n > 0:
            out[:n] = flat[:n]
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default).reshape(shape)


def _score(error: float, epsilon_max: float) -> float:
    if not np.isfinite(error):
        return 0.0
    return float(max(0.0, 1.0 - max(float(error), 0.0) / float(epsilon_max)))


@dataclass
class F2XPTTargetReward:
    """Dense F2 reward for direct XPT target training.

    LCFS, X point position, and X point flux are baseline-centered nonlinear
    quality scores. Keeping the initial value as the zero baseline makes moving
    away from the target negative without adding separate regression terms.
    """

    target: TargetReference
    reward_weights: dict[str, float] | None = None
    prev_action: np.ndarray | None = field(default=None, init=False)
    initial_lcfs_cm: float | None = field(default=None, init=False)
    initial_x_position_cm: float | None = field(default=None, init=False)
    initial_x_flux_error: float | None = field(default=None, init=False)
    last_components: dict[str, float] = field(default_factory=dict, init=False)

    def reset_state(self) -> None:
        self.prev_action = None
        self.initial_lcfs_cm = None
        self.initial_x_position_cm = None
        self.initial_x_flux_error = None
        self.last_components = {}

    def __call__(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> float:
        if terminated or _scalar(observation, "failure", 0.0) > 0.5:
            self.reset_state()
            failure_penalty = self._weight("failure_penalty", -100.0)
            self.last_components = {
                "reward": float(failure_penalty),
                "failure": 1.0,
                "failure_penalty": float(failure_penalty),
            }
            return float(failure_penalty)

        is_episode_start = bool(info and int(info.get("step", 1)) <= 1)
        if is_episode_start:
            self.prev_action = None

        ref_ip = _scalar(observation, "reference_Ip", self.target.ip_final)
        ip = _scalar(observation, "Ip", ref_ip)
        ip_abs_error = abs(ip - ref_ip)
        ip_rel = ip_abs_error / max(abs(ref_ip), 1.0)
        ip_score = _score(ip_rel, 0.05)
        ip_error_soft_limit = self._weight("ip_error_soft_limit_a", 40_000.0)
        ip_error_hard_limit = self._weight("ip_error_hard_limit_a", 50_000.0)
        ip_error_width = max(ip_error_hard_limit - ip_error_soft_limit, 1.0)
        ip_error_boundary_ratio = max(0.0, (ip_abs_error - ip_error_soft_limit) / ip_error_width)
        ip_error_boundary_penalty_raw = self._huber(ip_error_boundary_ratio)
        ip_error_boundary_penalty = min(
            ip_error_boundary_penalty_raw,
            self._weight("ip_error_boundary_penalty_cap", 4.0),
        )

        lcfs_cm = self._lcfs_distance_cm(observation)
        if is_episode_start or self.initial_lcfs_cm is None:
            self.initial_lcfs_cm = lcfs_cm
        lcfs_quality, lcfs_component, lcfs_relative_progress = self._relative_quality(
            current=lcfs_cm,
            initial=self.initial_lcfs_cm,
            good=self._weight("lcfs_good_cm", 3.0),
            min_scale=self._weight("lcfs_min_scale_cm", 1.0),
            slope=self._weight("lcfs_quality_slope", 4.0),
        )

        pack = self._xpt_pack(observation)
        x_count_score = self._x_count_score(pack)
        x_position_cm = self._x_position_distance_cm(pack)
        if is_episode_start or self.initial_x_position_cm is None:
            self.initial_x_position_cm = x_position_cm
        x_position_quality, x_position_component, x_position_relative_progress = self._relative_quality(
            current=x_position_cm,
            initial=self.initial_x_position_cm,
            good=self._weight("x_position_good_cm", 3.0),
            min_scale=self._weight("x_position_min_scale_cm", 1.0),
            slope=self._weight("x_position_quality_slope", 4.0),
        )

        x_flux_error = self._x_flux_error(pack)
        if is_episode_start or self.initial_x_flux_error is None:
            self.initial_x_flux_error = x_flux_error
        x_flux_quality, x_flux_component, x_flux_relative_progress = self._relative_quality(
            current=x_flux_error,
            initial=self.initial_x_flux_error,
            good=self._weight("x_flux_good", 0.003),
            min_scale=self._weight("x_flux_min_scale", 0.003),
            slope=self._weight("x_flux_quality_slope", 4.0),
        )

        target_lx = _scalar(observation, "reference_lX", 1.0)
        lx = _scalar(observation, "lX", target_lx)
        topology_score = 1.0 if int(round(lx)) == int(round(target_lx)) else 0.0

        ipf = _array(observation, "I_PF", (12,))
        coil_margin = np.max(np.abs(ipf) / COIL_LIMITS_A)
        coil_penalty = max(0.0, float(coil_margin) - 1.0)

        action_l2, action_smooth = self._action_penalties(action)
        action_l2_penalty = self._weight("action_l2_penalty", 0.0002) * action_l2
        action_smooth_penalty = self._weight("action_smooth_penalty", 0.001) * action_smooth

        score_values = {
            "ip_score": ip_score,
            "lcfs_component": lcfs_component,
            "x_count_score": x_count_score,
            "x_position_component": x_position_component,
            "x_flux_component": x_flux_component,
            "topology_score": topology_score,
        }
        weighted_score = 0.0
        score_weight_sum = 0.0
        for key, value in score_values.items():
            weight = self._weight(key, 0.0)
            if weight <= 0.0:
                continue
            weighted_score += weight * float(value)
            score_weight_sum += weight
        normalized_score = weighted_score / score_weight_sum if score_weight_sum > 0.0 else 0.0

        reward = (
            self._weight("alive_offset", 0.2)
            + self._weight("score_scale", 2.0) * normalized_score
            - self._weight("coil_penalty", 1.0) * coil_penalty
            - self._weight("ip_error_boundary_penalty", 1.0) * ip_error_boundary_penalty
            - action_l2_penalty
            - action_smooth_penalty
        )

        self.last_components = {
            "reward": float(reward),
            "failure": 0.0,
            "failure_penalty": 0.0,
            "ip_abs_error": float(ip_abs_error),
            "ip_score": float(ip_score),
            "ip_error_boundary_ratio": float(ip_error_boundary_ratio),
            "ip_error_boundary_penalty_raw": float(ip_error_boundary_penalty_raw),
            "ip_error_boundary_penalty": float(ip_error_boundary_penalty),
            "lcfs_cm": float(lcfs_cm),
            "lcfs_initial_cm": float(self.initial_lcfs_cm or 0.0),
            "lcfs_relative_progress": float(lcfs_relative_progress),
            "lcfs_quality": float(lcfs_quality),
            "lcfs_component": float(lcfs_component),
            "x_count_score": float(x_count_score),
            "x_position_cm": float(x_position_cm),
            "x_position_initial_cm": float(self.initial_x_position_cm or 0.0),
            "x_position_relative_progress": float(x_position_relative_progress),
            "x_position_quality": float(x_position_quality),
            "x_position_component": float(x_position_component),
            "x_flux_error": float(x_flux_error),
            "x_flux_initial_error": float(self.initial_x_flux_error or 0.0),
            "x_flux_relative_progress": float(x_flux_relative_progress),
            "x_flux_quality": float(x_flux_quality),
            "x_flux_component": float(x_flux_component),
            "topology_score": float(topology_score),
            "coil_penalty": float(coil_penalty),
            "action_l2": float(action_l2),
            "action_smooth": float(action_smooth),
            "action_l2_penalty": float(action_l2_penalty),
            "action_smooth_penalty": float(action_smooth_penalty),
        }
        return float(reward) if np.isfinite(reward) else -5.0

    def _weight(self, key: str, default: float) -> float:
        if not self.reward_weights:
            return float(default)
        return float(self.reward_weights.get(key, default))

    def _huber(self, ratio: float) -> float:
        ratio = max(0.0, float(ratio))
        delta = max(self._weight("ip_error_boundary_huber_delta", 1.0), 1e-6)
        if ratio <= delta:
            return ratio**2
        return 2.0 * delta * ratio - delta**2

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = float(np.clip(value, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-value)))

    def _relative_quality(
        self,
        *,
        current: float,
        initial: float,
        good: float,
        min_scale: float,
        slope: float,
    ) -> tuple[float, float, float]:
        scale = max(float(initial) - float(good), float(min_scale), 1e-6)
        relative_progress = (float(initial) - float(current)) / scale
        quality = self._sigmoid(float(slope) * relative_progress)
        baseline = self._sigmoid(0.0)
        return float(quality), float(quality - baseline), float(relative_progress)

    def _lcfs_distance_cm(self, observation: dict[str, Any]) -> float:
        lcfs = _array(observation, "lcfs_points", (32, 2))
        ref_lcfs = _array(observation, "reference_lcfs_points", (32, 2))
        return float(np.sqrt(np.mean(np.sum((lcfs - ref_lcfs) ** 2, axis=1))) * 100.0)

    def _xpt_pack(self, observation: dict[str, Any]) -> dict[str, Any]:
        ref_rx = _array(observation, "reference_rX", (4,))
        ref_zx = _array(observation, "reference_zX", (4,))
        ref_sr = _array(observation, "reference_strike_r", (8,))
        ref_sz = _array(observation, "reference_strike_z", (8,))
        try:
            return extract_xpt_observation_pack(observation, ref_rx, ref_zx, ref_sr, ref_sz)
        except Exception:
            return {
                "nX": int(_scalar(observation, "nX")),
                "x_r": np.zeros(4, dtype=np.float64),
                "x_z": np.zeros(4, dtype=np.float64),
                "x_valid": np.zeros(4, dtype=np.float64),
                "x_flux_diff": np.zeros(4, dtype=np.float64),
            }

    def _x_count_score(self, pack: dict[str, Any]) -> float:
        reference_nx = 4.0
        nx = float(pack["nX"])
        score = 1.0 - abs(nx - reference_nx) / reference_nx
        return float(np.clip(score, 0.0, 1.0))

    def _x_position_distance_cm(self, pack: dict[str, Any]) -> float:
        ref_r = np.asarray(self.target.rX, dtype=np.float64).reshape(4)
        ref_z = np.asarray(self.target.zX, dtype=np.float64).reshape(4)
        x_r = np.asarray(pack["x_r"], dtype=np.float64).reshape(4)
        x_z = np.asarray(pack["x_z"], dtype=np.float64).reshape(4)
        valid = np.asarray(pack["x_valid"], dtype=np.float64).reshape(4) > 0.5
        distances = np.full(4, self._weight("x_position_missing_cm", 30.0), dtype=np.float64)
        if np.any(valid):
            distances[valid] = np.sqrt((x_r[valid] - ref_r[valid]) ** 2 + (x_z[valid] - ref_z[valid]) ** 2) * 100.0
        return float(np.mean(distances))

    def _x_flux_error(self, pack: dict[str, Any]) -> float:
        valid = np.asarray(pack["x_valid"], dtype=np.float64).reshape(4) > 0.5
        errors = np.full(4, self._weight("x_flux_missing_error", 0.02), dtype=np.float64)
        flux = np.asarray(pack["x_flux_diff"], dtype=np.float64).reshape(4)
        if np.any(valid):
            errors[valid] = np.abs(flux[valid])
        return float(np.mean(errors))

    def _action_penalties(self, action: np.ndarray) -> tuple[float, float]:
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        action_l2 = float(np.mean((action_arr / 500.0) ** 2)) if action_arr.size else 0.0
        if self.prev_action is None or self.prev_action.shape != action_arr.shape:
            smooth = 0.0
        else:
            smooth = float(np.mean(((action_arr - self.prev_action) / 500.0) ** 2))
        self.prev_action = action_arr.copy()
        return action_l2, smooth


@dataclass
class F2OfficialProxyReward(F2XPTTargetReward):
    """F2 reward proxy aligned to the public scoring formula.

    Per-step reward:
        r_k = eta_k * mu_k * rho_k * r_norm_k
              - lambda_fail * (N - k) / N * 1_failure

    ``r_norm_k`` is the official weighted per-step F2 score divided by N, so a
    perfect 500-step rollout has return 30. XPT-only metrics are masked by
    ``nX == 4``. ``rho`` latches to zero after the first coil over-limit step.
    """

    coil_overlimit_latched: bool = field(default=False, init=False)

    def reset_state(self) -> None:
        super().reset_state()
        self.coil_overlimit_latched = False

    def __call__(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> float:
        step = self._step_index(info)
        total_steps = max(int(self._weight("total_steps", 500.0)), 1)
        is_episode_start = step <= 1
        if is_episode_start:
            self.prev_action = None
            self.coil_overlimit_latched = False

        failure = bool(terminated or _scalar(observation, "failure", 0.0) > 0.5)
        if failure:
            remaining = max(0.0, (float(total_steps) - float(step)) / float(total_steps))
            failure_penalty = -self._weight("lambda_fail", 3.0) * remaining
            self.reset_state()
            self.last_components = {
                "reward": float(failure_penalty),
                "failure": 1.0,
                "failure_penalty": float(failure_penalty),
                "failure_remaining_fraction": float(remaining),
                "step_index": float(step),
            }
            return float(failure_penalty)

        ref_ip = _scalar(observation, "reference_Ip", self.target.ip_final)
        ip = _scalar(observation, "Ip", ref_ip)
        ip_abs_error = abs(ip - ref_ip)
        ip_rel = ip_abs_error / max(abs(ref_ip), 1.0)
        ip_score = _score(ip_rel, self._weight("ip_epsilon", 0.05))

        lcfs_cm = self._lcfs_distance_cm(observation)
        lcfs_score = _score(lcfs_cm, self._weight("lcfs_epsilon_cm", 3.0))

        pack = self._xpt_pack(observation)
        main_slots = np.asarray([1, 2], dtype=np.int64)
        secondary_slots = np.asarray([0, 3], dtype=np.int64)
        x_main_cm, x_main_score = self._x_position_group_score(
            pack,
            main_slots,
            missing_cm=self._weight("x_main_missing_cm", 3.0),
            epsilon_cm=self._weight("x_epsilon_cm", 3.0),
        )
        x_secondary_cm, x_secondary_score = self._x_position_group_score(
            pack,
            secondary_slots,
            missing_cm=self._weight("x_secondary_missing_cm", 5.0),
            epsilon_cm=self._weight("x2_epsilon_cm", 5.0),
        )
        psix_main_error, psix_main_score = self._x_flux_group_score(
            pack,
            main_slots,
            missing_error=self._weight("psix_main_missing_error", 0.003),
            epsilon=self._weight("psix_epsilon", 0.003),
        )
        psix_secondary_error, psix_secondary_score = self._x_flux_group_score(
            pack,
            secondary_slots,
            missing_error=self._weight("psix_secondary_missing_error", 0.005),
            epsilon=self._weight("psix2_epsilon", 0.005),
        )
        strike_cm, strike_score = self._strike_score(pack)

        target_lx = _scalar(observation, "reference_lX", 1.0)
        lx = _scalar(observation, "lX", target_lx)
        eta = 1.0 if int(round(lx)) == int(round(target_lx)) else 0.5

        current_fuse_a = self._weight("current_fuse_a", 50_000.0)
        mu = 1.0 if ip_abs_error <= current_fuse_a else 0.0

        ipf = _array(observation, "I_PF", (12,))
        coil_margin = float(np.max(np.abs(ipf) / COIL_LIMITS_A))
        coil_overlimit = coil_margin > 1.0
        if coil_overlimit:
            self.coil_overlimit_latched = True
        rho = 0.0 if self.coil_overlimit_latched else 1.0

        target_x_valid = self._target_x_valid()
        reference_nx = int(np.sum(target_x_valid))
        nx = int(pack.get("nX", 0))
        nx_score = 1.0 if nx == reference_nx else 0.0
        xpt_mask = 1.0 if nx == 4 else 0.0

        weighted_step_score = (
            self._weight("official_ip", 6.0) * ip_score
            + self._weight("official_lcfs", 6.0) * lcfs_score
            + xpt_mask
            * (
                self._weight("official_x", 4.0) * x_main_score
                + self._weight("official_strike", 3.0) * strike_score
                + self._weight("official_psix", 4.0) * psix_main_score
                + self._weight("official_x2", 3.0) * x_secondary_score
                + self._weight("official_psix2", 4.0) * psix_secondary_score
            )
        )
        r_norm = weighted_step_score / 30
        reward = eta * mu * rho * r_norm

        action_l2, action_smooth = self._action_penalties(action)
        self.last_components = {
            "reward": float(reward),
            "failure": 0.0,
            "failure_penalty": 0.0,
            "step_index": float(step),
            "r_norm": float(r_norm),
            "weighted_step_score": float(weighted_step_score),
            "eta": float(eta),
            "mu": float(mu),
            "rho": float(rho),
            "official_like_score": float(weighted_step_score / max(self._weight("official_total", 30.0), 1e-9)),
            "ip_abs_error": float(ip_abs_error),
            "ip_relative_error": float(ip_rel),
            "ip_score": float(ip_score),
            "lcfs_cm": float(lcfs_cm),
            "lcfs_score": float(lcfs_score),
            "x_position_cm": float(x_main_cm),
            "x_score": float(x_main_score),
            "x_main_cm": float(x_main_cm),
            "x_main_score": float(x_main_score),
            "x_secondary_cm": float(x_secondary_cm),
            "x_secondary_score": float(x_secondary_score),
            "x_count_score": float(nx_score),
            "nx_score": float(nx_score),
            "xpt_mask": float(xpt_mask),
            "strike_cm": float(strike_cm),
            "strike_score": float(strike_score),
            "x_flux_error": float(psix_main_error),
            "psix_score": float(psix_main_score),
            "psix_main_error": float(psix_main_error),
            "psix_main_score": float(psix_main_score),
            "psix_secondary_error": float(psix_secondary_error),
            "psix_secondary_score": float(psix_secondary_score),
            "topology_score": float(eta),
            "coil_margin": float(coil_margin),
            "coil_overlimit": float(coil_overlimit),
            "coil_overlimit_latched": float(self.coil_overlimit_latched),
            "action_l2": float(action_l2),
            "action_smooth": float(action_smooth),
        }
        return float(reward) if np.isfinite(reward) else 0.0

    @staticmethod
    def _step_index(info: dict[str, Any] | None) -> int:
        if not info:
            return 1
        for key in ("step", "timestep", "time_step"):
            if key in info:
                try:
                    return max(1, int(info[key]))
                except Exception:
                    pass
        return 1

    def _target_x_valid(self) -> np.ndarray:
        if self.target.x_valid is None:
            return np.ones(4, dtype=bool)
        return np.asarray(self.target.x_valid, dtype=np.float64).reshape(4) > 0.5

    def _target_strike_valid(self) -> np.ndarray:
        if self.target.strike_valid is None:
            return np.ones(8, dtype=bool)
        return np.asarray(self.target.strike_valid, dtype=np.float64).reshape(8) > 0.5

    def _x_position_group_score(
        self,
        pack: dict[str, Any],
        slots: np.ndarray,
        *,
        missing_cm: float,
        epsilon_cm: float,
    ) -> tuple[float, float]:
        ref_r = np.asarray(self.target.rX, dtype=np.float64).reshape(4)
        ref_z = np.asarray(self.target.zX, dtype=np.float64).reshape(4)
        x_r = np.asarray(pack.get("x_r", np.zeros(4)), dtype=np.float64).reshape(4)
        x_z = np.asarray(pack.get("x_z", np.zeros(4)), dtype=np.float64).reshape(4)
        x_valid = np.asarray(pack.get("x_valid", np.zeros(4)), dtype=np.float64).reshape(4) > 0.5
        target_valid = self._target_x_valid()
        distances: list[float] = []
        for slot in slots:
            idx = int(slot)
            if not target_valid[idx]:
                continue
            if x_valid[idx]:
                distance = np.sqrt((x_r[idx] - ref_r[idx]) ** 2 + (x_z[idx] - ref_z[idx]) ** 2) * 100.0
                distances.append(float(distance))
            else:
                distances.append(float(missing_cm))
        error_cm = float(np.mean(distances)) if distances else float(missing_cm)
        return error_cm, _score(error_cm, epsilon_cm)

    def _x_flux_group_score(
        self,
        pack: dict[str, Any],
        slots: np.ndarray,
        *,
        missing_error: float,
        epsilon: float,
    ) -> tuple[float, float]:
        flux = np.asarray(pack.get("x_flux_diff", np.zeros(4)), dtype=np.float64).reshape(4)
        x_valid = np.asarray(pack.get("x_valid", np.zeros(4)), dtype=np.float64).reshape(4) > 0.5
        target_valid = self._target_x_valid()
        errors: list[float] = []
        for slot in slots:
            idx = int(slot)
            if not target_valid[idx]:
                continue
            errors.append(float(abs(flux[idx])) if x_valid[idx] else float(missing_error))
        error = float(np.mean(errors)) if errors else float(missing_error)
        return error, _score(error, epsilon)

    def _strike_score(self, pack: dict[str, Any]) -> tuple[float, float]:
        ref_r = np.asarray(self.target.strike_r, dtype=np.float64).reshape(8)
        ref_z = np.asarray(self.target.strike_z, dtype=np.float64).reshape(8)
        strike_r = np.asarray(pack.get("strike_r", np.zeros(8)), dtype=np.float64).reshape(8)
        strike_z = np.asarray(pack.get("strike_z", np.zeros(8)), dtype=np.float64).reshape(8)
        strike_valid = np.asarray(pack.get("strike_valid", np.zeros(8)), dtype=np.float64).reshape(8) > 0.5
        target_valid = self._target_strike_valid()
        valid = strike_valid & target_valid
        missing_cm = self._weight("strike_missing_cm", 10.0)
        if np.any(valid):
            distances = np.sqrt((strike_r[valid] - ref_r[valid]) ** 2 + (strike_z[valid] - ref_z[valid]) ** 2) * 100.0
            error_cm = float(np.mean(distances))
        else:
            error_cm = float(missing_cm)
        return error_cm, _score(error_cm, self._weight("strike_epsilon_cm", 10.0))

