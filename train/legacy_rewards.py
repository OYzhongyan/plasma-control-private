"""Dense rewards aligned with the public semifinal scoring script."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from environment.xpt_utils import extract_xpt_observation_pack
from train.references import TargetReference

COIL_LIMITS_A = np.asarray([45000.0] + [14000.0] * 10 + [4000.0], dtype=np.float64)


def _scalar(obs: dict[str, Any], key: str, default: float = 0.0) -> float:
    arr = np.asarray(obs.get(key, default), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float(default)
    value = float(arr[0])
    return value if np.isfinite(value) else float(default)


def _array(
    obs: dict[str, Any], key: str, shape: tuple[int, ...], default: float = 0.0
) -> np.ndarray:
    out = np.full(int(np.prod(shape)), default, dtype=np.float64)
    value = obs.get(key)
    if value is not None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        n = min(out.size, flat.size)
        if n > 0:
            out[:n] = flat[:n]
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default).reshape(
        shape
    )


def _score(error: float, epsilon_max: float, lower_bound: bool = True) -> float:
    if not np.isfinite(error):
        return 0.0
    if lower_bound:
        return float(max(0.0, 1.0 - max(float(error), 0.0) / float(epsilon_max)))
    else:
        return float(1.0 - max(float(error), 0.0) / float(epsilon_max))


@dataclass
class SemifinalDenseReward:
    """Stateful per-env dense reward for F2b PPO training."""

    target: TargetReference
    action_smooth_weight: float = 0.002
    action_l2_weight: float = 0.0005
    reward_weights: dict[str, float] | None = None
    prev_action: np.ndarray | None = field(default=None, init=False)
    initial_lcfs_cm: float | None = field(default=None, init=False)
    initial_x_position_cm: float | None = field(default=None, init=False)
    last_components: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.reward_weights:
            self.action_smooth_weight = float(
                self.reward_weights.get(
                    "action_smooth_penalty", self.action_smooth_weight
                )
            )
            self.action_l2_weight = float(
                self.reward_weights.get("action_l2_penalty", self.action_l2_weight)
            )

    def reset_state(self) -> None:
        self.prev_action = None
        self.initial_lcfs_cm = None
        self.initial_x_position_cm = None
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
            failure_penalty = self._weight("failure_penalty", -5.0)
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
        ip_score = self._ip_score(ip_rel)
        ip_error_soft_limit = self._weight("ip_error_soft_limit_a", 40000.0)
        ip_error_hard_limit = self._weight("ip_error_hard_limit_a", 50000.0)
        ip_safe_score = _score(ip_abs_error, max(ip_error_hard_limit, 1.0))
        ip_error_width = max(ip_error_hard_limit - ip_error_soft_limit, 1.0)
        ip_error_boundary_ratio = max(
            0.0, (ip_abs_error - ip_error_soft_limit) / ip_error_width
        )
        ip_error_boundary_penalty_raw = self._ip_error_boundary_penalty(
            ip_error_boundary_ratio
        )
        ip_error_boundary_penalty = self._apply_penalty_cap(
            ip_error_boundary_penalty_raw, "ip_error_boundary_penalty_cap"
        )
        ref_r = _scalar(observation, "reference_R", _scalar(observation, "R", 0.0))
        r = _scalar(observation, "R", ref_r)
        r_score = _score(abs(r - ref_r), 0.03)
        ref_z = _scalar(observation, "reference_Z", _scalar(observation, "Z", 0.0))
        z = _scalar(observation, "Z", ref_z)
        z_score = _score(abs(z - ref_z), 0.03)
        lcfs = _array(observation, "lcfs_points", (32, 2))
        ref_lcfs = _array(observation, "reference_lcfs_points", (32, 2))
        lcfs_cm = float(
            np.sqrt(np.mean(np.sum((lcfs - ref_lcfs) ** 2, axis=1))) * 100.0
        )
        if is_episode_start or self.initial_lcfs_cm is None:
            self.initial_lcfs_cm = lcfs_cm
        lcfs_score, lcfs_progress_score, lcfs_precision_score = self._lcfs_score(
            lcfs_cm
        )
        lcfs_absolute_score = lcfs_score
        (
            lcfs_regression_cm,
            lcfs_regression_ratio,
            lcfs_regression_penalty_raw,
            lcfs_regression_penalty,
        ) = self._lcfs_regression_penalty(lcfs_cm)
        default_lx = 1.0
        target_lx = _scalar(observation, "reference_lX", default_lx)
        lx = _scalar(observation, "lX", target_lx)
        mismatch_score = 0.0 if self.reward_weights else 0.5
        topology_score = (
            1.0 if int(round(lx)) == int(round(target_lx)) else mismatch_score
        )
        limiter_transition_score = topology_score * lcfs_absolute_score * ip_safe_score
        x_components = self._xpt_scores(observation, is_episode_start=is_episode_start)
        x_score = x_components["x_score"]
        strike_score = x_components["strike_score"]
        psix_score = x_components["psix_score"]
        nx_score = x_components["nx_score"]
        base_score = (
            0.2 * ip_score
            + 0.2 * lcfs_absolute_score
            + 0.22 * x_score
            + 0.14 * strike_score
            + 0.24 * psix_score
        )
        ipf = _array(observation, "I_PF", (12,))
        coil_margin = np.max(np.abs(ipf) / COIL_LIMITS_A)
        coil_penalty = max(0.0, float(coil_margin) - 1.0)
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        action_l2 = (
            float(np.mean((action_arr / 500.0) ** 2)) if action_arr.size else 0.0
        )
        if self.prev_action is None or self.prev_action.shape != action_arr.shape:
            smooth = 0.0
        else:
            smooth = float(np.mean(((action_arr - self.prev_action) / 500.0) ** 2))
        self.prev_action = action_arr.copy()
        action_l2_penalty = self.action_l2_weight * action_l2
        action_smooth_penalty = self.action_smooth_weight * smooth
        if self.reward_weights:
            score_values = {
                "ip_score": ip_score,
                "r_score": r_score,
                "z_score": z_score,
                "lcfs_absolute_score": lcfs_absolute_score,
                "topology_score": topology_score,
                "limiter_transition_score": limiter_transition_score,
                "x_score": x_score,
                "strike_score": strike_score,
                "psix_score": psix_score,
                "nx_score": nx_score,
            }
            score_weight_sum = 0.0
            weighted_score = 0.0
            for key, score_value in score_values.items():
                weight = float(self.reward_weights.get(key, 0.0))
                if weight <= 0.0:
                    continue
                weighted_score += weight * float(score_value)
                score_weight_sum += weight
            normalized_score = (
                weighted_score / score_weight_sum
                if score_weight_sum > 0
                else base_score
            )
            reward = (
                self._weight("alive_offset", -1.0)
                + self._weight("score_scale", 2.0) * normalized_score
                - self._weight("coil_penalty", 0.5) * coil_penalty
                - self._weight("ip_error_boundary_penalty", 0.0)
                * ip_error_boundary_penalty
                - self._weight("lcfs_regression_penalty", 0.0) * lcfs_regression_penalty
                - self._weight("x_position_regression_penalty", 0.0)
                * x_components["x_position_regression_penalty"]
                - action_l2_penalty
                - action_smooth_penalty
            )
        else:
            reward = (
                2.0 * base_score * topology_score
                - 1.0
                - 0.5 * coil_penalty
                - action_l2_penalty
                - action_smooth_penalty
            )
            reward -= 0.25 * (1.0 - nx_score)
        self.last_components = {
            "reward": float(reward),
            "failure_penalty": 0.0,
            "ip_abs_error": float(ip_abs_error),
            "ip_relative_error": float(ip_rel),
            "ip_safe_score": float(ip_safe_score),
            "ip_score": float(ip_score),
            "ip_error_boundary_ratio": float(ip_error_boundary_ratio),
            "ip_error_boundary_penalty_raw": float(ip_error_boundary_penalty_raw),
            "ip_error_boundary_penalty": float(ip_error_boundary_penalty),
            "r_score": float(r_score),
            "z_score": float(z_score),
            "lcfs_cm": float(lcfs_cm),
            "lcfs_initial_cm": float(self.initial_lcfs_cm or 0.0),
            "lcfs_progress_score": float(lcfs_progress_score),
            "lcfs_precision_score": float(lcfs_precision_score),
            "lcfs_regression_cm": float(lcfs_regression_cm),
            "lcfs_regression_ratio": float(lcfs_regression_ratio),
            "lcfs_regression_penalty_raw": float(lcfs_regression_penalty_raw),
            "lcfs_regression_penalty": float(lcfs_regression_penalty),
            "lcfs_score": float(lcfs_score),
            "lcfs_absolute_score": float(lcfs_absolute_score),
            "x_score": float(x_score),
            "x_position_cm": float(x_components["x_position_cm"]),
            "x_position_initial_cm": float(x_components["x_position_initial_cm"]),
            "x_position_progress_score": float(
                x_components["x_position_progress_score"]
            ),
            "x_position_precision_score": float(
                x_components["x_position_precision_score"]
            ),
            "x_position_regression_cm": float(x_components["x_position_regression_cm"]),
            "x_position_regression_ratio": float(
                x_components["x_position_regression_ratio"]
            ),
            "x_position_regression_penalty_raw": float(
                x_components["x_position_regression_penalty_raw"]
            ),
            "x_position_regression_penalty": float(
                x_components["x_position_regression_penalty"]
            ),
            "x_valid_count": float(x_components["x_valid_count"]),
            "strike_score": float(strike_score),
            "psix_score": float(psix_score),
            "nx_score": float(nx_score),
            "topology_score": float(topology_score),
            "limiter_transition_score": float(limiter_transition_score),
            "coil_penalty": float(coil_penalty),
            "action_l2": float(action_l2),
            "action_smooth": float(smooth),
            "action_l2_penalty": float(action_l2_penalty),
            "action_smooth_penalty": float(action_smooth_penalty),
        }
        return float(reward) if np.isfinite(reward) else -5.0

    def _weight(self, key: str, default: float) -> float:
        if not self.reward_weights:
            return float(default)
        return float(self.reward_weights.get(key, default))

    def _option(self, key: str, default: Any) -> Any:
        if not self.reward_weights:
            return default
        return self.reward_weights.get(key, default)

    def _ip_score(self, ip_relative_error: float) -> float:
        shape = str(self._option("ip_score_shape", "linear")).strip().lower()
        error = max(0.0, float(ip_relative_error))
        if shape in {"linear", "official"}:
            return _score(error, 0.05)
        if shape == "exp_relative":
            sigma = max(float(self._option("ip_relative_sigma", 0.03)), 1e-06)
            power = max(float(self._option("ip_relative_power", 2.0)), 1e-06)
            return float(np.exp(-((error / sigma) ** power)))
        raise ValueError(f"Unsupported ip_score_shape: {shape!r}")

    def _ip_error_boundary_penalty(self, ratio: float) -> float:
        ratio = max(0.0, float(ratio))
        shape = (
            str(self._option("ip_error_boundary_penalty_shape", "square"))
            .strip()
            .lower()
        )
        if shape in {"huber", "huber_cap"}:
            delta = max(
                float(self._option("ip_error_boundary_huber_delta", 1.0)), 1e-06
            )
            if ratio <= delta:
                return ratio**2
            return 2.0 * delta * ratio - delta**2
        if shape in {"l1", "linear"}:
            return ratio
        if shape != "square":
            raise ValueError(f"Unsupported ip_error_boundary_penalty_shape: {shape!r}")
        return ratio**2

    def _apply_penalty_cap(self, value: float, cap_key: str) -> float:
        cap = self._option(cap_key, None)
        if cap is None:
            return float(value)
        cap_value = float(cap)
        if not np.isfinite(cap_value) or cap_value <= 0.0:
            return float(value)
        return float(min(float(value), cap_value))

    def _lcfs_score(self, lcfs_cm: float) -> tuple[float, float, float]:
        precision_cm = max(float(self._option("lcfs_precision_cm", 3.0)), 1e-06)
        precision_score = self._precision_score_cm(
            lcfs_cm,
            precision_cm=precision_cm,
            shape_key="lcfs_precision_shape",
            sigma_key="lcfs_precision_sigma_cm",
            power_key="lcfs_precision_power",
        )
        mode = str(self._option("lcfs_score_mode", "absolute")).strip().lower()
        if mode in {"absolute", "precision"}:
            return (precision_score, 0.0, precision_score)
        if mode not in {"progress_precision", "progress"}:
            raise ValueError(f"Unsupported lcfs_score_mode: {mode!r}")
        initial_cm = float(
            self.initial_lcfs_cm if self.initial_lcfs_cm is not None else lcfs_cm
        )
        min_scale = max(float(self._option("lcfs_progress_min_scale_cm", 1.0)), 1e-06)
        progress_scale = max(initial_cm - precision_cm, min_scale)
        progress_score = float(
            np.clip((initial_cm - lcfs_cm) / progress_scale, 0.0, 1.0)
        )
        if mode == "progress":
            return (progress_score, progress_score, precision_score)
        progress_weight = max(float(self._option("lcfs_progress_weight", 0.7)), 0.0)
        precision_weight = max(float(self._option("lcfs_precision_weight", 0.3)), 0.0)
        total = progress_weight + precision_weight
        if total <= 0.0:
            return (precision_score, progress_score, precision_score)
        score = (
            progress_weight * progress_score + precision_weight * precision_score
        ) / total
        return (float(np.clip(score, 0.0, 1.0)), progress_score, precision_score)

    def _precision_score_cm(
        self,
        value_cm: float,
        *,
        precision_cm: float,
        shape_key: str,
        sigma_key: str,
        power_key: str,
    ) -> float:
        value_cm = max(0.0, float(value_cm))
        shape = str(self._option(shape_key, "linear")).strip().lower()
        if shape in {"linear", "official"}:
            return _score(value_cm, precision_cm)
        if shape == "exp":
            sigma_cm = max(float(self._option(sigma_key, precision_cm)), 1e-06)
            power = max(float(self._option(power_key, 2.0)), 1e-06)
            return float(np.exp(-((value_cm / sigma_cm) ** power)))
        raise ValueError(f"Unsupported {shape_key}: {shape!r}")

    def _lcfs_regression_penalty(
        self, lcfs_cm: float
    ) -> tuple[float, float, float, float]:
        initial_cm = float(
            self.initial_lcfs_cm if self.initial_lcfs_cm is not None else lcfs_cm
        )
        margin_cm = max(float(self._option("lcfs_regression_margin_cm", 1.0)), 0.0)
        scale_cm = max(float(self._option("lcfs_regression_scale_cm", 5.0)), 1e-06)
        regression_cm = max(0.0, float(lcfs_cm) - initial_cm - margin_cm)
        ratio = regression_cm / scale_cm
        raw = self._shaped_penalty(
            ratio,
            shape_key="lcfs_regression_penalty_shape",
            delta_key="lcfs_regression_huber_delta",
            default_shape="huber",
        )
        penalty = self._apply_penalty_cap(raw, "lcfs_regression_penalty_cap")
        return (float(regression_cm), float(ratio), float(raw), float(penalty))

    def _shaped_penalty(
        self,
        ratio: float,
        *,
        shape_key: str,
        delta_key: str,
        default_shape: str = "square",
    ) -> float:
        ratio = max(0.0, float(ratio))
        shape = str(self._option(shape_key, default_shape)).strip().lower()
        if shape in {"huber", "huber_cap"}:
            delta = max(float(self._option(delta_key, 1.0)), 1e-06)
            if ratio <= delta:
                return ratio**2
            return 2.0 * delta * ratio - delta**2
        if shape in {"l1", "linear"}:
            return ratio
        if shape != "square":
            raise ValueError(f"Unsupported penalty shape for {shape_key}: {shape!r}")
        return ratio**2

    def _progress_precision_score(
        self,
        value_cm: float,
        *,
        mode_key: str,
        precision_key: str,
        progress_min_scale_key: str,
        progress_weight_key: str,
        precision_weight_key: str,
        initial_value_cm: float | None,
        default_precision_cm: float = 3.0,
    ) -> tuple[float, float, float]:
        precision_cm = max(
            float(self._option(precision_key, default_precision_cm)), 1e-06
        )
        precision_score = _score(value_cm, precision_cm)
        mode = str(self._option(mode_key, "absolute")).strip().lower()
        if mode in {"absolute", "precision"}:
            return (precision_score, 0.0, precision_score)
        if mode not in {"progress_precision", "progress"}:
            raise ValueError(f"Unsupported {mode_key}: {mode!r}")
        initial_cm = float(
            initial_value_cm if initial_value_cm is not None else value_cm
        )
        min_scale = max(float(self._option(progress_min_scale_key, 1.0)), 1e-06)
        progress_scale = max(initial_cm - precision_cm, min_scale)
        progress_score = float(
            np.clip((initial_cm - value_cm) / progress_scale, 0.0, 1.0)
        )
        if mode == "progress":
            return (progress_score, progress_score, precision_score)
        progress_weight = max(float(self._option(progress_weight_key, 0.7)), 0.0)
        precision_weight = max(float(self._option(precision_weight_key, 0.3)), 0.0)
        total = progress_weight + precision_weight
        if total <= 0.0:
            return (precision_score, progress_score, precision_score)
        score = (
            progress_weight * progress_score + precision_weight * precision_score
        ) / total
        return (float(np.clip(score, 0.0, 1.0)), progress_score, precision_score)

    def _relative_regression_penalty(
        self,
        value_cm: float,
        *,
        initial_value_cm: float | None,
        margin_key: str,
        scale_key: str,
        shape_key: str,
        delta_key: str,
        cap_key: str,
    ) -> tuple[float, float, float, float]:
        initial_cm = float(
            initial_value_cm if initial_value_cm is not None else value_cm
        )
        margin_cm = max(float(self._option(margin_key, 1.0)), 0.0)
        scale_cm = max(float(self._option(scale_key, 5.0)), 1e-06)
        regression_cm = max(0.0, float(value_cm) - initial_cm - margin_cm)
        ratio = regression_cm / scale_cm
        raw = self._shaped_penalty(
            ratio, shape_key=shape_key, delta_key=delta_key, default_shape="huber"
        )
        penalty = self._apply_penalty_cap(raw, cap_key)
        return (float(regression_cm), float(ratio), float(raw), float(penalty))

    def _empty_x_components(self) -> dict[str, float]:
        return {
            "x_score": 0.0,
            "strike_score": 0.0,
            "psix_score": 0.0,
            "nx_score": 0.0,
            "x_position_cm": 0.0,
            "x_position_initial_cm": 0.0,
            "x_position_progress_score": 0.0,
            "x_position_precision_score": 0.0,
            "x_position_regression_cm": 0.0,
            "x_position_regression_ratio": 0.0,
            "x_position_regression_penalty_raw": 0.0,
            "x_position_regression_penalty": 0.0,
            "x_valid_count": 0.0,
        }

    def _xpt_scores(
        self, observation: dict[str, Any], *, is_episode_start: bool = False
    ) -> dict[str, float]:
        if not self.target.has_xpt:
            return self._empty_x_components()
        try:
            pack = extract_xpt_observation_pack(
                observation,
                self.target.rX,
                self.target.zX,
                self.target.strike_r,
                self.target.strike_z,
                fx_order="C",
            )
        except Exception:
            return self._empty_x_components()
        target_x_valid = np.asarray(self.target.x_valid, dtype=np.float64) > 0.5
        x_valid = (pack["x_valid"] > 0.5) & target_x_valid
        if np.any(x_valid):
            x_cm = (
                np.sqrt(
                    (pack["x_r"][x_valid] - self.target.rX[x_valid]) ** 2
                    + (pack["x_z"][x_valid] - self.target.zX[x_valid]) ** 2
                )
                * 100.0
            )
            x_position_cm = float(np.mean(x_cm))
        else:
            x_position_cm = float(self._option("x_position_missing_cm", 30.0))
        if is_episode_start or self.initial_x_position_cm is None:
            self.initial_x_position_cm = x_position_cm
        x_score, x_progress_score, x_precision_score = self._progress_precision_score(
            x_position_cm,
            mode_key="x_position_score_mode",
            precision_key="x_position_precision_cm",
            progress_min_scale_key="x_position_progress_min_scale_cm",
            progress_weight_key="x_position_progress_weight",
            precision_weight_key="x_position_precision_weight",
            initial_value_cm=self.initial_x_position_cm,
            default_precision_cm=3.0,
        )
        (
            x_regression_cm,
            x_regression_ratio,
            x_regression_penalty_raw,
            x_regression_penalty,
        ) = self._relative_regression_penalty(
            x_position_cm,
            initial_value_cm=self.initial_x_position_cm,
            margin_key="x_position_regression_margin_cm",
            scale_key="x_position_regression_scale_cm",
            shape_key="x_position_regression_penalty_shape",
            delta_key="x_position_regression_huber_delta",
            cap_key="x_position_regression_penalty_cap",
        )
        target_strike_valid = (
            np.asarray(self.target.strike_valid, dtype=np.float64) > 0.5
        )
        strike_valid = (pack["strike_valid"] > 0.5) & target_strike_valid
        if np.any(strike_valid):
            strike_cm = (
                np.sqrt(
                    (
                        pack["strike_r"][strike_valid]
                        - self.target.strike_r[strike_valid]
                    )
                    ** 2
                    + (
                        pack["strike_z"][strike_valid]
                        - self.target.strike_z[strike_valid]
                    )
                    ** 2
                )
                * 100.0
            )
            strike_score = _score(float(np.mean(strike_cm)), 10.0)
        else:
            strike_score = 0.0
        psix = (
            float(np.mean(np.abs(pack["x_flux_diff"][target_x_valid])))
            if np.any(target_x_valid)
            else 1.0
        )
        psix_score = _score(psix, 0.003)
        nx_score = 1.0 if int(pack["nX"]) == int(np.sum(target_x_valid)) else 0.0
        return {
            "x_score": float(x_score),
            "strike_score": float(strike_score),
            "psix_score": float(psix_score),
            "nx_score": float(nx_score),
            "x_position_cm": float(x_position_cm),
            "x_position_initial_cm": float(self.initial_x_position_cm or 0.0),
            "x_position_progress_score": float(x_progress_score),
            "x_position_precision_score": float(x_precision_score),
            "x_position_regression_cm": float(x_regression_cm),
            "x_position_regression_ratio": float(x_regression_ratio),
            "x_position_regression_penalty_raw": float(x_regression_penalty_raw),
            "x_position_regression_penalty": float(x_regression_penalty),
            "x_valid_count": float(np.sum(x_valid)),
        }


@dataclass
class SemifinalDenseRewardV2(SemifinalDenseReward):
    """Stateful per-env dense reward for F2b PPO training (Version 2)."""

    coil_max_current_mask: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()

    def reset_state(self) -> None:
        super().reset_state()
        self.coil_max_current_mask = 1.0

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
            failure_penalty = self._weight("failure_penalty", -5.0)
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
        ip_error_soft_limit = self._weight("ip_error_soft_limit_a", 40000.0)
        ip_error_hard_limit = self._weight("ip_error_hard_limit_a", 50000.0)
        ip_safe_score = _score(ip_abs_error, max(ip_error_hard_limit, 1.0))
        ip_error_width = max(ip_error_hard_limit - ip_error_soft_limit, 1.0)
        ip_error_boundary_ratio = max(
            0.0, (ip_abs_error - ip_error_soft_limit) / ip_error_width
        )
        ip_error_boundary_penalty_raw = self._ip_error_boundary_penalty(
            ip_error_boundary_ratio
        )
        ip_error_boundary_penalty = self._apply_penalty_cap(
            ip_error_boundary_penalty_raw, "ip_error_boundary_penalty_cap"
        )
        ref_r = _scalar(observation, "reference_R", _scalar(observation, "R", 0.0))
        r = _scalar(observation, "R", ref_r)
        r_score = _score(abs(r - ref_r), 0.03)
        ref_z = _scalar(observation, "reference_Z", _scalar(observation, "Z", 0.0))
        z = _scalar(observation, "Z", ref_z)
        z_score = _score(abs(z - ref_z), 0.03)
        lcfs = _array(observation, "lcfs_points", (32, 2))
        ref_lcfs = _array(observation, "reference_lcfs_points", (32, 2))
        lcfs_error_cm = float(
            np.sqrt(np.mean(np.sum((lcfs - ref_lcfs) ** 2, axis=1))) * 100.0
        )
        lcfs_score = _score(
            lcfs_error_cm,
            3.0,
            lower_bound=self.reward_weights.get("lcfs_lower_bound", True),
        )
        lcfs_competition_score = _score(lcfs_error_cm, 3.0, lower_bound=True)
        default_lx = 1.0
        target_lx = _scalar(observation, "reference_lX", default_lx)
        lx = _scalar(observation, "lX", target_lx)
        topology_score = 1.0 if int(round(lx)) == int(round(target_lx)) else 0.0
        x_score, strike_score, psix_score, nx_score = self._xpt_scores(observation)
        ipf = _array(observation, "I_PF", (12,))
        coil_margin = np.max(np.abs(ipf) / COIL_LIMITS_A)
        if self.reward_weights.get("coil_penalty_all_future_steps", False):
            if coil_margin >= 1.0:
                self.coil_max_current_mask = 0.0
            if self.coil_max_current_mask == 0:
                coil_penalty = 1.0
            else:
                coil_penalty = 0.0
        else:
            coil_penalty = max(0.0, float(coil_margin) - 1.0)
        pass
        ip_error_fuse_limit = 50000.0
        ip_error_mask = 1.0 if ip_abs_error <= ip_error_fuse_limit else 0.0
        if coil_margin > 1.0:
            self.coil_max_current_mask = 0.0
        lx = _scalar(observation, "lX", 0.0)
        pass
        competition_score = (
            competition_score
            * ip_error_mask
            * self.coil_max_current_mask
            * topology_mask
        )
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        action_l2 = (
            float(np.mean((action_arr / 500.0) ** 2)) if action_arr.size else 0.0
        )
        if self.prev_action is None or self.prev_action.shape != action_arr.shape:
            smooth = 0.0
        else:
            smooth = float(np.mean(((action_arr - self.prev_action) / 500.0) ** 2))
        self.prev_action = action_arr.copy()
        action_l2_penalty = self.action_l2_weight * action_l2
        action_smooth_penalty = self.action_smooth_weight * smooth
        if self.reward_weights:
            score_values = {
                "ip_score": ip_score,
                "r_score": r_score,
                "z_score": z_score,
                "lcfs_score": lcfs_score,
                "topology_score": topology_score,
                "x_score": x_score,
                "strike_score": strike_score,
                "psix_score": psix_score,
                "nx_score": nx_score,
            }
            score_weight_sum = 0.0
            weighted_score = 0.0
            for key, score_value in score_values.items():
                weight = float(self.reward_weights.get(key, 0.0))
                if weight <= 0.0:
                    continue
                weighted_score += weight * float(score_value)
                score_weight_sum += weight
            normalized_score = weighted_score / score_weight_sum
            reward = (
                self._weight("alive_offset", 0.0)
                + self._weight("score_scale", 2.0) * normalized_score
                - self._weight("coil_penalty", 0.5) * coil_penalty
                - self._weight("ip_error_boundary_penalty", 0.0)
                * ip_error_boundary_penalty
                - self._weight("action_l2_penalty", 0.0) * action_l2_penalty
                - self._weight("action_smooth_penalty", 0.0) * action_smooth_penalty
            )
        else:
            reward = competition_score
        self.last_components = {
            "reward": float(reward),
            "failure_penalty": 0.0,
            "ip_abs_error": float(ip_abs_error),
            "ip_relative_error": float(ip_rel),
            "ip_score": float(ip_score),
            "ip_safe_score": float(ip_safe_score),
            "ip_error_boundary_ratio": float(ip_error_boundary_ratio),
            "ip_error_boundary_penalty_raw": float(ip_error_boundary_penalty_raw),
            "ip_error_boundary_penalty": float(ip_error_boundary_penalty),
            "r_score": float(r_score),
            "z_score": float(z_score),
            "lcfs_error_cm": float(lcfs_error_cm),
            "lcfs_score": float(lcfs_score),
            "x_score": float(x_score),
            "strike_score": float(strike_score),
            "psix_score": float(psix_score),
            "nx_score": float(nx_score),
            "topology_score": float(topology_score),
            "coil_penalty": float(coil_penalty),
            "action_l2": float(action_l2),
            "action_smooth": float(smooth),
            "action_l2_penalty": float(action_l2_penalty),
            "action_smooth_penalty": float(action_smooth_penalty),
            "competition_score": float(competition_score),
            "coil_margin": float(coil_margin),
            "topology_mask": float(topology_mask),
        }
        return float(reward) if np.isfinite(reward) else -5.0
