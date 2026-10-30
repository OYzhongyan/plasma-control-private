"""Dense rewards aligned with the public semifinal scoring script."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from environment.xpt_utils import extract_xpt_observation_pack

COIL_LIMITS_A = np.asarray([45_000.0] + [14_000.0] * 10 + [4_000.0], dtype=np.float64)
MAIN_X_POINT_IDX = [1, 2]
SECOND_X_POINT_IDX = [0, 3]


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


# 考虑误差最大阈值的评分函数
def _score(error: float, epsilon_max: float, lower_bound: bool = True) -> float:
    """
    let ε represent the error
    if lower_bound=True:
        if ε<0, the score returned is 1.
        elif ε≥0:
            elif ε>ε_{max}:
                return score=0.
            elif ε≤ε_{max}:
                return 1-ε/ε_{max}
    elif lower_bound=False:
        if ε<0:
            return 2.
        elif 0<ε<ε_{max}:
            return 2-ε/ε_{max}
        elif ε≥ε_{max}:
            return exp[1-ε/ε_{max}]
         
    """
    if not np.isfinite(error):
        return 0.0
    if lower_bound:
        return float(max(0.0, 1.0 - max(float(error), 0.0) / float(epsilon_max)))
    else:
        raw_score = float(1.0 - max(float(error), 0.0) / float(epsilon_max))
        if raw_score < 0.0:
            return np.exp(raw_score)
        else:
            return raw_score + 1.0


@dataclass
class SemifinalDenseReward:
    """Stateful per-env dense reward for F1/F2 PPO training (Version 2)."""

    task: str
    # action_smooth_weight: float = 0.002
    # action_l2_weight: float = 0.0005
    reward_weights: dict[str, float] | None = None
    prev_action: np.ndarray | None = field(default=None, init=False)
    # initial_lcfs_cm: float | None = field(default=None, init=False)
    # initial_x_position_cm: float | None = field(default=None, init=False)
    last_components: dict[str, float] = field(default_factory=dict, init=False)
    coil_max_current_mask: float = 1.0
    score_lower_bound: bool = False

    # def __post_init__(self) -> None:
    #     if self.reward_weights:
    #         self.action_smooth_weight = float(
    #             self.reward_weights.get("action_smooth_penalty", self.action_smooth_weight)
    #         )
    #         self.action_l2_weight = float(
    #             self.reward_weights.get("action_l2_penalty", self.action_l2_weight)
    #         )

    def reset_state(self) -> None:
        self.prev_action = None
        # self.initial_lcfs_cm = None
        # self.initial_x_position_cm = None
        self.coil_max_current_mask = 1.0

    def __call__(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        terminated: bool = False,  # 由于破裂导致的异常早停
        truncated: bool = False,  # 超过最大环境步数导致的正常终止
        info: dict[str, Any] | None = None,
    ) -> float:
        if _scalar(observation, "failure", 0.0) > 0.5:
            failure_penalty = self.reward_weights.get("failure_penalty", 0.0)
            self.last_components = {
                "reward": float(failure_penalty),
                "failure": 1.0,
                "failure_penalty": float(failure_penalty),
            }
            reward = failure_penalty

        # 电流约束：超过最大值惩罚
        ipf = _array(observation, "I_PF", (12,))
        coil_margin = np.max(np.abs(ipf) / COIL_LIMITS_A)
        # 线圈最大电流对应的reward mask
        if coil_margin >= 1.0:
            self.coil_max_current_mask = 0.0
            coil_penalty = self.reward_weights.get("coil_current_exceed_limit_penalty", 0.0)
            self.last_components = {
                "reward": float(coil_penalty),
                "coil_current_exceeded": 1.0,
            }
            reward = coil_penalty
            terminated = True
        
        if terminated:
            self.reset_state()
            return reward, terminated

        is_episode_start = bool(info and int(info.get("step", 1)) <= 1)
        if is_episode_start:
            self.prev_action = None

        # Ip误差对应的reward
        ref_ip = _scalar(observation, "reference_Ip", 500_000.0)
        ip = _scalar(observation, "Ip", ref_ip)
        ip_abs_error = abs(ip - ref_ip)
        # ip_rel: [0, \inf)
        ip_rel = ip_abs_error / max(abs(ref_ip), 1.0)
        # ip_score: [0, 1]
        ip_score = _score(ip_rel, 0.05, lower_bound=self.score_lower_bound)
        ip_competition_score = _score(ip_rel, 0.05, lower_bound=True)

        # lcfs score
        lcfs = _array(observation, "lcfs_points", (32, 2))
        ref_lcfs = _array(observation, "reference_lcfs_points", (32, 2))
        lcfs_error_cm = float(np.sqrt(np.mean(np.sum((lcfs - ref_lcfs) ** 2, axis=1))) * 100.0)
        lcfs_score = _score(lcfs_error_cm, 3.0, lower_bound=self.score_lower_bound)
        lcfs_competition_score = _score(lcfs_error_cm, 3.0, lower_bound=True)

        # xpt对应的reward（F1 时以下变量保持默认值，供 score_values / last_components 引用）
        main_x_dist_score = main_x_psi_score = 0.0
        second_x_dist_score = second_x_psi_score = 0.0
        strike_dist_score = 0.0
        main_x_valid_ratio = second_x_valid_ratio = strike_valid_ratio = 0.0
        if self.task == "F2":
            pack = extract_xpt_observation_pack(
                observation,
                observation["reference_rX"],
                observation["reference_zX"],
                observation["reference_strike_r"],
                observation["reference_strike_z"],
                fx_order="C",
            )
            target_x_valid = np.asarray(observation["reference_x_valid"], dtype=np.float64) > 0.5
            x_valid = pack["x_valid"] > 0.5
            n_x_point_mask = 1.0 if pack["nX"] == 4 else 0.0

            # main X point distance score and psi score
            main_x_valid = target_x_valid[MAIN_X_POINT_IDX] & x_valid[MAIN_X_POINT_IDX]
            main_x_valid_ratio = np.mean(np.asarray(main_x_valid, dtype=np.float32))
            if np.any(main_x_valid):
                main_x_dist_cm = np.sqrt(
                    np.mean(
                        (pack["x_r"][MAIN_X_POINT_IDX] - observation["reference_rX"][MAIN_X_POINT_IDX]) ** 2
                        + (pack["x_z"][MAIN_X_POINT_IDX] - observation["reference_zX"][MAIN_X_POINT_IDX]) ** 2
                    )
                ) * 100.0
                main_x_dist_score = _score(main_x_dist_cm, 3, lower_bound=self.score_lower_bound) * main_x_valid_ratio
                main_x_dist_competition_score = _score(main_x_dist_cm, 3, lower_bound=True) * n_x_point_mask

                main_x_psi, lcfs_psi, axis_psi = pack["x_fx"][MAIN_X_POINT_IDX], pack["fb"], observation["FA"]
                psi_denom = lcfs_psi - axis_psi
                safe_psi_denom = psi_denom if abs(psi_denom) > 1e-12 else 1.0
                main_x_psi_error = np.sqrt(np.mean(((main_x_psi - lcfs_psi) / safe_psi_denom) ** 2))
                main_x_psi_score = _score(main_x_psi_error, 0.003, lower_bound=self.score_lower_bound) * main_x_valid_ratio
                main_x_psi_competition_score = _score(main_x_psi_error, 0.003, lower_bound=True) * n_x_point_mask
            else:
                main_x_dist_score = main_x_dist_competition_score = main_x_psi_score = main_x_psi_competition_score = 0.0

            # secondary X point distance score and psi score
            second_x_valid = target_x_valid[SECOND_X_POINT_IDX] & x_valid[SECOND_X_POINT_IDX]
            second_x_valid_ratio = np.mean(np.asarray(second_x_valid, dtype=np.float32))
            if np.any(second_x_valid):
                second_x_dist_cm = np.sqrt(
                    np.mean(
                        (pack["x_r"][SECOND_X_POINT_IDX] - observation["reference_rX"][SECOND_X_POINT_IDX]) ** 2
                        + (pack["x_z"][SECOND_X_POINT_IDX] - observation["reference_zX"][SECOND_X_POINT_IDX]) ** 2
                    )
                ) * 100.0
                second_x_dist_score = _score(second_x_dist_cm, 5, lower_bound=self.score_lower_bound) * second_x_valid_ratio
                second_x_dist_competition_score = _score(second_x_dist_cm, 5, lower_bound=True) * n_x_point_mask

                second_x_psi, lcfs_psi, axis_psi = pack["x_fx"][SECOND_X_POINT_IDX], pack["fb"], observation["FA"]
                psi_denom = lcfs_psi - axis_psi
                safe_psi_denom = psi_denom if abs(psi_denom) > 1e-12 else 1.0
                second_x_psi_error = np.sqrt(np.mean(((second_x_psi - lcfs_psi) / safe_psi_denom) ** 2))
                second_x_psi_score = _score(second_x_psi_error, 0.005, lower_bound=self.score_lower_bound) * second_x_valid_ratio
                second_x_psi_competition_score = _score(second_x_psi_error, 0.005, lower_bound=True) * n_x_point_mask
            else:
                second_x_dist_score = second_x_dist_competition_score = second_x_psi_score = second_x_psi_competition_score = 0.0

            # strike point distance score
            target_strike_valid = np.asarray(observation["reference_n_strike"], dtype=np.float64) > 0.5
            strike_valid = (pack["strike_valid"] > 0.5) & target_strike_valid
            strike_valid_ratio = np.mean(np.asarray(strike_valid, dtype=np.float32))
            if np.any(strike_valid):
                strike_dist_cm = np.sqrt(
                    np.mean(
                        (pack["strike_r"] - observation["reference_strike_r"]) ** 2
                        + (pack["strike_z"] - observation["reference_strike_z"]) ** 2
                    )
                ) * 100.0
                strike_dist_score = _score(strike_dist_cm, 10, lower_bound=self.score_lower_bound) * strike_valid_ratio
                strike_dist_competition_score = _score(strike_dist_cm, 10, lower_bound=True) * n_x_point_mask
            else:
                strike_dist_score = strike_dist_competition_score = 0.0

        # Ip误差对应的reward mask
        ip_error_fuse_limit = 50_000.0
        ip_error_mask = 1.0 if ip_abs_error <= ip_error_fuse_limit else 0.0
        # soft_ip_error_mask用于reward加权
        if ip_abs_error <= 50_000.0:
            soft_ip_error_mask = 1.0
        elif ip_abs_error <= 100_000.0:
            soft_ip_error_mask = (100_000.0 - ip_abs_error) / 50_000.0
        else:
            soft_ip_error_mask = 0.0

        # 位形类型对应的reward mask
        default_lx = 0.0 if self.task == "F1" else 1.0
        target_lx = _scalar(observation, "reference_lX", default_lx)
        lx = _scalar(observation, "lX", target_lx)
        topology_score = 1.0 if int(round(lx)) == int(round(target_lx)) else 0.0
        topology_mask = 1.0 if int(round(lx)) == int(round(target_lx)) else 0.5

        # 按比赛评分标准的单步得分
        if self.task == "F1":
            competition_score = 0.3 * ip_competition_score + 0.7 * lcfs_competition_score
        else:
            competition_score = (
                6.0 * ip_competition_score
                + 6.0 * lcfs_competition_score
                + 4.0 * main_x_dist_competition_score
                + 4.0 * main_x_psi_competition_score
                + 3.0 * second_x_dist_competition_score
                + 4.0 * second_x_psi_competition_score
                + 3.0 * strike_dist_competition_score
            ) / 30.0
        competition_score = competition_score * ip_error_mask * self.coil_max_current_mask * topology_mask
        # competition_score = competition_score * ip_error_mask * topology_mask

        # # action penalty and delta smoothness penalty
        # action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        # action_l2 = float(np.mean((action_arr / 500.0) ** 2)) if action_arr.size else 0.0
        # if self.prev_action is None or self.prev_action.shape != action_arr.shape:
        #     smooth = 0.0
        # else:
        #     smooth = float(np.mean(((action_arr - self.prev_action) / 500.0) ** 2))
        # self.prev_action = action_arr.copy()

        # action_l2_penalty = self.action_l2_weight * action_l2
        # action_smooth_penalty = self.action_smooth_weight * smooth

        if self.reward_weights.get("use_competition_score", False):
            reward = competition_score
        else:
            score_values = {
                "ip_score": ip_score,
                "lcfs_score": lcfs_score,
                "main_x_dist_score": main_x_dist_score,
                "main_x_psi_score": main_x_psi_score,
                "second_x_psi_score": second_x_psi_score,
                "second_x_dist_score": second_x_dist_score,
                "strike_dist_score": strike_dist_score,
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
            normalized_score = normalized_score * soft_ip_error_mask
            normalized_score = normalized_score * topology_mask

            reward = (
                # self.reward_weights.get("alive_offset", 0.0)
                normalized_score
                # + self.reward_weights.get("topology_score", 0.0) * topology_score
                # - self.reward_weights.get("ip_error_boundary_penalty", 0.0) * (1.0 - ip_error_mask)
                # - self.reward_weights.get("coil_penalty", 1.0) * coil_penalty
                # - self._weight("action_l2_penalty", 0.0) * action_l2_penalty
                # - self._weight("action_smooth_penalty", 0.0) * action_smooth_penalty
            )

            # if self.coil_max_current_mask == 0:
            #     reward = 0.0

        self.last_components = {
            "reward": float(reward),
            "failure_penalty": 0.0,
            "coil_current_exceeded": 0.0,
            "ip_abs_error": float(ip_abs_error),
            "ip_relative_error": float(ip_rel),
            "ip_score": float(ip_score),
            "ip_competition_score": float(ip_competition_score),
            # "ip_safe_score": float(ip_safe_score),
            # "ip_error_boundary_ratio": float(ip_error_boundary_ratio),
            # "ip_error_boundary_penalty_raw": float(ip_error_boundary_penalty_raw),
            # "ip_error_boundary_penalty": float(ip_error_boundary_penalty),
            # "r_score": float(r_score),
            # "z_score": float(z_score),
            "lcfs_error_cm": float(lcfs_error_cm),
            "lcfs_score": float(lcfs_score),
            "lcfs_competition_score": float(lcfs_competition_score),
            "topology_score": float(topology_score),
            "coil_margin": float(coil_margin),
            # "coil_penalty": float(coil_penalty),
            "ip_error_mask": float(ip_error_mask),
            "soft_ip_error_mask": float(soft_ip_error_mask),
            # "action_l2": float(action_l2),
            # "action_smooth": float(smooth),
            # "action_l2_penalty": float(action_l2_penalty),
            # "action_smooth_penalty": float(action_smooth_penalty),
            "competition_score": float(competition_score),
        }
        if self.task == "F2":
            self.last_components.update(
                {
                    "main_x_dist_score": main_x_dist_score,
                    "main_x_psi_score": main_x_psi_score,
                    "main_x_valid_ratio": main_x_valid_ratio,
                    "second_x_psi_score": second_x_psi_score,
                    "second_x_dist_score": second_x_dist_score,
                    "second_x_valid_ratio": second_x_valid_ratio,
                    "strike_dist_score": strike_dist_score,
                    "strike_valid_ratio": strike_valid_ratio,
                }
            )

        if truncated:
            self.reset_state()

        return float(reward) if np.isfinite(reward) else -5.0, terminated
