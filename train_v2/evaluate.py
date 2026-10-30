"""Evaluate a trained PPO policy: roll out per shot, record trajectories, and score.

从训练 save_dir 加载 configs.pkl 恢复完整训练配置，命令行仅需覆盖 eval.* 参数。
model / vecnormalize 路径根据 eval.model_path + eval.cp_num 自动解析。

用法示例：
  # 评估最终模型（ppo_<task>_latest.zip）
  python train_v2/evaluate.py eval.model_path=results/20260620_xxx

  # 评估指定 checkpoint
  python train_v2/evaluate.py eval.model_path=results/20260620_xxx eval.cp_num=149952

  # 指定 shots / 关闭评分
  python train_v2/evaluate.py eval.model_path=results/xxx eval.shots=[13844_500] eval.score=false

  # 零动作基线（不需要模型，但仍需 model_path 以加载 env 配置）
  python train_v2/evaluate.py eval.model_path=results/xxx eval.policy_mode=zero
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_V2 = Path(__file__).resolve().parent
for _p in (str(ROOT), str(TRAIN_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
EVAL_ROOT = ROOT / "evaluation" / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
EVAL_FILES = ROOT / "eval_files"
if str(EVAL_FILES) not in sys.path:
    sys.path.insert(0, str(EVAL_FILES))

from environment import HFMSimulator  # noqa: E402
from environment.xpt_utils import extract_xpt_observation_pack  # noqa: E402
from score.evaluate_core import evaluate  # noqa: E402
from trajectory_utils import extract_semifinal_trajectory_from_obs_list, to_jsonable  # noqa: E402

from rewards import SemifinalDenseReward  # noqa: E402
from wrappers import (  # noqa: E402
    COIL_LIMITS_A,
    EpisodeResetWrapper,
    HoldInitialResetWrapper,
    HistoricalOBservationWrapper,
    NormalizedAction7DTo12DWrapper,
    # RawObservationNoiseWrapper,
    ObservationDelayNoiseWrapper,
    RewardInfoWrapper,
)
from train_ppo import build_training_context, make_env_config, flatten_shot_groups, parse_ports  # noqa: E402


def _rooted(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# =====================================================================================
# Observation capture wrapper
# =====================================================================================

class DictObservationCaptureWrapper(gym.Wrapper):
    """Capture raw dict observations before feature flattening for recording/scoring."""

    def __init__(self, env: gym.Env, sink: dict[str, Any]):
        super().__init__(env)
        self.sink = sink

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.sink["last_observation"] = observation
        self.sink["last_info"] = dict(info or {})
        self.sink["last_reward_components"] = {}
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info["captured_observation"] = observation
        self.sink["last_observation"] = observation
        self.sink["last_info"] = info
        self.sink["last_reward_components"] = dict(info.get("reward_components", {}))
        return observation, reward, terminated, truncated, info


# =====================================================================================
# Env construction (mirrors train_ppo.make_env + capture wrapper)
# =====================================================================================

def make_eval_env(
    *,
    cfg: DictConfig,
    references: list[dict[str, Any]] | None,
    reset_perturbation: dict[str, Any] | None,
    base_config: dict[str, Any],
    shot_id: str,
    port: int,
    seed: int,
    sink: dict[str, Any],
):
    """Build a single eval env with raw-observation capture (mirrors train_ppo.make_env)."""

    def _init() -> gym.Env:
        env_config = make_env_config(
            base_config,
            port=port,
            shot_id=shot_id,
            max_steps=int(cfg.max_steps),
            reference_mode=str(cfg.target.reference_mode),
        )
        reward_fn = SemifinalDenseReward(
            task=str(cfg.task),
            reward_weights=cfg.reward_weights,
            score_lower_bound=cfg.reward_weights.get("score_lower_bound", False),
        )
        env: gym.Env = HFMSimulator(env_config, reward_fn=reward_fn)
        if cfg.target.reference_mode == "hold_initial":
            env = HoldInitialResetWrapper(
                env, shot_id=shot_id, reset_perturbation=reset_perturbation, seed=seed,
            )
        else:
            env = EpisodeResetWrapper(
                env, references=references, shot_id=shot_id,
                reset_perturbation=reset_perturbation,
                fix_reset_params=bool(cfg.reset_perturbation.get("fix_reset_params", False)),
                seed=seed,
            )
        env = NormalizedAction7DTo12DWrapper(env, action_mode=str(cfg.actions.mode))
        env = RewardInfoWrapper(env)
        # Capture the clean observation for scoring before applying the optional
        # inference-side delay/noise seen by the policy.
        env = DictObservationCaptureWrapper(env, sink)
        perturb_cfg = cfg.eval.get("obs_perturb", {})
        if bool(perturb_cfg.get("enabled", False)):
            env = ObservationDelayNoiseWrapper(
                env,
                delay_steps=int(perturb_cfg.get("delay_steps", 0) or 0),
                noise_std=dict(perturb_cfg.get("noise_std") or {}),
                seed=seed + 10_000,
            )
        env = HistoricalOBservationWrapper(
            env,
            obs_history_step=int(cfg.features.history_steps),
            include_xpt=(str(cfg.task).strip().upper() == "F2"),
            action_mode=str(cfg.actions.mode)
        )
        return env

    return _init


# =====================================================================================
# Recording helpers
# =====================================================================================

def _scalar(obs: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = obs.get(key)
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _array(obs: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = obs.get(key)
    out = np.zeros(int(np.prod(shape)), dtype=np.float64)
    if value is not None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        n = min(out.size, flat.size)
        out[:n] = flat[:n]
    return out.reshape(shape)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def lcfs_rms_cm(obs: dict[str, Any]) -> float:
    lcfs = _array(obs, "lcfs_points", (32, 2))
    ref = _array(obs, "reference_lcfs_points", (32, 2))
    return float(np.sqrt(np.mean(np.sum((lcfs - ref) ** 2, axis=1))) * 100.0)


def record_row(
    *,
    step: int,
    shot_id: str,
    episode: int,
    port: int,
    observation: dict[str, Any],
    reward: float,
    done: bool,
    info: dict[str, Any],
    reward_components: dict[str, Any],
) -> dict[str, Any]:
    ip = _scalar(observation, "Ip")
    ref_ip = _scalar(observation, "reference_Ip", ip)
    lx = _scalar(observation, "lX")
    ref_lx = _scalar(observation, "reference_lX", lx)
    topology_mask = 1.0 if int(round(lx)) == int(round(ref_lx)) else 0.5
    i_pf = _array(observation, "I_PF", (12,))
    coil_margin = float(np.max(np.abs(i_pf) / COIL_LIMITS_A))
    row: dict[str, Any] = {
        "episode": episode,
        "env_index": episode,
        "step": step,
        "env_reward": float(reward),
        "ip": ip,
        "reference_ip": ref_ip,
        "ip_abs_error": abs(ip - ref_ip),
        "terminated": int(bool(done)),
        "truncated": 0,
        "shot_id": shot_id,
        "port": port,
        "hfm_timeout": 0,
        "failure": int(bool(info.get("failure", False)) or _scalar(observation, "failure", 0.0) > 0.5),
        "lX": lx,
        "reference_lX": ref_lx,
        "nX": _scalar(observation, "nX"),
        "coil_margin": coil_margin,
        "topology_mask": topology_mask,
        "lcfs_rms_cm": lcfs_rms_cm(observation),
    }
    for key, value in reward_components.items():
        if isinstance(value, (int, float, np.generic)):
            row[str(key)] = float(value)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], *, max_steps: int) -> dict[str, Any]:
    if not rows:
        return {}
    step_rows = [row for row in rows if int(row["step"]) > 0]
    final = rows[-1]

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(row.get(key, np.nan)) for row in step_rows], dtype=np.float64)

    lengths = int(final["step"])
    reward_sum = float(np.nansum(arr("env_reward"))) if step_rows else 0.0
    failure = bool(final.get("failure", 0))
    return {
        "shot_id": final["shot_id"],
        "episode": int(final["episode"]),
        "steps": lengths,
        "max_steps": max_steps,
        "survived_full": bool(lengths >= max_steps and not failure),
        "failure": failure,
        "reward_sum": reward_sum,
        "mean_ip_abs_error_ka": float(np.nanmean(arr("ip_abs_error")) / 1000.0) if step_rows else None,
        "max_ip_abs_error_ka": float(np.nanmax(arr("ip_abs_error")) / 1000.0) if step_rows else None,
        "mean_lcfs_rms_cm": float(np.nanmean(arr("lcfs_rms_cm"))) if step_rows else None,
        "max_lcfs_rms_cm": float(np.nanmax(arr("lcfs_rms_cm"))) if step_rows else None,
        "final_lcfs_rms_cm": float(final["lcfs_rms_cm"]),
        "topology_match_rate": (
            float(np.nanmean(np.round(arr("lX")) == np.round(arr("reference_lX")))) if step_rows else None
        ),
        "max_coil_margin": float(np.nanmax(arr("coil_margin"))) if step_rows else None,
    }


def save_observations_npz(
    obs_lists: list[list[dict[str, Any]]],
    max_steps: int,
    path: Path,
) -> None:
    """Save rollout observations as NPZ for visualize_rollout.py.

    Arrays shape: [num_rollouts, max_steps, ...]
      - lcfs_points: [N, T, 32, 2]
      - reference_lcfs_points: [N, T, 32, 2]
      - ip: [N, T]
      - reference_ip: [N, T]
      - valid: [N, T] (bool)
      - fb: [N, T] 边界参考极向磁通
      - fx: [N, T, 4290] (float32) 磁通网格展开向量（C 序 reshape 为 (66, 65)，
            axis0 对应 rx，axis1 对应 zx）
      - rx / zx: 磁通网格坐标 (66,) / (65,)，网格固定，只保存一份
      - rX / zX: [N, T, 6] X 点坐标（仅前 nX 个有效）
      - nX: [N, T] 有效 X 点个数

    F2 XPT 槽位数据（仅在 obs 含 reference_* 字段时保存）：
      - x_slot_r / x_slot_z: [N, T, 4] 槽位匹配后的当前 X 点（匈牙利匹配到 reference 槽位）
      - x_valid: [N, T, 4] X 点槽位有效掩码
      - strike_r / strike_z: [N, T, 8] 槽位匹配后的当前打击点
      - strike_valid: [N, T, 8] 打击点槽位有效掩码
      - ref_rX / ref_zX: [N, T, 4] 目标 X 点坐标
      - ref_strike_r / ref_strike_z: [N, T, 8] 目标打击点坐标
    """
    num_rollouts = len(obs_lists)
    lcfs = np.zeros((num_rollouts, max_steps, 32, 2), dtype=np.float64)
    ref_lcfs = np.zeros((num_rollouts, max_steps, 32, 2), dtype=np.float64)
    ip = np.zeros((num_rollouts, max_steps), dtype=np.float64)
    ref_ip = np.zeros((num_rollouts, max_steps), dtype=np.float64)
    valid = np.zeros((num_rollouts, max_steps), dtype=bool)
    fx = np.zeros((num_rollouts, max_steps, 4290), dtype=np.float32)
    fb = np.zeros((num_rollouts, max_steps), dtype=np.float64)
    rX = np.zeros((num_rollouts, max_steps, 6), dtype=np.float64)
    zX = np.zeros((num_rollouts, max_steps, 6), dtype=np.float64)
    nX = np.zeros((num_rollouts, max_steps), dtype=np.float64)
    x_slot_r = np.zeros((num_rollouts, max_steps, 4), dtype=np.float64)
    x_slot_z = np.zeros((num_rollouts, max_steps, 4), dtype=np.float64)
    x_valid = np.zeros((num_rollouts, max_steps, 4), dtype=np.float64)
    strike_r = np.zeros((num_rollouts, max_steps, 8), dtype=np.float64)
    strike_z = np.zeros((num_rollouts, max_steps, 8), dtype=np.float64)
    strike_valid = np.zeros((num_rollouts, max_steps, 8), dtype=np.float64)
    ref_rX = np.zeros((num_rollouts, max_steps, 4), dtype=np.float64)
    ref_zX = np.zeros((num_rollouts, max_steps, 4), dtype=np.float64)
    ref_strike_r = np.zeros((num_rollouts, max_steps, 8), dtype=np.float64)
    ref_strike_z = np.zeros((num_rollouts, max_steps, 8), dtype=np.float64)
    rx: np.ndarray | None = None
    zx: np.ndarray | None = None
    has_xpt_ref = False

    for env_idx, obs_list in enumerate(obs_lists):
        for step_idx, obs in enumerate(obs_list):
            valid[env_idx, step_idx] = True
            lcfs[env_idx, step_idx] = _array(obs, "lcfs_points", (32, 2))
            ref_lcfs[env_idx, step_idx] = _array(obs, "reference_lcfs_points", (32, 2))
            ip[env_idx, step_idx] = _scalar(obs, "Ip")
            ref_ip[env_idx, step_idx] = _scalar(obs, "reference_Ip", ip[env_idx, step_idx])
            fx[env_idx, step_idx] = _array(obs, "Fx", (4290,))
            fb[env_idx, step_idx] = _scalar(obs, "FB")
            rX[env_idx, step_idx] = _array(obs, "rX", (6,))
            zX[env_idx, step_idx] = _array(obs, "zX", (6,))
            nX[env_idx, step_idx] = _scalar(obs, "nX")
            if rx is None and obs.get("rx") is not None:
                rx = np.asarray(obs["rx"], dtype=np.float64).ravel()
            if zx is None and obs.get("zx") is not None:
                zx = np.asarray(obs["zx"], dtype=np.float64).ravel()
            if obs.get("reference_rX") is not None and obs.get("reference_strike_r") is not None:
                has_xpt_ref = True
                ref_rX[env_idx, step_idx] = _array(obs, "reference_rX", (4,))
                ref_zX[env_idx, step_idx] = _array(obs, "reference_zX", (4,))
                ref_strike_r[env_idx, step_idx] = _array(obs, "reference_strike_r", (8,))
                ref_strike_z[env_idx, step_idx] = _array(obs, "reference_strike_z", (8,))
                try:
                    pack = extract_xpt_observation_pack(
                        obs,
                        np.asarray(obs["reference_rX"], dtype=np.float64),
                        np.asarray(obs["reference_zX"], dtype=np.float64),
                        np.asarray(obs["reference_strike_r"], dtype=np.float64),
                        np.asarray(obs["reference_strike_z"], dtype=np.float64),
                        fx_order="C",
                    )
                    x_slot_r[env_idx, step_idx] = np.asarray(pack["x_r"], dtype=np.float64).reshape(4)
                    x_slot_z[env_idx, step_idx] = np.asarray(pack["x_z"], dtype=np.float64).reshape(4)
                    x_valid[env_idx, step_idx] = np.asarray(pack["x_valid"], dtype=np.float64).reshape(4)
                    strike_r[env_idx, step_idx] = np.asarray(pack["strike_r"], dtype=np.float64).reshape(8)
                    strike_z[env_idx, step_idx] = np.asarray(pack["strike_z"], dtype=np.float64).reshape(8)
                    strike_valid[env_idx, step_idx] = np.asarray(pack["strike_valid"], dtype=np.float64).reshape(8)
                except Exception:
                    pass

    payload: dict[str, np.ndarray] = {
        "lcfs_points": lcfs,
        "reference_lcfs_points": ref_lcfs,
        "ip": ip,
        "reference_ip": ref_ip,
        "valid": valid,
        "fb": fb,
        "rX": rX,
        "zX": zX,
        "nX": nX,
    }
    if rx is not None and zx is not None:
        payload["fx"] = fx
        payload["rx"] = rx
        payload["zx"] = zx
    if has_xpt_ref:
        payload.update(
            {
                "x_slot_r": x_slot_r,
                "x_slot_z": x_slot_z,
                "x_valid": x_valid,
                "strike_r": strike_r,
                "strike_z": strike_z,
                "strike_valid": strike_valid,
                "ref_rX": ref_rX,
                "ref_zX": ref_zX,
                "ref_strike_r": ref_strike_r,
                "ref_strike_z": ref_strike_z,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


# =====================================================================================
# Scoring
# =====================================================================================

def infer_task_id(task: str, shots: list[str]) -> str:
    task_upper = str(task).upper()
    if task_upper.startswith("F1"):
        return "F1"
    if shots and all(str(shot).startswith("15892") or str(shot).startswith("14795") for shot in shots):
        return "F2b"
    return "F2a"


def build_scoring_target(reference: dict[str, Any]) -> dict[str, Any]:
    """将训练 reference dict 转成官方评分器（score.tasks）期望的 target 格式。

    官方逐任务评分需要 target 含 lcfs_points / Xpt_main / Xpt_sec / strike；
    训练 reference 的 X 点槽位序为 [sec0, main0, main1, sec1]（main=槽位[1,2]）。
    """
    target: dict[str, Any] = {
        "lcfs_points": np.asarray(reference["lcfs_points"], dtype=np.float64),
    }
    if "rX" in reference and "zX" in reference:
        rX = np.asarray(reference["rX"], dtype=np.float64).reshape(4)
        zX = np.asarray(reference["zX"], dtype=np.float64).reshape(4)
        target["Xpt_main"] = np.column_stack([rX[[1, 2]], zX[[1, 2]]])
        target["Xpt_sec"] = np.column_stack([rX[[0, 3]], zX[[0, 3]]])
    if "strike_r" in reference and "strike_z" in reference:
        target["strike"] = np.column_stack([
            np.asarray(reference["strike_r"], dtype=np.float64).reshape(-1),
            np.asarray(reference["strike_z"], dtype=np.float64).reshape(-1),
        ])
    return target


def build_score_outputs(
    *,
    infer_tasks: dict[str, Any],
    target_all: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for rollout_id, entry in infer_tasks.items():
        tid = str(entry.get("task_id", task_id))
        result = evaluate(
            {tid: {"trajectory": entry["trajectory"], "timeout": bool(entry.get("timeout", False))}},
            target_all,
            task_ids=[tid],
        )
        scored.append({
            "rollout_id": rollout_id,
            "task_id": tid,
            "shot_id": entry.get("shot_id"),
            "episode": entry.get("episode"),
            "n_steps": entry.get("n_steps"),
            "task_score": float(result["task_scores"].get(tid, 0.0)),
            "metric_scores": result["task_metric_scores"].get(tid, {}),
            "total_score": float(result.get("total_score", 0.0)),
        })

    if len(infer_tasks) == 1:
        entry = next(iter(infer_tasks.values()))
        tid = str(entry.get("task_id", task_id))
        infer_result: dict[str, Any] = {
            tid: {"trajectory": entry["trajectory"], "timeout": bool(entry.get("timeout", False))}
        }
    else:
        infer_result = {
            rollout_id: {
                "task_id": entry.get("task_id", task_id),
                "shot_id": entry.get("shot_id"),
                "episode": entry.get("episode"),
                "trajectory": entry["trajectory"],
                "timeout": bool(entry.get("timeout", False)),
            }
            for rollout_id, entry in infer_tasks.items()
        }

    scores = np.asarray([item["task_score"] for item in scored], dtype=np.float64)
    summary = {
        "task_id": task_id,
        "rollouts": scored,
        "mean_task_score": float(np.nanmean(scores)) if scores.size else 0.0,
        "min_task_score": float(np.nanmin(scores)) if scores.size else 0.0,
        "max_task_score": float(np.nanmax(scores)) if scores.size else 0.0,
    }
    return {"infer_result": infer_result, "score_summary": summary}


# =====================================================================================
# Rollout
# =====================================================================================

def rollout_one(
    *,
    model: Any,
    cfg: DictConfig,
    references: list[dict[str, Any]] | None,
    reset_perturbation: dict[str, Any] | None,
    base_config: dict[str, Any],
    shot_id: str,
    episode: int,
    port: int,
    seed: int,
    vecnormalize_path: Path | None,
    deterministic: bool,
    policy_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    sink: dict[str, Any] = {}
    env_fn = make_eval_env(
        cfg=cfg, references=references, reset_perturbation=reset_perturbation,
        base_config=base_config, shot_id=shot_id, port=port, seed=seed, sink=sink,
    )
    vec_env = DummyVecEnv([env_fn])
    if vecnormalize_path is not None:
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    rows: list[dict[str, Any]] = []
    obs_list: list[dict[str, Any]] = []
    try:
        obs = vec_env.reset()
        reset_info = dict(sink.get("last_info", {}))
        reset_info["task"] = str(cfg.task)
        rows.append(record_row(
            step=0, shot_id=shot_id, episode=episode, port=port,
            observation=sink["last_observation"],
            reward=0.0, done=False, info=reset_info, reward_components={},
        ))
        max_steps = int(cfg.max_steps)
        for step in range(1, max_steps + 1):
            if policy_mode == "zero":
                action = np.zeros((1, 7), dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=np.float32)
            obs, rewards, dones, infos = vec_env.step(action_arr)
            done = bool(dones[0])
            info = dict(infos[0] or {})
            info["task"] = str(cfg.task)
            raw_observation = info.get("captured_observation", sink["last_observation"])
            obs_list.append(dict(raw_observation))
            rows.append(record_row(
                step=step, shot_id=shot_id, episode=episode, port=port,
                observation=raw_observation,
                reward=float(rewards[0]), done=done, info=info,
                reward_components=info.get(
                    "reward_components", sink.get("last_reward_components", {}),
                ),
            ))
            if done:
                break
    finally:
        vec_env.close()

    summary = summarize_rows(rows, max_steps=int(cfg.max_steps))
    return rows, summary, obs_list


# =====================================================================================
# Main entrypoint
# =====================================================================================

def resolve_model_paths(model_path: str | Path, task: str, cp_num: int | None) -> tuple[Path, Path | None]:
    """根据 model_path + cp_num 解析 PPO 模型与 VecNormalize 路径。

    - cp_num=None: 读取 <model_path>/ppo_<task>_latest.zip 和 vecnormalize.pkl
    - cp_num=<N>:  读取 <model_path>/checkpoints/ppo_<task>_<N>_steps.zip 和对应 _vecnormalize.pkl
    """
    model_path = Path(model_path)
    task_lower = str(task).lower()
    if cp_num is None:
        model_file = model_path / f"ppo_{task_lower}_latest.zip"
        vecnorm_file = model_path / "vecnormalize.pkl"
    else:
        cp_base = model_path / "checkpoints" / f"ppo_{task_lower}_{int(cp_num)}_steps"
        model_file = cp_base.with_suffix(".zip")
        vecnorm_file = cp_base.with_name(f"{cp_base.name}_vecnormalize.pkl")
    if not model_file.exists():
        raise FileNotFoundError(f"model not found: {model_file}")
    vecnormalize_path = vecnorm_file if vecnorm_file.exists() else None
    return model_file, vecnormalize_path


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> int:

    # Build training context (references, reset_perturbation, base_config)
    cfg, references, reset_perturbation, base_config, _ = build_training_context(cfg)

    eval_cfg = cfg.eval

    # 输出目录: 默认 <model_path>/eval
    output_dir = _rooted(eval_cfg.model_path) / "eval"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine shots
    shots_cfg = eval_cfg.shots
    if shots_cfg is None:
        if cfg.task == "F1":
            shots = ["13705_500"]
        elif cfg.task == "F2":
            shots = ["13705_500", "14795_400"]
        else:
            raise ValueError(f"unknown task: {cfg.task}")
    elif isinstance(shots_cfg, str):
        shots = [shots_cfg]
    else:
        shots = [str(s) for s in shots_cfg]

    # Determine port
    port = parse_ports(str(cfg.ports))[0]

    # Resolve model / vecnormalize paths from model_path + cp_num
    cp_num = eval_cfg.cp_num
    cp_num = int(cp_num) if cp_num is not None else None
    model_file, vecnormalize_path = resolve_model_paths(eval_cfg.model_path, str(cfg.task), cp_num)

    # Load model
    model = None
    if str(eval_cfg.policy_mode) == "model":
        from stable_baselines3 import PPO
        model = PPO.load(str(model_file), device=str(cfg.ppo.device))

    # Scoring target: 直接使用 references[0] 作为 target_obs
    score_enabled = bool(eval_cfg.score) and references is not None and len(references) > 0
    if bool(eval_cfg.score) and not score_enabled:
        print("scoring disabled: references is empty (hold_initial mode has no target trajectory)")

    print(
        f"eval setup: task={cfg.task}, model={model_file}, vecnormalize={vecnormalize_path}, "
        f"port={port}, shots={shots}, max_steps={cfg.max_steps}, "
        f"policy_mode={eval_cfg.policy_mode}, deterministic={eval_cfg.deterministic}"
    )

    all_rows: list[dict[str, Any]] = []
    all_obs_lists: list[list[dict[str, Any]]] = []
    all_summaries: list[dict[str, Any]] = []
    task_id = infer_task_id(str(cfg.task), shots)
    infer_tasks: dict[str, Any] = {}

    for shot_id in shots:
        for episode in range(int(eval_cfg.episodes_per_shot)):
            seed = int(eval_cfg.seed) + 1000 * shots.index(shot_id) + episode
            print(f"rolling out shot={shot_id}, episode={episode}, seed={seed}")
            rows, summary, obs_list = rollout_one(
                model=model, cfg=cfg, references=references,
                reset_perturbation=reset_perturbation, base_config=base_config,
                shot_id=shot_id, episode=episode, port=port, seed=seed,
                vecnormalize_path=vecnormalize_path,
                deterministic=bool(eval_cfg.deterministic), policy_mode=str(eval_cfg.policy_mode),
            )
            all_rows.extend(rows)
            all_obs_lists.append(obs_list)
            all_summaries.append(summary)
            if score_enabled:
                trajectory = extract_semifinal_trajectory_from_obs_list(
                    obs_list, task_id=task_id, target_obs=references[0],
                )
                infer_tasks[f"{task_id}_{shot_id}_ep{episode:02d}"] = {
                    "task_id": task_id,
                    "shot_id": shot_id,
                    "episode": episode,
                    "trajectory": trajectory,
                    "timeout": False,
                    "n_steps": len(obs_list),
                }
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))

    # Save combined CSV and NPZ for visualize_rollout.py
    csv_path = output_dir / "step_reward_components.csv"
    write_csv(csv_path, all_rows)
    npz_path = output_dir / "rollout_observations.npz"
    save_observations_npz(all_obs_lists, max_steps=int(cfg.max_steps), path=npz_path)
    print(f"saved {len(all_rows)} rows to {csv_path}")
    print(f"saved {len(all_obs_lists)} rollouts to {npz_path}")

    # Score
    if score_enabled and infer_tasks:
        target_all = {task_id: build_scoring_target(references[0])}
        score_payload = build_score_outputs(
            infer_tasks=infer_tasks, target_all=target_all, task_id=task_id,
        )
        infer_path = output_dir / "infer_result.json"
        score_path = output_dir / "score_summary.json"
        infer_path.write_text(
            json.dumps(to_jsonable(score_payload["infer_result"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        score_path.write_text(
            json.dumps(to_jsonable(score_payload["score_summary"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"infer result: {infer_path}")
        print(f"score summary: {score_path}")

    write_csv(output_dir / "rollout_summary.csv", all_summaries)
    (output_dir / "rollout_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"summary: {output_dir / 'rollout_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
