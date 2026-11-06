"""
子任务运行器：初始化环境、加载配置与 reference、启动 submission Docker（一次）、执行单任务对接。
仅依赖 environment（HFMSimulator）。
"""

from __future__ import annotations
import copy
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import numpy as np
import yaml
import obs_perturb
import submission_client
import trajectory_utils

TASK_IDS = ["F2b"]
SCRIPT_DIR = Path(__file__).resolve().parent
SAISDATA_ROOT = SCRIPT_DIR.parent
from environment import HFMSimulator


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_reference_spec(reference_config_path: str) -> dict[str, Any]:
    with open(reference_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return (cfg, int(cfg.get("max_steps", 500)))


def _reset_reference_obs(
    env_base_config: dict[str, Any], shot_id: str
) -> dict[str, Any]:
    """Reset a shot once to obtain its shape reference."""
    cfg = copy.deepcopy(env_base_config)
    cfg.setdefault("predictor", {})
    cfg["predictor"]["shot_id"] = shot_id
    env = HFMSimulator(cfg)
    try:
        obs, _ = env.reset(options={"reference_mode": "hold"})
    finally:
        env.close()
    return obs


def start_submission_service(
    project_root: Path,
    service_url: str,
    service_url2: str | None = None,
    image_name: str = "standalone-env-submission-eval",
) -> str | None:
    """
    构建并启动 submission 容器，等待健康检查通过。
    返回 container_id，失败则抛异常。
    """
    selected_url = service_url2 or service_url
    host_port = urlparse(selected_url).port or 9000
    submission_client.build_submission_image(project_root, image_name)
    container_id = submission_client.start_submission_container(image_name, host_port)
    submission_client.wait_for_health(selected_url)
    return container_id


def run_rollout(
    reference_spec: dict[str, Any],
    service_url: str,
    env_base_config: dict[str, Any],
    act_timeout_s: float,
    obs_perturb_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    执行单个子任务：env reset → 调用 submission /reset、/act 循环 → 收集 obs 并提取轨迹。
    /reset 与第一步 /act 不设 HTTP 超时（初始化可能较久）；从第二步起每步 /act 使用 act_timeout_s。
    返回 { "trajectory": {...}, "f_surv": float, "timeout": bool }。
    """
    print("[subtask] start F2b", flush=True)
    rollout_config = copy.deepcopy(env_base_config)
    rollout_config["max_steps"] = int(
        reference_spec.get("max_steps", rollout_config.get("max_steps", 500))
    )
    rollout_config.setdefault("predictor", {})
    rollout_config["predictor"]["shot_id"] = reference_spec["shot_id"]
    warmup_env = HFMSimulator(rollout_config)
    try:
        initial_obs, _ = warmup_env.reset(options={"reference_mode": "hold"})
    finally:
        warmup_env.close()
    target_obs = trajectory_utils.build_official_target_observation(reference_spec)
    target_shot_id = reference_spec.get("target_shot_id")
    if target_obs is None and target_shot_id:
        target_obs = _reset_reference_obs(rollout_config, str(target_shot_id))
    reference_trajectory = trajectory_utils.build_reference_trajectory(
        reference_spec, initial_obs, rollout_config["max_steps"], target_obs=target_obs
    )
    env = HFMSimulator(rollout_config)
    options = {
        "reference_mode": reference_spec.get("reference_mode", "hold"),
        "reference": (
            reference_trajectory
            if reference_spec.get("reference_mode", "hold") == "trajectory"
            else None
        ),
    }
    obs_list: list[dict[str, Any]] = []
    steps_survived = 0
    timed_out = False
    record_scalars = bool(
        obs_perturb_cfg and obs_perturb_cfg.get("record_sent_scalars", False)
    )
    true_scalars: list[dict[str, float]] = []
    sent_scalars: list[dict[str, float]] = []
    true_record: list[dict[str, Any]] = []
    sent_record: list[dict[str, Any]] = []
    record_max_flat = 128
    if record_scalars and obs_perturb_cfg:
        record_max_flat = int(obs_perturb_cfg.get("record_sent_max_flat", 128))
    try:
        reset_payload = submission_client.post_json(
            service_url,
            "/reset",
            {"episode_id": "F2b", "options": trajectory_utils.to_jsonable(options)},
            timeout=None,
        )
        if not reset_payload.get("ok", False):
            raise RuntimeError(f"submission reset failed for F2b: {reset_payload}")
        obs, _ = env.reset(options=options)
        perturb_on = bool(obs_perturb_cfg and obs_perturb_cfg.get("enabled", False))
        delay_steps = 0
        if perturb_on:
            delay_steps = obs_perturb.delay_steps_from_config(obs_perturb_cfg)
            nb = obs_perturb_cfg.get("noise") or {}
            if delay_steps <= 0 and (not bool(nb.get("enabled", False))):
                perturb_on = False
        obs_hist: list[dict[str, np.ndarray]] = []
        if perturb_on:
            obs_hist.append(obs_perturb.snapshot_obs_arrays(obs))
            seed = int(obs_perturb_cfg.get("seed", 0))
            rng = obs_perturb.make_observation_rng(seed)
        for step_t in range(rollout_config["max_steps"]):
            try:
                if record_scalars:
                    true_scalars.append(
                        {
                            "Ip": float(np.asarray(obs["Ip"]).ravel()[0]),
                            "R": float(np.asarray(obs["R"]).ravel()[0]),
                            "Z": float(np.asarray(obs["Z"]).ravel()[0]),
                        }
                    )
                if perturb_on:
                    base = obs_perturb.select_delayed_base(
                        obs_hist, step_t, delay_steps
                    )
                    policy_obs = obs_perturb.build_policy_observation(
                        base, rng, obs_perturb_cfg
                    )
                    obs_payload = {
                        k: np.asarray(v).tolist() for k, v in policy_obs.items()
                    }
                else:
                    policy_obs = None
                    obs_payload = {k: np.asarray(v).tolist() for k, v in obs.items()}
                if record_scalars:
                    src = policy_obs if policy_obs is not None else obs
                    sent_scalars.append(
                        {
                            "Ip": float(np.asarray(src["Ip"]).ravel()[0]),
                            "R": float(np.asarray(src["R"]).ravel()[0]),
                            "Z": float(np.asarray(src["Z"]).ravel()[0]),
                        }
                    )
                    true_record.append(
                        obs_perturb.serialize_obs_for_record(obs, record_max_flat)
                    )
                    sent_record.append(
                        obs_perturb.serialize_obs_for_record(src, record_max_flat)
                    )
                act_timeout = None if step_t == 0 else act_timeout_s
                act_payload = submission_client.post_json(
                    service_url,
                    "/act",
                    {"observation": obs_payload},
                    timeout=act_timeout,
                )
            except Exception:
                timed_out = True
                break
            if not act_payload.get("ok", False):
                timed_out = True
                break
            action = np.asarray(act_payload["action"], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            obs_list.append(dict(obs))
            if perturb_on:
                obs_hist.append(obs_perturb.snapshot_obs_arrays(obs))
            steps_survived += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    f_surv = min(1.0, steps_survived / max(rollout_config["max_steps"], 1))
    trajectory = trajectory_utils.extract_semifinal_trajectory_from_obs_list(
        obs_list, target_obs=target_obs
    )
    out: dict[str, Any] = {
        "trajectory": trajectory_utils.to_jsonable(trajectory),
        "f_surv": float(f_surv),
        "timeout": bool(timed_out),
    }
    if record_scalars:
        out["observation_true_scalars"] = true_scalars
        out["observation_sent_scalars"] = sent_scalars
        out["observation_true_record"] = true_record
        out["observation_sent_record"] = sent_record
    print(
        f"[subtask] done F2b: f_surv={out['f_surv']:.3f}, timeout={out['timeout']}",
        flush=True,
    )
    return out
