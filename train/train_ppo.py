"""Hydra PPO training entrypoint for the F2b experiment."""

from __future__ import annotations
import pickle
import json
import sys
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import yaml
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from environment import HFMSimulator
from train.callbacks import (
    PlasmaCheckpointCallback,
    RewardComponentsCallback,
    find_vecnormalize,
    save_vecnormalize_stats,
)
from train.references import build_reference, default_reference_path
from train.rewards import CompetitionReward
from train.wrappers import *
from train.utils import warmup_cosine_schedule
from submission.history import reference_metadata


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def _rooted_path(value: str | Path | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _to_dict(node: Any) -> Any:
    """OmegaConf 节点 -> 纯 Python 容器（resolve 插值）。None 原样返回。"""
    if node is None:
        return None
    if isinstance(node, DictConfig) or isinstance(node, list):
        return OmegaConf.to_container(node, resolve=True)
    return node


def parse_ports(spec: str) -> list[int]:
    ports: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ports.extend(range(int(start), int(end) + 1))
        else:
            ports.append(int(part))
    if not ports:
        raise ValueError("at least one port is required")
    return ports


def flatten_shot_groups(shot_groups: Any) -> list[str]:
    if shot_groups is None:
        return []
    if isinstance(shot_groups, dict):
        shots: list[str] = []
        for group_shots in shot_groups.values():
            if isinstance(group_shots, str):
                shots.append(group_shots)
            else:
                shots.extend((str(shot_id) for shot_id in group_shots))
        return shots
    elif isinstance(shot_groups, str):
        return [shot_groups]
    else:
        return [str(shot_id) for shot_id in shot_groups]


def assign_shots_to_envs(
    shot_groups: Any, num_envs: int, fallback_shot_id: str
) -> list[str]:
    num_envs = max(1, int(num_envs))
    shot_ids = flatten_shot_groups(shot_groups) or [str(fallback_shot_id)]
    return [shot_ids[i % len(shot_ids)] for i in range(num_envs)]


def ports_for_envs(spec: str | None, num_envs: int) -> list[int]:
    num_envs = max(1, int(num_envs))
    ports = parse_ports(spec or "2223")
    if len(ports) == 1 and num_envs > 1:
        start = ports[0]
        return list(range(start, start + num_envs))
    if len(ports) < num_envs:
        raise ValueError(f"need {num_envs} ports, got {ports}")
    return ports[:num_envs]


def make_env_config(
    base_config: dict[str, Any],
    *,
    port: int,
    shot_id: str,
    max_steps: int,
    reference_mode: str,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_config))
    cfg["max_steps"] = int(max_steps)
    cfg.setdefault("predictor", {})
    cfg["predictor"]["host"] = "127.0.0.1"
    cfg["predictor"]["port"] = int(port)
    cfg["predictor"]["shot_id"] = str(shot_id)
    cfg["reference"] = {
        "mode": "hold" if reference_mode == "hold_initial" else "trajectory"
    }
    return cfg


def make_env(
    base_config: dict[str, Any],
    *,
    port: int,
    shot_id: str,
    max_steps: int,
    reference_mode: str,
    references: list[dict[str, Any]] | None,
    action_mode: str,
    obs_history_step: int,
    reward_weights: dict[str, float] | None = None,
    reset_perturbation: dict[str, Any] | None = None,
    fix_reset_params: bool = False,
    observation_noise: dict[str, Any] | None = None,
    delay_step: int = 0,
    delay_noise_std: dict[str, float] | None = None,
    seed: int = 0,
):

    def _init() -> gym.Env:
        env_configs = make_env_config(
            base_config,
            port=port,
            shot_id=shot_id,
            max_steps=max_steps,
            reference_mode=reference_mode,
        )
        reward_fn = CompetitionReward()
        env: gym.Env = HFMSimulator(env_configs, reward_fn=reward_fn)
        if reference_mode == "hold_initial":
            env = HoldInitialResetWrapper(
                env, shot_id=shot_id, reset_perturbation=reset_perturbation, seed=seed
            )
        else:
            if references is None:
                raise ValueError(
                    "trajectory reference is required when reference_mode='trajectory'"
                )
            env = EpisodeResetWrapper(
                env,
                references=references,
                shot_id=shot_id,
                reset_perturbation=reset_perturbation,
                fix_reset_params=fix_reset_params,
                seed=seed,
            )
        env = NormalizedAction7DTo12DWrapper(env, action_mode=action_mode)
        env = RewardInfoWrapper(env)
        env = ReferenceObservationWrapper(
            env,
            obs_history_step=obs_history_step,
            include_xpt=True,
            action_mode=action_mode,
        )
        return env

    return _init


def parse_net_arch(value: Any) -> dict[str, list[int]]:
    if isinstance(value, dict):
        pi = value.get("pi", value.get("policy", [256, 256]))
        vf = value.get("vf", value.get("value", pi))
        return {"pi": parse_net_arch(pi)["pi"], "vf": parse_net_arch(vf)["pi"]}
    if isinstance(value, str):
        widths = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        widths = [int(width) for width in list(value or [256, 256])]
    if not widths or any((width <= 0 for width in widths)):
        raise ValueError(f"net_arch must contain positive widths, got {value!r}")
    return {"pi": widths, "vf": list(widths)}


def parse_activation_fn(value: Any):
    import torch

    name = str(value or "tanh").strip().lower()
    activations = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "linear": torch.nn.Identity,
    }
    if name not in activations:
        raise ValueError(
            f"activation_fn must be one of {sorted(activations)}, got {value!r}"
        )
    return activations[name]


def build_policy_kwargs(ppo: dict[str, Any] | DictConfig) -> dict[str, Any]:
    ppo_dict = _to_dict(ppo) or {}
    raw = dict(ppo_dict.get("policy_kwargs") or {})
    net_arch = raw.pop("net_arch", ppo_dict.get("net_arch", [256, 256]))
    activation_fn = raw.pop("activation_fn", ppo_dict.get("activation_fn", "tanh"))
    log_std_init = raw.pop("log_std_init", ppo_dict.get("log_std_init", -1.0))
    policy_kwargs = dict(raw)
    policy_kwargs["net_arch"] = parse_net_arch(net_arch)
    policy_kwargs["activation_fn"] = parse_activation_fn(activation_fn)
    policy_kwargs["log_std_init"] = float(log_std_init)
    return policy_kwargs


def build_ppo_init_kwargs(
    ppo: dict[str, Any] | DictConfig
) -> tuple[str, dict[str, Any]]:
    """从 ppo 配置提取 SB3 PPO.__init__ 接受的参数。

    policy_kwargs 单独构建后嵌套传入；total_timesteps / lr_anneal /
    warmup_updates / progress_bar 等非 PPO 构造参数被排除。
    """
    ppo_dict = _to_dict(ppo) or {}
    policy = str(ppo_dict.get("policy", "MlpPolicy"))
    kwargs: dict[str, Any] = {
        "learning_rate": float(ppo_dict["learning_rate"]),
        "n_steps": int(ppo_dict["n_steps"]),
        "batch_size": int(ppo_dict["batch_size"]),
        "n_epochs": int(ppo_dict["n_epochs"]),
        "gamma": float(ppo_dict["gamma"]),
        "gae_lambda": float(ppo_dict.get("gae_lambda", 0.95)),
        "clip_range": float(ppo_dict.get("clip_range", 0.2)),
        "normalize_advantage": bool(ppo_dict.get("normalize_advantage", True)),
        "ent_coef": float(ppo_dict.get("ent_coef", 0.0)),
        "vf_coef": float(ppo_dict.get("vf_coef", 0.5)),
        "max_grad_norm": float(ppo_dict.get("max_grad_norm", 0.5)),
        "use_sde": bool(ppo_dict.get("use_sde", False)),
        "sde_sample_freq": int(ppo_dict.get("sde_sample_freq", -1)),
        "policy_kwargs": build_policy_kwargs(ppo_dict),
        "verbose": int(ppo_dict.get("verbose", 1)),
        "seed": None if ppo_dict.get("seed") is None else int(ppo_dict["seed"]),
        "device": str(ppo_dict.get("device", "auto")),
    }
    for key in ("clip_range_vf", "target_kl", "tensorboard_log"):
        if ppo_dict.get(key) is not None:
            kwargs[key] = ppo_dict[key]
    return (policy, kwargs)


def _resolve_save_dir() -> Path:
    """从 Hydra 运行时取出 run.dir，相对路径则挂在项目根目录下。"""
    run_dir = Path(HydraConfig.get().run.dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return run_dir


def build_training_context(cfg: DictConfig):
    """从 Hydra cfg 构建训练上下文，返回 (cfg, references, base_config, env_specs)。"""
    OmegaConf.set_struct(cfg, False)
    references = None
    if cfg.target.reference_mode != "hold_initial":
        if cfg.target.reference_variants_enabled:
            reference_path = _rooted_path(cfg.target.reference_variants_path)
        else:
            reference_path = _rooted_path(
                (
                    cfg.target.default_reference_path
                    if cfg.target.default_reference_path is not None
                    else None
                ),
                default_reference_path(),
            )
        if reference_path is None:
            raise ValueError("reference_path could not be resolved")
        references = build_reference(reference_path, max_steps=int(cfg.max_steps))
    reset_perturbation_range = dict()
    if cfg.reset_perturbation.enabled:
        reset_perturbation_range["enabled"] = True
        reset_perturbation_range["apply_mode"] = cfg.reset_perturbation.apply_mode
        reset_perturbation_range["signeo_scale_range"] = (
            cfg.reset_perturbation.signeo_scale_range
        )
        reset_perturbation_range["bp_scale_range"] = (
            cfg.reset_perturbation.bp_scale_range
        )
        reset_perturbation_range["q0_scale_range"] = (
            cfg.reset_perturbation.q0_scale_range
        )
    env_config_path = _rooted_path(
        cfg.env_config, ROOT / "configs" / "env_default.yaml"
    )
    if env_config_path is None:
        raise ValueError("env_config_path could not be resolved")
    base_config = load_yaml(env_config_path)
    fallback_shot_id = str(cfg.shot_id)
    ports_spec = str(cfg.ports)
    num_envs = int(cfg.num_envs)
    shot_groups = _to_dict(cfg.shot_groups)
    env_shot_ids = assign_shots_to_envs(shot_groups, num_envs, fallback_shot_id)
    ports = ports_for_envs(ports_spec, num_envs)
    base_seed = 0 if cfg.ppo.get("seed") is None else int(cfg.ppo["seed"])
    env_specs = []
    for i in range(num_envs):
        env_spec = {
            "index": i,
            "port": ports[i],
            "shot_id": env_shot_ids[i],
            "seed": base_seed + i,
        }
        env_specs.append(env_spec)
    cfg.save_dir = _resolve_save_dir()
    return (cfg, references, reset_perturbation_range, base_config, env_specs)


def run_smoke_rollout(cfg: DictConfig) -> int:
    cfg, references, reset_perturbation_range, base_config, env_specs = (
        build_training_context(cfg)
    )
    max_steps = cfg.max_steps
    env_spec = env_specs[0]
    env = make_env(
        base_config,
        port=env_spec["port"],
        shot_id=env_spec["shot_id"],
        max_steps=max_steps,
        reference_mode=cfg.target.reference_mode,
        references=references,
        action_mode=cfg.actions.mode,
        obs_history_step=cfg.features.history_steps,
        reward_weights=cfg.reward_weights,
        reset_perturbation=reset_perturbation_range,
        fix_reset_params=cfg.reset_perturbation.fix_reset_params,
        observation_noise=cfg.features.noise,
        delay_step=int(cfg.features.get("delay_step", 0) or 0),
        delay_noise_std=cfg.features.get("delay_noise_std", None),
        seed=env_spec["seed"],
    )()
    try:
        obs, info = env.reset(seed=int(cfg.ppo["seed"]))
        print(
            f"F2 reset ok: shot_id={info['shot_id']}, port={env_spec['port']}, reference_mode={info.get('reference_mode')}, obs_shape={obs.shape}"
        )
        for step in range(100):
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"F2 step {step}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


def run_training(cfg: DictConfig) -> int:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecMonitor,
            VecNormalize,
        )
    except ImportError:
        print("Missing dependencies: pip install stable-baselines3 torch")
        return 1
    wandb_run = None
    tensorboard_log_path = None
    if bool(cfg.wandb.enabled):
        try:
            import wandb

            wandb_kwargs = {"sync_tensorboard": True}
            wandb_kwargs["project"] = str(cfg.wandb.project)
            wandb_kwargs["name"] = str(cfg.output_dir)
            wandb_kwargs["tags"] = [str(tag) for tag in cfg.wandb.tags]
            wandb_run = wandb.init(**wandb_kwargs)
            tensorboard_log_path = f"runs/{wandb_run.id}"
            print(f"WandB run initialized: {wandb_run.name}")
            print(
                f"TensorBoard logs will be synced to wandb from: {tensorboard_log_path}"
            )
        except ImportError:
            print("WandB not installed, skipping wandb logging")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
    cfg, references, reset_perturbation_range, base_config, env_specs = (
        build_training_context(cfg)
    )
    max_steps = cfg.max_steps
    save_dir = cfg.save_dir
    ppo = cfg.ppo
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "configs.pkl", "wb") as f:
        pickle.dump(cfg, f)
    (save_dir / "f2_features.metadata.json").write_text(
        json.dumps(
            reference_metadata(
                action_mode=str(cfg.actions.mode),
                norm_obs=bool(cfg.vecnormalize.enabled and cfg.vecnormalize.norm_obs),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env_fns = [
        make_env(
            base_config,
            port=env_spec["port"],
            shot_id=env_spec["shot_id"],
            max_steps=max_steps,
            reference_mode=cfg.target.reference_mode,
            references=references,
            action_mode=cfg.actions.mode,
            obs_history_step=cfg.features.history_steps,
            reward_weights=cfg.reward_weights,
            reset_perturbation=reset_perturbation_range,
            fix_reset_params=cfg.reset_perturbation.fix_reset_params,
            observation_noise=cfg.features.noise,
            delay_step=int(cfg.features.get("delay_step", 0) or 0),
            delay_noise_std=cfg.features.get("delay_noise_std", None),
            seed=env_spec["seed"],
        )
        for env_spec in env_specs
    ]
    vec_env = (
        DummyVecEnv(env_fns)
        if len(env_fns) == 1
        else SubprocVecEnv(env_fns, start_method="spawn")
    )
    print(
        f"env observation_space: {vec_env.observation_space.shape}, action_space: {vec_env.action_space.shape} (reference_only_v1, no history, action_mode={cfg.actions.mode})"
    )
    vec_env = VecMonitor(vec_env, filename=str(save_dir / "monitor.csv"))
    vecnormalize_cfg = cfg.vecnormalize
    resume_cfg = cfg.resume
    if bool(vecnormalize_cfg.enabled):
        vecnormalize_path = _rooted_path(
            vecnormalize_cfg.load_path or resume_cfg.vecnormalize_path
        )
        if vecnormalize_path is not None and bool(resume_cfg.enabled):
            if not vecnormalize_path.exists():
                raise FileNotFoundError(
                    f"VecNormalize state not found: {vecnormalize_path}"
                )
            vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
            print(f"loaded VecNormalize: {vecnormalize_path}")
            vec_env.training = True
            vec_env.norm_reward = bool(vecnormalize_cfg.get("norm_reward", True))
        else:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=bool(vecnormalize_cfg.norm_obs),
                norm_reward=bool(vecnormalize_cfg.norm_reward),
                clip_obs=float(vecnormalize_cfg.clip_obs),
                clip_reward=float(vecnormalize_cfg.clip_reward),
                gamma=float(vecnormalize_cfg.gamma or ppo.gamma),
                epsilon=float(vecnormalize_cfg.epsilon),
            )
        print(
            f"VecNormalize: norm_obs={vec_env.norm_obs}, norm_reward={vec_env.norm_reward}, clip_obs={vec_env.clip_obs}, clip_reward={vec_env.clip_reward}"
        )
    assignments = ", ".join(
        (f"env{spec['index']}:{spec['shot_id']}@{spec['port']}" for spec in env_specs)
    )
    print(
        f"training F2: reference_mode={cfg.target.reference_mode}, max_steps={max_steps}, assignments=[{assignments}]"
    )
    policy, ppo_kwargs = build_ppo_init_kwargs(ppo)
    lr_anneal = ppo.lr_anneal
    warmup_updates = int(ppo.warmup_updates)
    n_envs = len(env_fns)
    n_steps = int(ppo_kwargs.get("n_steps", 2048))
    total_timesteps = int(ppo.total_timesteps)
    timesteps_per_update = n_steps * n_envs
    initial_lr = float(ppo_kwargs["learning_rate"])
    lr_schedule = warmup_cosine_schedule(
        initial_lr=initial_lr,
        timesteps_per_update=timesteps_per_update,
        total_timesteps=total_timesteps,
        warmup_updates=warmup_updates,
        anneal=lr_anneal,
    )
    ppo_kwargs["learning_rate"] = lr_schedule
    if lr_anneal:
        print(
            f"Using warmup + cosine annealing: initial_lr={initial_lr}, warmup_updates={warmup_updates}, timesteps_per_update={timesteps_per_update}"
        )
    else:
        print(
            f"Using warmup only (no annealing): initial_lr={initial_lr}, warmup_updates={warmup_updates}, timesteps_per_update={timesteps_per_update}"
        )
    if tensorboard_log_path is not None:
        ppo_kwargs["tensorboard_log"] = tensorboard_log_path
    elif ppo_kwargs.get("tensorboard_log") is not None:
        ppo_kwargs["tensorboard_log"] = str(_rooted_path(ppo_kwargs["tensorboard_log"]))
    if bool(resume_cfg.enabled):
        resume_path = _rooted_path(resume_cfg.model_path)
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"resume model not found: {resume_path}")
        model = PPO.load(
            str(resume_path),
            env=vec_env,
            device=str(ppo.device),
            custom_objects=ppo_kwargs,
        )
        print(f"resumed: {resume_path} with PPO overrides from train config")
    else:
        model = PPO(policy, vec_env, **ppo_kwargs)
    callbacks = []
    reward_log_interval = int(cfg.num_envs * ppo.n_steps)
    callbacks.append(
        RewardComponentsCallback(
            log_dir=save_dir / "logs", log_interval=reward_log_interval
        )
    )
    checkpoint_cfg = cfg.checkpoint
    if bool(checkpoint_cfg.enabled):
        callbacks.append(
            PlasmaCheckpointCallback(
                save_freq=checkpoint_cfg.save_freq,
                save_path=save_dir / "checkpoints",
                name_prefix="ppo_f2",
                save_vecnormalize=bool(checkpoint_cfg.save_vecnormalize),
            )
        )
    reset_num_timesteps = bool(resume_cfg.enabled)
    try:
        model.learn(
            total_timesteps=int(ppo.total_timesteps),
            callback=callbacks if callbacks else None,
            progress_bar=bool(ppo.progress_bar),
            reset_num_timesteps=reset_num_timesteps,
        )
        out = save_dir / "ppo_f2_latest"
        model.save(str(out))
        vecnormalize = find_vecnormalize(vec_env)
        if vecnormalize is not None and bool(vecnormalize_cfg.get("save", True)):
            vecnormalize.training = False
            vecnormalize.norm_reward = False
            vecnormalize.save(str(save_dir / "vecnormalize.pkl"))
            save_vecnormalize_stats(vecnormalize, save_dir / "vecnormalize_stats.npz")
            print(f"VecNormalize saved: {save_dir / 'vecnormalize.pkl'}")
        print(f"saved: {out}.zip")
        print(f"metadata: {save_dir / 'f2_features.metadata.json'}")
    finally:
        vec_env.close()
        if wandb_run is not None:
            try:
                wandb_run.finish()
                print("WandB run finished")
            except Exception as e:
                print(f"Failed to finish WandB run: {e}")
    return 0


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> int:
    OmegaConf.set_struct(cfg, False)
    if bool(cfg.smoke_rollout):
        return run_smoke_rollout(cfg)
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
