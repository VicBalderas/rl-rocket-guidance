import os
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize
from environment.rocket_env import RocketInterceptEnv


# =============================================================================
# Curriculum stages
# =============================================================================
#
# Each stage maps to a target_motion value in RocketInterceptEnv v4.
# Obstacles are handled internally by the environment:
#   static / simple / simple_fast          →  1 passive edge obstacle
#   dual_axis_evasive                      →  1 edge + 1 near-target obstacle
#   evasive                                →  2 near-target obstacles
#   evasive_flares                         →  2-3 near-target obstacles
#
# Stages 0-2 are UNCHANGED from the original training.
# Stages 3-5 are redesigned with perpendicular evasion.
#
CURRICULUM_STAGES = [
    dict(target_motion="static"),              # 0 — basic intercept geometry
    dict(target_motion="simple"),              # 1 — 1D lead pursuit, moderate speed
    dict(target_motion="simple_fast"),         # 2 — 1D lead pursuit, fast
    dict(target_motion="dual_axis_evasive"),   # 3 — 2D + mild perpendicular dodge
    dict(target_motion="evasive"),             # 4 — fast + aggressive perpendicular dodge
    dict(target_motion="evasive_flares"),      # 5 — aggressive dodge + obstacle luring
]


# =============================================================================
# Folder structure
# =============================================================================
#
#   models/
#   ├── stage_0/
#   │   ├── final_model.zip
#   │   ├── vec_normalize.pkl
#   │   └── checkpoints/
#   │       ├── model_500000.zip
#   │       ├── stats_500000.pkl
#   │       └── ...
#   ├── stage_1/
#   │   ├── final_model.zip
#   │   ├── vec_normalize.pkl
#   │   └── checkpoints/
#   │       └── ...
#   └── ...
#

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")


def stage_dir(stage_index):
    return os.path.join(MODELS_ROOT, f"stage_{stage_index}")


def checkpoint_dir(stage_index):
    return os.path.join(stage_dir(stage_index), "checkpoints")


def final_model_path(stage_index):
    """Returns the path stem (no .zip) for the final model of a stage."""
    return os.path.join(stage_dir(stage_index), "final_model")


def final_stats_path(stage_index):
    return os.path.join(stage_dir(stage_index), "vec_normalize.pkl")


# =============================================================================
# Environment factory
# =============================================================================

def make_env(**env_kwargs):
    def _init():
        return RocketInterceptEnv(render_mode=None, **env_kwargs)
    return _init


def build_envs(n_train_envs=16, **env_kwargs):
    train_env_raw = make_vec_env(make_env(**env_kwargs), n_envs=n_train_envs)
    eval_env_raw  = make_vec_env(make_env(**env_kwargs), n_envs=1)
    return train_env_raw, eval_env_raw


# =============================================================================
# Callbacks
# =============================================================================

class PeriodicCheckpointCallback(BaseCallback):
    """
    Saves the model and VecNormalize stats every `save_every` timesteps
    into the stage's checkpoints/ folder.
    """
    def __init__(self, save_every, checkpoint_path):
        super().__init__()
        self.save_every = save_every
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.save_every == 0 and self.num_timesteps > 0:
            model_path = os.path.join(
                self.checkpoint_path, f"model_{self.num_timesteps}"
            )
            stats_path = os.path.join(
                self.checkpoint_path, f"stats_{self.num_timesteps}.pkl"
            )
            self.model.save(model_path)
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(stats_path)
            print(f"  [Checkpoint] Saved at {self.num_timesteps} steps")
        return True


class SyncEvalNormCallback(BaseCallback):
    """
    Periodically copies training VecNormalize obs_rms to the eval env
    so eval observations use up-to-date normalization statistics.
    """
    def __init__(self, eval_env, sync_every=10_000):
        super().__init__()
        self.eval_env = eval_env
        self.sync_every = sync_every

    def _on_step(self):
        if self.num_timesteps % self.sync_every == 0:
            train_vec = self.model.get_vec_normalize_env()
            if train_vec is not None:
                self.eval_env.obs_rms = copy.deepcopy(train_vec.obs_rms)
        return True


class HitRateLoggerCallback(BaseCallback):
    """
    Tracks episode outcomes and prints a summary every `log_every`
    episodes: hit %, obstacle %, OOB %, timeout %.
    """
    def __init__(self, log_every=200):
        super().__init__()
        self.log_every = log_every
        self.outcomes = {"hit": 0, "obstacle": 0, "oob": 0, "timeout": 0, "total": 0}

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "hit_target" not in info:
                continue

            done = (
                info.get("hit_target", False)
                or info.get("hit_obstacle", False)
                or info.get("out_of_bounds", False)
                or info.get("timeout", False)
            )
            if not done:
                continue

            self.outcomes["total"] += 1
            if info.get("hit_target"):
                self.outcomes["hit"] += 1
            elif info.get("hit_obstacle"):
                self.outcomes["obstacle"] += 1
            elif info.get("out_of_bounds"):
                self.outcomes["oob"] += 1
            elif info.get("timeout"):
                self.outcomes["timeout"] += 1

            if self.outcomes["total"] % self.log_every == 0:
                t = self.outcomes["total"]
                h = self.outcomes["hit"]
                o = self.outcomes["obstacle"]
                b = self.outcomes["oob"]
                to = self.outcomes["timeout"]
                print(
                    f"\n[HitRate] {t} eps | "
                    f"HIT {h/t*100:5.1f}% | "
                    f"OBS {o/t*100:5.1f}% | "
                    f"OOB {b/t*100:5.1f}% | "
                    f"TMO {to/t*100:5.1f}%\n"
                )
        return True


def build_callbacks(
    eval_env, stage_index,
    eval_freq=10_000, n_eval_episodes=50,
    checkpoint_every=500_000, log_every=200,
):
    """Constructs all callbacks for a training run."""
    # EvalCallback for logging only — no best-model saving.
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=None,
        verbose=1,
    )
    sync_cb     = SyncEvalNormCallback(eval_env)
    hit_rate_cb = HitRateLoggerCallback(log_every=log_every)
    ckpt_cb     = PeriodicCheckpointCallback(
        save_every=checkpoint_every,
        checkpoint_path=checkpoint_dir(stage_index),
    )

    return CallbackList([eval_cb, sync_cb, hit_rate_cb, ckpt_cb])


# =============================================================================
# Core training runner
# =============================================================================

def run_training(
    model, train_env, eval_env, stage_index,
    total_timesteps, reset_timesteps=True,
):
    """Runs training and saves the final model + stats to the stage folder."""
    os.makedirs(stage_dir(stage_index), exist_ok=True)

    callback = build_callbacks(eval_env, stage_index)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_timesteps,
    )

    # Save final model and stats.
    model_save = final_model_path(stage_index)
    stats_save = final_stats_path(stage_index)
    model.save(model_save)
    train_env.save(stats_save)

    print(f"\nTraining complete.")
    print(f"  Model: {model_save}.zip")
    print(f"  Stats: {stats_save}")


# =============================================================================
# Unified training loop
# =============================================================================

def loop(mode="A", stage=0):
    """
    Mode A — Train from scratch on a curriculum stage.
    Mode B — Fine-tune the previous stage's final model onto the next stage.

    Args:
        mode:  "A" = train from scratch, "B" = fine-tune from previous stage.
        stage: Index into CURRICULUM_STAGES (0-5).
    """

    mode = mode.upper()
    assert mode in ("A", "B"), "Mode must be 'A' or 'B'"
    assert 0 <= stage < len(CURRICULUM_STAGES), (
        f"stage must be 0-{len(CURRICULUM_STAGES)-1}, got {stage}"
    )

    # -------------------------------------------------------------------------
    # CONFIG BLOCK — edit this section only
    # -------------------------------------------------------------------------

    timesteps_a = 3_000_000    # Mode A total timesteps
    timesteps_b = 2_000_000    # Mode B total timesteps

    # Mode B fine-tuning hyperparameters
    learning_b = 3e-4  # 5e-5
    epochs_b = 10
    clip_b = 0.2
    entropy_b = 0.01

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------

    env_kwargs  = CURRICULUM_STAGES[stage]
    stage_label = f"stage {stage}: {env_kwargs['target_motion']}"

    # Sanity-check
    env_check = RocketInterceptEnv(**env_kwargs)
    check_env(env_check)
    env_check.close()
    print(f"Environment check passed.  ({stage_label})\n")

    if mode == "A":
        print(f"Mode A: Training from scratch — {stage_label}\n")

        train_env_raw, eval_env_raw = build_envs(**env_kwargs)

        train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=False)
        eval_env  = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False)
        eval_env.obs_rms  = copy.deepcopy(train_env.obs_rms)
        eval_env.training = False
        eval_env.norm_reward = False

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            device="cpu",
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
        )

        run_training(
            model=model,
            train_env=train_env,
            eval_env=eval_env,
            stage_index=stage,
            total_timesteps=timesteps_a,
            reset_timesteps=True,
        )

    else:  # Mode B
        prev_stage = stage - 1
        prev_model = final_model_path(prev_stage) + ".zip"
        prev_stats = final_stats_path(prev_stage)

        if not os.path.exists(prev_model):
            raise FileNotFoundError(
                f"Previous stage model not found: '{prev_model}'\n"
                f"Complete stage {prev_stage} first."
            )
        if not os.path.exists(prev_stats):
            raise FileNotFoundError(
                f"Previous stage stats not found: '{prev_stats}'\n"
                f"Complete stage {prev_stage} first."
            )

        print(
            f"Mode B: Fine-tuning stage {prev_stage} → {stage}\n"
            f"  Loading: {prev_model}\n"
            f"  Stats:   {prev_stats}\n"
            f"  Target:  {stage_label}\n"
        )

        train_env_raw, eval_env_raw = build_envs(**env_kwargs)

        train_env = VecNormalize.load(prev_stats, train_env_raw)
        train_env.training = True
        train_env.norm_reward = False

        eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False)
        eval_env.obs_rms  = copy.deepcopy(train_env.obs_rms)
        eval_env.training = False
        eval_env.norm_reward = False

        model = PPO.load(
            final_model_path(prev_stage),
            env=train_env,
            device="cpu",
            learning_rate=learning_b,
            n_steps=1024,
            batch_size=128,
            n_epochs=epochs_b,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_b,
            ent_coef=entropy_b,
            verbose=1,
        )

        run_training(
            model=model,
            train_env=train_env,
            eval_env=eval_env,
            stage_index=stage,
            total_timesteps=timesteps_b,
            reset_timesteps=False,
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    loop(mode="B", stage=3)