import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.rocket_env import RocketInterceptEnv


# =============================================================================
# Folder structure (mirrors train.py)
# =============================================================================

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")


def stage_dir(stage_index):
    return os.path.join(MODELS_ROOT, f"stage_{stage_index}")


def final_model_path(stage_index):
    return os.path.join(stage_dir(stage_index), "final_model")


def final_stats_path(stage_index):
    return os.path.join(stage_dir(stage_index), "vec_normalize.pkl")


def checkpoint_model_path(stage_index, timestep):
    return os.path.join(stage_dir(stage_index), "checkpoints", f"model_{timestep}")


def checkpoint_stats_path(stage_index, timestep):
    return os.path.join(stage_dir(stage_index), "checkpoints", f"stats_{timestep}.pkl")


# =============================================================================
# Curriculum stages (mirrors train.py)
# =============================================================================

CURRICULUM_STAGES = [
    dict(target_motion="static"),
    dict(target_motion="simple"),
    dict(target_motion="simple_fast"),
    dict(target_motion="dual_axis_evasive"),
    dict(target_motion="evasive"),
    dict(target_motion="evasive_flares"),
]


# =============================================================================
# Evaluate
# =============================================================================

def evaluate(
    stage=0,
    n_episodes=100,
    render=False,
    checkpoint=None,
    test_on_stage=None,
):
    """
    Evaluate a trained model.

    Args:
        stage:         Which stage's model to load (0-5).
        n_episodes:    Number of episodes to run.
        render:        If True, renders with pygame (slows evaluation).
        checkpoint:    If set, load a specific checkpoint timestep instead
                       of the final model. E.g. checkpoint=1000000.
        test_on_stage: If set, evaluate the model on a DIFFERENT stage's
                       environment. Useful for testing generalization.
                       E.g. load stage 0 model, test on stage 1 env.
    """
    # Resolve model and stats paths.
    if checkpoint is not None:
        model_path = checkpoint_model_path(stage, checkpoint)
        stats_path = checkpoint_stats_path(stage, checkpoint)
        source_label = f"stage {stage} checkpoint {checkpoint}"
    else:
        model_path = final_model_path(stage)
        stats_path = final_stats_path(stage)
        source_label = f"stage {stage} final"

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    # Resolve which environment to evaluate on.
    eval_stage = test_on_stage if test_on_stage is not None else stage
    env_kwargs = CURRICULUM_STAGES[eval_stage]
    env_label = f"stage {eval_stage}: {env_kwargs['target_motion']}"

    print(f"Evaluating: {source_label}")
    print(f"Environment: {env_label}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print()

    # Build environment.
    render_mode = "human" if render else None

    def make_env():
        return RocketInterceptEnv(render_mode=render_mode, **env_kwargs)

    env_raw = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env_raw)
    env.training = False
    env.norm_reward = False

    # Load model.
    model = PPO.load(model_path, env=env, device="cpu")

    # Run episodes.
    outcomes = {"hit": 0, "obstacle": 0, "oob": 0, "timeout": 0}
    episode_rewards = []
    episode_lengths = []
    distances_at_end = []

    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0

    episodes_done = 0

    while episodes_done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward[0]
        ep_length += 1

        if render:
            env.envs[0].render()

        if done[0]:
            ep_info = info[0]

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            distances_at_end.append(ep_info.get("distance_to_target", -1))

            if ep_info.get("hit_target"):
                outcomes["hit"] += 1
            elif ep_info.get("hit_obstacle"):
                outcomes["obstacle"] += 1
            elif ep_info.get("out_of_bounds"):
                outcomes["oob"] += 1
            elif ep_info.get("timeout"):
                outcomes["timeout"] += 1

            episodes_done += 1
            ep_reward = 0.0
            ep_length = 0
            obs = env.reset()

    env.close()

    # Print results.
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    distances = np.array(distances_at_end)

    total = n_episodes
    print("=" * 55)
    print(f"  RESULTS — {source_label} on {env_label}")
    print("=" * 55)
    print(f"  Episodes:    {total}")
    print(f"  Hit rate:    {outcomes['hit'] / total * 100:5.1f}%  ({outcomes['hit']}/{total})")
    print(f"  Obstacle:    {outcomes['obstacle'] / total * 100:5.1f}%  ({outcomes['obstacle']}/{total})")
    print(f"  Out of bounds: {outcomes['oob'] / total * 100:5.1f}%  ({outcomes['oob']}/{total})")
    print(f"  Timeout:     {outcomes['timeout'] / total * 100:5.1f}%  ({outcomes['timeout']}/{total})")
    print("-" * 55)
    print(f"  Reward:      {rewards.mean():.2f} +/- {rewards.std():.2f}")
    print(f"  Ep length:   {lengths.mean():.1f} +/- {lengths.std():.1f}")
    print(f"  Final dist:  {distances.mean():.1f} +/- {distances.std():.1f}")
    print("=" * 55)


def evaluate_human(stage=0, n_episodes=10, checkpoint=None):
    """
    Human-controlled target evaluation.
    You control the target with WASD/arrow keys while the trained
    policy controls the missile. The ultimate stress test.

    Args:
        stage:      Which stage's model to load.
        n_episodes: Number of rounds to play.
        checkpoint: Optional specific checkpoint timestep.
    """
    # Resolve model and stats paths.
    if checkpoint is not None:
        model_path = checkpoint_model_path(stage, checkpoint)
        stats_path = checkpoint_stats_path(stage, checkpoint)
        source_label = f"stage {stage} checkpoint {checkpoint}"
    else:
        model_path = final_model_path(stage)
        stats_path = final_stats_path(stage)
        source_label = f"stage {stage} final"

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    print(f"Human Evaluation: {source_label}")
    print(f"Episodes: {n_episodes}")
    print("Control the target with WASD or arrow keys!")
    print()

    # Build environment with human control.
    def make_env():
        return RocketInterceptEnv(render_mode="human", target_motion="human")

    env_raw = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env_raw)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device="cpu")

    outcomes = {"hit": 0, "survived": 0, "total": 0}
    obs = env.reset()
    ep_length = 0
    episodes_done = 0

    while episodes_done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_length += 1
        env.envs[0].render()

        if done[0]:
            ep_info = info[0]
            outcomes["total"] += 1

            if ep_info.get("hit_target"):
                outcomes["hit"] += 1
                result = "MISSILE WINS"
            else:
                outcomes["survived"] += 1
                result = "YOU SURVIVED"

            episodes_done += 1
            print(f"  Round {episodes_done}/{n_episodes}: {result} ({ep_length} steps)")
            ep_length = 0
            obs = env.reset()

    env.close()

    t = outcomes["total"]
    print()
    print("=" * 40)
    print(f"  Missile hit rate: {outcomes['hit'] / t * 100:.1f}%")
    print(f"  You survived:     {outcomes['survived'] / t * 100:.1f}%")
    print("=" * 40)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # evaluate(
    #     stage=3,
    #     n_episodes=10,
    #     render=True,
    #     checkpoint=None,
    #     test_on_stage=2,
    # )

    # To run human evaluation, uncomment below:
    evaluate_human(stage=2, n_episodes=10)