import ray
import numpy as np

from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env

from multi_agent_environment_2 import ParkingMultiAgentEnv2
from gymnasium import spaces


# =========================================================
# Einstellungen
# =========================================================

NUM_EPISODES = 100

ENV_NAME = "traffic_aware_parking"

CHECKPOINT_DIR = "./checkpoints/dqn_parking_100_episodes"


# =========================================================
# PettingZoo-Environment für RLlib erzeugen
# =========================================================

def env_creator(config):
    return ParkingMultiAgentEnv2(
        max_agent_slots=2000
    )

# Environment bei RLlib registrieren
register_env(
    ENV_NAME,
    env_creator
)

# =========================================================
# Ray starten
# =========================================================

ray.init()

# =========================================================
# Test-Environment erzeugen
# =========================================================

observation_space = spaces.Box(
    low=0.0,
    high=1.0,
    shape=(12,),
    dtype=np.float32
)

action_space = spaces.Discrete(3)

# =========================================================
# Gemeinsame Policy
# =========================================================
#
# Alle Fahrzeug-Agenten verwenden dieselbe Policy.
#
# vehicle_agent_0 -> shared_policy
# vehicle_agent_1 -> shared_policy
# vehicle_agent_2 -> shared_policy
# ...
# =========================================================

policies = {
    "shared_policy": (
        None,
        observation_space,
        action_space,
        {}
    )
}

def policy_mapping_fn(
    agent_id,
    episode,
    **kwargs
):
    return "shared_policy"

# =========================================================
# DQN-Konfiguration
# =========================================================

config = (
    DQNConfig()

    .environment(
        env=ENV_NAME
    )

    .env_runners(
        num_env_runners=0
    )

    .framework(
        "torch"
    )

    .training(
        train_batch_size_per_learner=256,
        num_steps_sampled_before_learning_starts=1000,
        lr=0.0005,
        target_network_update_freq=1000,
        replay_buffer_config={
            "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
            "capacity": 50_000,
            "alpha": 0.6,
            "beta": 0.4,
    }
    )

    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=[
            "shared_policy"
        ]
    )
)

algo = config.build_algo()


# =========================================================
# Training
# =========================================================

print("\n==========================================")
print("Starte Multi-Agent-DQN-Training")
print(f"Ziel: mindestens {NUM_EPISODES} Episoden")
print("Policy: shared_policy")
print("==========================================\n")


episodes_completed = 0
training_iteration = 0


while episodes_completed < NUM_EPISODES:

    print("\n===== REPLAY BUFFER CONFIG =====")
    print(config.replay_buffer_config)

    result = algo.train()

    training_iteration += 1
    print("Training iteration:", training_iteration)

    # -----------------------------------------------------
    # RLlib speichert diese Kennzahl je nach API-Version
    # entweder direkt im Ergebnis oder unter env_runners.
    # -----------------------------------------------------

    episodes_completed = result.get(
        "num_episodes_lifetime",
        result.get(
            "env_runners",
            {}
        ).get(
            "num_episodes_lifetime",
            0
        )
    )

    # -----------------------------------------------------
    # Durchschnittlicher Episoden-Reward
    # -----------------------------------------------------

    episode_reward_mean = result.get(
        "env_runners",
        {}
    ).get(
        "episode_return_mean",
        result.get(
            "episode_reward_mean",
            0.0
        )
    )

    print(
        f"Training Iteration: {training_iteration} | "
        f"Episoden: {episodes_completed} | "
        f"Reward-Mittelwert: "
        f"{episode_reward_mean:.4f}"
    )


# =========================================================
# Modell speichern
# =========================================================

checkpoint = algo.save(
    CHECKPOINT_DIR
)


print("\n==========================================")
print("Training abgeschlossen.")
print(f"Episoden: {episodes_completed}")
print(f"Checkpoint: {checkpoint}")
print("==========================================\n")


# =========================================================
# Aufräumen
# =========================================================

algo.stop()

ray.shutdown()

print("Multi-Agent-DQN beendet.")
