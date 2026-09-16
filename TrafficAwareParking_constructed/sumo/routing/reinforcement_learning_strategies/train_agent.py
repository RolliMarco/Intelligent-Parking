from parking_env import ParkingEnv

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


# =========================================================
# Einstellungen
# =========================================================

NUM_EPISODES = 10

MODEL_PATH = "dqn_parking_10_episodes"


# =========================================================
# Callback:
# Training nach 10 Episoden beenden
# =========================================================

class StopAfterEpisodesCallback(BaseCallback):

    def __init__(self, max_episodes, verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:

        # Bei einer einzelnen Environment enthält
        # "dones" ein boolesches Ergebnis.
        dones = self.locals.get("dones")

        if dones is not None:

            if dones[0]:
                self.episode_count += 1

                print(
                    f"\n========== "
                    f"Episode {self.episode_count}"
                    f" / {self.max_episodes} "
                    f"abgeschlossen =========="
                )

                if self.episode_count >= self.max_episodes:

                    print(
                        "\n10 Episoden erreicht. "
                        "Training wird beendet."
                    )

                    return False

        return True


# =========================================================
# Environment
# =========================================================

env = ParkingEnv()


# =========================================================
# DQN-Modell
# =========================================================

model = DQN(
    policy="MlpPolicy",
    env=env,

    # Ausgabe im Terminal
    verbose=1,

    # Kleine Replay-Memory für den ersten Test
    buffer_size=10_000,

    # Anzahl Schritte, bevor das eigentliche
    # Lernen beginnt
    learning_starts=100,

    # Wie oft ein Training-Update durchgeführt wird
    train_freq=1,

    # Ein Gradientenschritt pro Update
    gradient_steps=1,

    # CPU verwenden
    device="cpu",

    # Reproduzierbarer Start
    seed=42
)


# =========================================================
# Callback
# =========================================================

callback = StopAfterEpisodesCallback(
    max_episodes=NUM_EPISODES
)


# =========================================================
# Training
# =========================================================

print("\n======================================")
print("Starte DQN-Training")
print(f"Ziel: {NUM_EPISODES} Episoden")
print("======================================\n")


model.learn(
    total_timesteps=100_000,
    callback=callback,
    progress_bar=True
)


# =========================================================
# Modell speichern
# =========================================================

model.save(MODEL_PATH)

print("\n======================================")
print("Training abgeschlossen.")
print(f"Modell gespeichert als: {MODEL_PATH}.zip")
print("======================================\n")


# =========================================================
# Environment schließen
# =========================================================

env.close()