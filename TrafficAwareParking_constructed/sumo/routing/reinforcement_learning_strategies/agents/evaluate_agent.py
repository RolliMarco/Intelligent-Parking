from TrafficAwareParking_constructed.sumo.routing.reinforcement_learning_strategies.parking_enviroment.parking_env import ParkingEnv
from stable_baselines3 import DQN


# =========================================================
# Einstellungen
# =========================================================

MODEL_PATH = "dqn_parking_1000_episodes"


# =========================================================
# Environment erstellen
# =========================================================

env = ParkingEnv()


# =========================================================
# Trainiertes Modell laden
# =========================================================

model = DQN.load(
    MODEL_PATH,
    env=env
)


# =========================================================
# Neue Episode starten
# =========================================================

observation, info = env.reset()

terminated = False
truncated = False

total_reward = 0.0
step = 0


# =========================================================
# Episode ausführen
# =========================================================

while not (terminated or truncated):

    # DQN entscheidet
    action, _ = model.predict(
        observation,
        deterministic=True
    )

    print(
        f"Fahrzeug: {info['vehicle']} | "
        f"Aktion: {int(action)}"
    )

    # Aktion ausführen
    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward
    step += 1


# =========================================================
# Ergebnis
# =========================================================

print("\n======================================")
print("DQN Evaluation")
print("======================================")

print(
    f"Anzahl Schritte: {step}"
)

print(
    f"Gesamtreward: {total_reward:.4f}"
)


# =========================================================
# Environment schließen
# =========================================================

env.close()

print("\nEvaluation abgeschlossen.")