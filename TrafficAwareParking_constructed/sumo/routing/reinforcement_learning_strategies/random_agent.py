from parking_env import ParkingEnv


# =========================================================
# Einstellungen
# =========================================================

NUM_EPISODES = 20


# =========================================================
# Environment erstellen
# =========================================================

env = ParkingEnv()


# =========================================================
# Training / Test
# =========================================================

episode_rewards = []

for episode in range(NUM_EPISODES):

    print(
        f"\n========== EPISODE {episode + 1} "
        f"/ {NUM_EPISODES} =========="
    )

    # -----------------------------------------------------
    # Neue Episode starten
    # -----------------------------------------------------

    observation, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0.0
    step = 0

    # -----------------------------------------------------
    # Episode durchlaufen
    # -----------------------------------------------------

    while not (terminated or truncated):

        # Zufällige Aktion wählen
        #
        # 0 -> P1
        # 1 -> P2
        # 2 -> P3

        action = env.action_space.sample()

        # Aktion ausführen
        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        # Gesamtreward aufsummieren
        total_reward += reward

        step += 1

    # -----------------------------------------------------
    # Ergebnis der Episode
    # -----------------------------------------------------

    episode_rewards.append(total_reward)

    print(
        f"Episode beendet | "
        f"Schritte: {step} | "
        f"Gesamtreward: {total_reward:.4f}"
    )


# =========================================================
# Gesamtauswertung
# =========================================================

if episode_rewards:

    average_reward = (
        sum(episode_rewards)
        / len(episode_rewards)
    )

    print("\n===================================")
    print("Random-Agent Ergebnis")
    print("===================================")

    print(
        f"Anzahl Episoden: {len(episode_rewards)}"
    )

    print(
        f"Durchschnittlicher Reward: "
        f"{average_reward:.4f}"
    )

    print(
        f"Bestes Ergebnis: "
        f"{max(episode_rewards):.4f}"
    )

    print(
        f"Schlechtestes Ergebnis: "
        f"{min(episode_rewards):.4f}"
    )


# =========================================================
# Environment schließen
# =========================================================

env.close()

print("\nRandom-Agent Test abgeschlossen.")