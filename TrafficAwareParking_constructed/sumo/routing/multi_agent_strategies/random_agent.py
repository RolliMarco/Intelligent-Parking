from multi_agent_environment import ParkingMultiAgentEnv


# =========================================================
# Einstellungen
# =========================================================

NUM_EPISODES = 20
MAX_STEPS_PER_EPISODE = 10000


# =========================================================
# Environment erstellen
# =========================================================

env = ParkingMultiAgentEnv(
    max_agent_slots=2000
)


# =========================================================
# Ergebnisse über alle Episoden
# =========================================================

episode_rewards = []


# =========================================================
# Episoden
# =========================================================

for episode in range(NUM_EPISODES):

    print(
        f"\n========== EPISODE {episode + 1} "
        f"/ {NUM_EPISODES} =========="
    )

    # -----------------------------------------------------
    # Neue Episode starten
    # -----------------------------------------------------

    env.reset()

    total_episode_reward = 0.0
    step_count = 0

    action_counts = {
        0: 0,
        1: 0,
        2: 0
    }

    # -----------------------------------------------------
    # Multi-Agent-Episode durchlaufen
    # -----------------------------------------------------

    for agent in env.agent_iter(
        max_iter=MAX_STEPS_PER_EPISODE
    ):

        observation, reward, termination, truncation, info = (
            env.last()
        )

        if reward != 0:
            print(
                f"REWARD AUS env.last() | "
                f"Agent: {agent} | "
                f"Reward: {reward:.4f}"
            )

        # -------------------------------------------------
        # Reward des aktuellen Agenten aufsummieren
        # -------------------------------------------------

        total_episode_reward += reward

        print("total_episode_reward:", total_episode_reward)

        # -------------------------------------------------
        # Prüfen, ob der Agent bereits fertig ist
        # -------------------------------------------------

        if termination or truncation:

            action = None

        else:

            # -------------------------------------------------
            # Zufällige Aktion
            #
            # 0 -> P1
            # 1 -> P2
            # 2 -> P3
            # -------------------------------------------------

            action = env.action_space(
                agent
            ).sample()

            action_counts[action] += 1

            print(
                f"Agent: {agent} | "
                f"Fahrzeug: {info.get('vehicle_id')} | "
                f"Aktion: {action} "
                f"(0=P1, 1=P2, 2=P3)"
            )

        # -------------------------------------------------
        # Aktion an die Environment übergeben
        # -------------------------------------------------

        env.step(action)

        step_count += 1

    # -----------------------------------------------------
    # Ergebnis der Episode speichern
    # -----------------------------------------------------

    episode_rewards.append(
        total_episode_reward
    )

    print(
        f"\nEpisode beendet | "
        f"PettingZoo-Schritte: {step_count}"
    )

    print(
        f"Gesamtreward: "
        f"{total_episode_reward:.4f}"
    )

    print(
        "Aktionsverteilung: "
        f"P1={action_counts[0]}, "
        f"P2={action_counts[1]}, "
        f"P3={action_counts[2]}"
    )


# =========================================================
# Gesamtauswertung
# =========================================================

if episode_rewards:

    average_reward = (
        sum(episode_rewards)
        / len(episode_rewards)
    )

    print("\n======================================")
    print("Random-Agent Ergebnis")
    print("======================================")

    print(
        f"Anzahl Episoden: "
        f"{len(episode_rewards)}"
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