from multi_agent_environment_2 import ParkingMultiAgentEnv2


# =========================================================
# Einstellungen
# =========================================================

NUM_EPISODES = 20
MAX_STEPS_PER_EPISODE = 10000


# =========================================================
# Environment erstellen
# =========================================================

env = ParkingMultiAgentEnv2(
    max_agent_slots=2000
)


# =========================================================
# Ergebnisse über alle Episoden
# =========================================================

episode_rewards = []
episode_agent_rewards = []


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

    observations, infos = env.reset()

    total_episode_reward = 0.0
    total_completed_agents = 0
    step_count = 0

    action_counts = {
        0: 0,
        1: 0,
        2: 0
    }

    # -----------------------------------------------------
    # Episode durchlaufen
    #
    # RLlib MultiAgentEnv:
    # observations = {agent_id: observation}
    # action_dict  = {agent_id: action}
    # -----------------------------------------------------

    episode_finished = False

    while not episode_finished:

        if step_count >= MAX_STEPS_PER_EPISODE:

            print(
                "Maximale Schrittzahl erreicht."
            )
            break

        # -------------------------------------------------
        # Aktionen für alle aktuell entscheidungsbereiten
        # Agenten erzeugen.
        #
        # In unserer aktuellen Environment ist normalerweise
        # genau ein Agent entscheidungsbereit.
        # -------------------------------------------------

        action_dict = {}

        for agent in observations.keys():

            action = env.action_spaces[
                agent
            ].sample()

            action_dict[agent] = action

            action_counts[action] += 1

            vehicle = env.agent_to_vehicle.get(
                agent,
                "unbekannt"
            )

            print(
                f"Agent: {agent} | "
                f"Fahrzeug: {vehicle} | "
                f"Aktion: {action} "
                f"(0=P1, 1=P2, 2=P3)"
            )

        # -------------------------------------------------
        # Einen Multi-Agent-Schritt ausführen
        # -------------------------------------------------

        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos
        ) = env.step(action_dict)

        # -------------------------------------------------
        # Rewards aller Fahrzeuge, die seit dem letzten
        # Schritt fertig geworden sind, aufsummieren.
        # -------------------------------------------------

        if rewards:

            for agent, reward in rewards.items():

                total_episode_reward += reward

                total_completed_agents += 1

                print(
                    f"Reward erhalten | "
                    f"Agent: {agent} | "
                    f"Reward: {reward:.4f}"
                )

        step_count += 1

        # -------------------------------------------------
        # Episode beendet?
        # -------------------------------------------------

        episode_finished = (
            terminateds.get(
                "__all__",
                False
            )
            or truncateds.get(
                "__all__",
                False
            )
        )

    # -----------------------------------------------------
    # Ergebnisse dieser Episode
    # -----------------------------------------------------

    episode_rewards.append(
        total_episode_reward
    )

    episode_agent_rewards.append(
        total_completed_agents
    )

    print(
        f"\nEpisode beendet | "
        f"Schritte: {step_count}"
    )

    print(
        f"Abgeschlossene Fahrzeuge: "
        f"{total_completed_agents}"
    )

    print(
        f"Gesamtreward: "
        f"{total_episode_reward:.4f}"
    )

    if total_completed_agents > 0:

        average_agent_reward = (
            total_episode_reward
            / total_completed_agents
        )

        print(
            f"Durchschnittlicher Reward "
            f"pro Fahrzeug: "
            f"{average_agent_reward:.4f}"
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

    total_completed = sum(
        episode_agent_rewards
    )

    total_reward = sum(
        episode_rewards
    )

    print("\n======================================")
    print("Random-Agent Ergebnis")
    print("======================================")

    print(
        f"Anzahl Episoden: "
        f"{len(episode_rewards)}"
    )

    print(
        f"Durchschnittlicher Gesamtreward: "
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

    print(
        f"Abgeschlossene Fahrzeuge gesamt: "
        f"{total_completed}"
    )

    if total_completed > 0:

        print(
            f"Durchschnittlicher Reward "
            f"pro Fahrzeug: "
            f"{total_reward / total_completed:.4f}"
        )


# =========================================================
# Environment schließen
# =========================================================

env.close()

print("\nRandom-Agent Test abgeschlossen.")
