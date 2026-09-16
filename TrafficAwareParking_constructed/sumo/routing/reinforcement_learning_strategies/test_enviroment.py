from parking_env import ParkingEnv


env = ParkingEnv()

# ---------------------------------------------------------
# Environment starten
# ---------------------------------------------------------

observation, info = env.reset()

print("Environment gestartet.")
print("Erstes Fahrzeug:", info["vehicle"])
print("Simulationzeit:", info["simulation_time"])
print("State:")
print(observation)


# ---------------------------------------------------------
# Mehrere RL-Schritte testen
# ---------------------------------------------------------

for i in range(2000):

    # Zum Test immer abwechselnd P1, P2, P3
    action = 1

    print("\n-----------------------------")
    print(f"RL-Schritt {i + 1}")
    print(f"Aktion: {action} "
          f"(0=P1, 1=P2, 2=P3)")

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    print("Fahrzeug:", info["vehicle"])
    print("Simulationzeit:", info["simulation_time"])
    print("Reward:", reward)
    print("State:", observation)

    if terminated or truncated:
        print("Simulation beendet.")
        break


env.close()

print("\nTest abgeschlossen.")