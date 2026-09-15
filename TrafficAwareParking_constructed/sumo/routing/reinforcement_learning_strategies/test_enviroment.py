from parking_env import ParkingEnv


# Environment erzeugen
env = ParkingEnv()


# ---------------------------------------------------------
# Episode starten
# ---------------------------------------------------------

observation, info = env.reset()

print("Environment gestartet.")

print("Initial State:")
print(observation)

print("Info:")
print(info)


# ---------------------------------------------------------
# Testaktion ausführen
# ---------------------------------------------------------

# 0 = P1
# 1 = P2
# 2 = P3

action = 1

new_observation, reward, terminated, truncated, info = env.step(action)


# ---------------------------------------------------------
# Ergebnisse ausgeben
# ---------------------------------------------------------

print("\nAktion:")
print(action)

print("\nReward:")
print(reward)

print("\nNeuer State:")
print(new_observation)

print("\nInfo:")
print(info)

print("\nSimulation beendet.")
env.close()