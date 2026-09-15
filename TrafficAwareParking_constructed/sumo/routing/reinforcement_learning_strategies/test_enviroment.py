from parking_env import ParkingEnv

env = ParkingEnv()

observation, info = env.reset()

print("Environment gestartet.")
print("State:")
print(observation)

print("Info:")
print(info)

env.close()