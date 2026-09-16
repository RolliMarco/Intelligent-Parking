from parking_env import ParkingEnv
from stable_baselines3.common.env_checker import check_env


env = ParkingEnv()

print("Starte Environment-Prüfung...")

check_env(env)

print("Environment ist gültig.")

env.close()