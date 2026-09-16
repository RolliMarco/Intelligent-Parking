from parking_env import ParkingEnv


env = ParkingEnv()

for episode in range(2):

    print(f"\n========== EPISODE {episode + 1} ==========")

    observation, info = env.reset()

    terminated = False
    truncated = False

    step = 0

    while not (terminated or truncated):

        # Testaktion
        action = 1

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        step += 1

        if step % 10 == 0:
            print(
                f"Step: {step} | "
                f"Zeit: {info['simulation_time']:.1f} | "
                f"Reward: {reward:.3f}"
            )

env.close()

print("\nBeide Episoden beendet.")