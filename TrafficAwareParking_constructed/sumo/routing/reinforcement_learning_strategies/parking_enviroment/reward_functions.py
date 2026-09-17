import numpy as np

def calculate_reward(vehicle, data, current_time, w_drive, w_wait, w_walk, w_failure):

    # -------------------------------------------------
    # 1. Tatsächliche Fahrzeit
    # -------------------------------------------------

    travel_time = (
        current_time - data["start_time"]
    )

    # -------------------------------------------------
    # 2. Wartezeit
    # -------------------------------------------------

    waiting_time = data["waiting_time"]

    # -------------------------------------------------
    # 3. Fußweg
    # -------------------------------------------------

    walking_distance = data["walking_distance"]

    # -------------------------------------------------
    # 4. Parking Failure
    # -------------------------------------------------

    parking_failure = data["parking_failure"]

    # -------------------------------------------------
    # 5. Normalisierung
    #
    # Empirische Wertebereiche:
    #
    # Fahrzeit:     76 - 250 s
    # Wartezeit:    0 - 150 s
    # Fußweg:       50 - 300 m
    # Parkfehler:   0 -170
    # -------------------------------------------------

    drive_norm = (
        (travel_time - 76.0)
        / (250.0 - 76.0)
    )

    wait_norm = (
        waiting_time/150
    )

    walk_norm = (
        (walking_distance - 50.0)
        / (300.0 - 50.0)
    )

    failure_norm = (
        parking_failure/170
    )


    # -------------------------------------------------
    # 6. Werte auf [0, 1] begrenzen
    # -------------------------------------------------

    drive_norm = np.clip(
        drive_norm,
        0.0,
        1.0
    )

    wait_norm = np.clip(
        wait_norm,
        0.0,
        1.0
    )

    walk_norm = np.clip(
        walk_norm,
        0.0,
        1.0
    )

    failure_norm = np.clip(
        failure_norm,
        0.0,
        1.0
    )

    # -------------------------------------------------
    # 7. Reward berechnen
    # -------------------------------------------------

    reward = (
        -w_drive * drive_norm
        -w_wait * wait_norm
        -w_walk * walk_norm
        -w_failure * failure_norm
    )

    # -------------------------------------------------
    # 8. Zum Test ausgeben
    # -------------------------------------------------

    print(
        f"Reward für {vehicle} | "
        f"Fahrzeit: {travel_time:.2f} s | "
        f"Wartezeit: {waiting_time:.2f} s | "
        f"Fußweg: {walking_distance:.2f} m | "
        f"Parking Failure: {parking_failure} | "
        f"Reward: {reward:.4f}"
    )

    return float(reward)