import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci


class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    # =====================================================
    # SUMO / Szenario
    # =====================================================

    SUMO_CONFIG = (
        r"C:\Users\User\Desktop\TrafficAwareParking_constructed_4parking"
        r"\TrafficAwareParking_constructed\sumo\config\constructed.sumocfg"
    )

    PARKING_AREAS = ["P1", "P2", "P3"]

    PARKING_LANES = {
        "P1": "HochfahrtLinks_0",
        "P2": "LinksLinks_0",
        "P3": "HochfahrtRechts_0"
    }

    PARKING_CAPACITY = {
        "P1": 40,
        "P2": 40,
        "P3": 40
    }

    WALKING_DISTANCE = {
        "P1": 300,
        "P2": 250,
        "P3": 50
    }

    # =====================================================
    # Initialisierung
    # =====================================================

    def __init__(self):

        super().__init__()

        # -----------------------------------------------------
        # Reward-Gewichte
        # -----------------------------------------------------

        self.W_DRIVE = 1.0
        self.W_WAIT = 1.0
        self.W_WALK = 0.1
        self.W_FAILURE = 10.0

        # -------------------------------------------------
        # Action Space
        #
        # 0 -> P1
        # 1 -> P2
        # 2 -> P3
        # -------------------------------------------------

        self.action_space = spaces.Discrete(3)

        # -------------------------------------------------
        # Observation Space
        #
        # 3 Parkplätze * 4 Werte
        #
        # Belegung
        # Entfernung
        # Fahrzeit
        # Fußweg
        # -------------------------------------------------

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        # -------------------------------------------------
        # Fahrzeugverwaltung
        # -------------------------------------------------

        # Bereits bekannte Fahrzeuge
        self.known_vehicles = set()

        # Fahrzeuge, die eine Entscheidung benötigen
        self.decision_queue = []

        # Fahrzeuge, für die bereits eine Entscheidung
        # getroffen wurde, deren Ergebnis aber noch aussteht
        self.pending_vehicles = {}

        # Fahrzeug, das gerade vom Agenten behandelt wird
        self.current_vehicle = None

        # Aktueller State
        self.current_state = None

    # =====================================================
    # Neue Episode
    # =====================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Alte SUMO-Verbindung schließen
        if traci.isLoaded():
            traci.close()

        # SUMO starten
        traci.start([
            "sumo",
            "-c", self.SUMO_CONFIG
        ])

        # Variablen zurücksetzen
        self.known_vehicles = set()
        self.decision_queue = []
        self.pending_vehicles = {}
        self.current_vehicle = None
        self.current_state = None

        # Bis zum ersten normalen Fahrzeug laufen
        observation, _, info = self._advance_until_decision()

        return observation, info

    # =====================================================
    # RL-Schritt
    # =====================================================

    def step(self, action):

        # -------------------------------------------------
        # 1. Aktuelles Fahrzeug holen
        # -------------------------------------------------

        if self.current_vehicle is None:
            raise RuntimeError(
                "Kein Fahrzeug für eine Aktion vorhanden."
            )

        vehicle = self.current_vehicle

        # -------------------------------------------------
        # 2. Aktion -> Parkplatz
        # -------------------------------------------------

        parking_area = self.PARKING_AREAS[int(action)]

        # -------------------------------------------------
        # 3. Aktuelle Fahrzeugposition
        # -------------------------------------------------

        current_edge = traci.vehicle.getRoadID(
            vehicle
        )

        # -------------------------------------------------
        # 4. Parking Lane -> Parking Edge
        # -------------------------------------------------

        parking_lane = self.PARKING_LANES[
            parking_area
        ]

        parking_edge = traci.lane.getEdgeID(
            parking_lane
        )

        # -------------------------------------------------
        # 5. Route berechnen
        # -------------------------------------------------

        route = traci.simulation.findRoute(
            current_edge,
            parking_edge,
            vType="car"
        )

        # -------------------------------------------------
        # 6. Route nicht möglich
        # -------------------------------------------------

        if not route.edges:

            reward = -100.0

            self.current_vehicle = None

            observation, info = self._advance_until_decision()

            return (
                observation,
                reward,
                info["simulation_finished"],
                False,
                info
            )

        # -------------------------------------------------
        # 7. Route setzen
        # -------------------------------------------------

        traci.vehicle.setRoute(
            vehicle,
            list(route.edges)
        )

        # -------------------------------------------------
        # 8. Parking Stop setzen
        # -------------------------------------------------

        traci.vehicle.setParkingAreaStop(
            vehicle,
            parking_area,
            duration=1200
        )

        # -------------------------------------------------
        # 9. Fahrzeug als "pending" speichern
        # -------------------------------------------------

        self.pending_vehicles[vehicle] = {

            "action": int(action),

            "parking_area": parking_area,

            "waiting_time": 0,

            "start_time":
                traci.simulation.getTime(),

            "walking_distance":
                self.WALKING_DISTANCE[parking_area],

            "parking_failure": 0
        }

        # -------------------------------------------------
        # 10. Aktuelles Fahrzeug ist entschieden
        # -------------------------------------------------

        self.current_vehicle = None

        # -------------------------------------------------
        # 11. Simulation weiterlaufen lassen,
        #     bis das nächste Entscheidungsereignis
        #     vorhanden ist.
        # -------------------------------------------------

        observation, reward, info = (
            self._advance_until_decision()
        )

        print(
            f"STEP-REWARD: {reward:.4f}"
            )

        terminated = info["simulation_finished"]
        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # =====================================================
    # Simulation weiterführen
    # =====================================================

    def _advance_until_decision(self):

        accumulated_reward = 0.0

        while True:

            # -------------------------------------------------
            # Einen SUMO-Schritt durchführen
            # -------------------------------------------------

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # -------------------------------------------------
            # Aktuelle Fahrzeuge
            # -------------------------------------------------

            current_vehicles = set(
                traci.vehicle.getIDList()
            )

            # -------------------------------------------------
            # Neue Fahrzeuge erkennen
            # -------------------------------------------------

            new_vehicles = (
                current_vehicles
                - self.known_vehicles
            )

            # Störverkehr entfernen
            new_vehicles = [
                vehicle
                for vehicle in new_vehicles
                if not vehicle.startswith("Störverkehr")
            ]

            # Neue Fahrzeuge in Entscheidungswarteschlange
            for vehicle in new_vehicles:

                if vehicle not in self.decision_queue:

                    self.decision_queue.append(
                        vehicle
                    )

            # Bekannte Fahrzeuge aktualisieren
            self.known_vehicles.update(
                current_vehicles
            )

            # -------------------------------------------------
            # Prüfen, ob pending Fahrzeuge fertig sind
            # -------------------------------------------------

            completed_vehicles = []

            for vehicle in list(
                self.pending_vehicles.keys()
                ):

            # Fahrzeug ist noch in der Simulation
                if vehicle in current_vehicles:

                    speed = traci.vehicle.getSpeed(vehicle)

                    if not traci.vehicle.isStoppedParking(vehicle):

                        if speed < 0.1:
                            self.pending_vehicles[vehicle]["waiting_time"] += 1.0



                    parking_area = (
                        self.pending_vehicles[vehicle]["parking_area"]
                        )

                    parking_lane = self.PARKING_LANES[parking_area]
                    parking_edge = traci.lane.getEdgeID(parking_lane)

                    current_edge = traci.vehicle.getRoadID(vehicle)

                    # Fahrzeug ist an der Parkplatz-Edge angekommen
                    if current_edge == parking_edge:

                        occupied = traci.parkingarea.getVehicleCount(
                            parking_area
                            )

                        capacity = self.PARKING_CAPACITY[
                            parking_area
                            ]

                        # Parkplatz ist voll
                        if occupied >= capacity:

                            self.pending_vehicles[vehicle][
                                "parking_failure"
                                ] += 1

                    # Aktuelle Parkinformation
                    if traci.vehicle.isStoppedParking(vehicle):
                        completed_vehicles.append(vehicle)

            # -------------------------------------------------
            # Vorläufige Rewards
            #
            # Noch kein endgültiger Reward!
            # -------------------------------------------------

            for vehicle in completed_vehicles:

                data = self.pending_vehicles[vehicle]

                current_time = traci.simulation.getTime()

                travel_time = (
                    current_time - data["start_time"]
                )

                waiting_time = data["waiting_time"]

                walking_distance = data["walking_distance"]

                parking_failure = data["parking_failure"]

                print(
                    f"Fahrzeug {vehicle} hat geparkt | "
                    f"Fahrzeit: {travel_time:.2f} s | "
                    f"Wartezeit: {waiting_time:.2f} s | "
                    f"Weg Distanz: {walking_distance:.2f} s | "
                    f"Parkfehler: {parking_failure:.2f} s | "
                    f"Parkplatz: {data['parking_area']}"
                )

                # Nur Platzhalter
                reward = self._calculate_reward(
                    vehicle,
                    data
                )

                print(
                    f"Fahrzeug-Reward: {reward:.4f}"
                    )

                accumulated_reward += reward

                del self.pending_vehicles[
                    vehicle
                ]

            # -------------------------------------------------
            # Gibt es ein Fahrzeug, das jetzt
            # entschieden werden muss?
            # -------------------------------------------------

            if self.decision_queue:

                self.current_vehicle = (
                    self.decision_queue.pop(0)
                )

                observation = self._build_observation(
                    self.current_vehicle
                )

                info = {
                    "vehicle": self.current_vehicle,
                    "simulation_time": current_time,
                    "simulation_finished": False
                }

                self.current_state = observation

                return (
                    observation,
                    accumulated_reward,
                    info
                )

            # -------------------------------------------------
            # Simulation beendet?
            # -------------------------------------------------

            if traci.simulation.getMinExpectedNumber() <= 0:

                observation = np.zeros(
                    self.observation_space.shape,
                    dtype=np.float32
                )

                info = {
                    "vehicle": None,
                    "simulation_time": current_time,
                    "simulation_finished": True
                }

                return (
                    observation,
                    accumulated_reward,
                    info
                )

    # =====================================================
    # State erzeugen
    # =====================================================

    def _build_observation(self, vehicle):

        current_edge = traci.vehicle.getRoadID(
            vehicle
        )

        observation = []

        for parking_area in self.PARKING_AREAS:

            # -------------------------------------------------
            # Parking Area
            # -------------------------------------------------

            occupied = (
                traci.parkingarea.getVehicleCount(
                    parking_area
                )
            )

            capacity = self.PARKING_CAPACITY[
                parking_area
            ]

            occupancy_normalized = (
                occupied / capacity
            )

            # -------------------------------------------------
            # Parking Lane -> Edge
            # -------------------------------------------------

            parking_lane = self.PARKING_LANES[
                parking_area
            ]

            parking_edge = traci.lane.getEdgeID(
                parking_lane
            )

            # -------------------------------------------------
            # Route
            # -------------------------------------------------

            route = traci.simulation.findRoute(
                current_edge,
                parking_edge,
                vType="car"
            )

            # -------------------------------------------------
            # Distanz
            # -------------------------------------------------

            distance_normalized = min(
                route.length / 1000.0,
                1.0
            )

            # -------------------------------------------------
            # Fahrzeit
            # -------------------------------------------------

            travel_time_normalized = min(
                route.travelTime / 300.0,
                1.0
            )

            # -------------------------------------------------
            # Fußweg
            # -------------------------------------------------

            walking_normalized = min(
                self.WALKING_DISTANCE[
                    parking_area
                ] / 500.0,
                1.0
            )

            observation.extend([
                occupancy_normalized,
                distance_normalized,
                travel_time_normalized,
                walking_normalized
            ])

        return np.array(
            observation,
            dtype=np.float32
        )

        # =====================================================
        # Umgebung schließen
        # =====================================================

    def close(self):

        if traci.isLoaded():
            traci.close()