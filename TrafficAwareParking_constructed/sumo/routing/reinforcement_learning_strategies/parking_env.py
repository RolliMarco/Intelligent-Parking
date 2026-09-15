import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci


class ParkingEnv(gym.Env):

    # -----------------------------------------------------
    # Grundeinstellungen
    # -----------------------------------------------------

    metadata = {"render_modes": ["human"]}

    SUMO_CONFIG = (
        r"C:\Users\User\Desktop\TrafficAwareParking_constructed_4parking"
        r"\TrafficAwareParking_constructed\sumo\config\constructed.sumocfg"
    )

    PARKING_AREAS = ["P1", "P2", "P3"]

    PARKING_EDGES = {
        "P1": "HochfahrtLinks",
        "P2": "LinksLinks",
        "P3": "HochfahrtRechts"
    }

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

    # -----------------------------------------------------
    # Initialisierung
    # -----------------------------------------------------

    def __init__(self):

        super().__init__()

        # Drei mögliche Aktionen:
        #
        # 0 -> P1
        # 1 -> P2
        # 2 -> P3

        self.action_space = spaces.Discrete(3)

        # State:
        #
        # Für jeden Parkplatz:
        #   Belegung
        #   Routendistanz
        #   Fahrzeit
        #   Fußweg
        #
        # 3 Parkplätze * 4 Werte = 12 Werte

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self.known_vehicles = set()
        self.current_vehicle = None
        self.current_state = None

    # -----------------------------------------------------
    # Neue Episode starten
    # -----------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Alte TraCI-Verbindung schließen
        if traci.isLoaded():
            traci.close()

        # SUMO neu starten
        traci.start([
            "sumo-gui",
            "-c", self.SUMO_CONFIG
        ])

        self.known_vehicles = set()
        self.current_vehicle = None
        self.current_state = None

        # Zum ersten neuen Fahrzeug laufen
        return self._get_next_vehicle_state()

    # -----------------------------------------------------
    # Einen RL-Schritt ausführen
    # -----------------------------------------------------

    def step(self, action):

        # Aus der Aktion den Parkplatz bestimmen
        parking_area = self.PARKING_AREAS[int(action)]

        vehicle = self.current_vehicle

        # Aktuelle Edge des Fahrzeugs
        current_edge = traci.vehicle.getRoadID(vehicle)

        parking_lane = self.PARKING_LANES[parking_area]
        parking_edge = traci.lane.getEdgeID(parking_lane)

        route = traci.simulation.findRoute(
            current_edge,
            parking_edge,
            vType="car"
        )


        # Sicherheitsprüfung
        if not route.edges:

            reward = -100.0

            observation, info = self._get_next_vehicle_state()

            return (
                observation,
                reward,
                False,
                False,
                info
            )

        # Route setzen
        traci.vehicle.setRoute(
            vehicle,
            list(route.edges)
        )

        # Parking Stop setzen
        traci.vehicle.setParkingAreaStop(
            vehicle,
            parking_area,
            duration=1200
        )

        # -------------------------------------------------
        # Vorläufiger Reward
        # -------------------------------------------------
        #
        # WICHTIG:
        # Dies ist zunächst nur ein funktionierender
        # Prototyp. Die endgültige Reward-Funktion bauen
        # wir erst nach dem Environment-Test.
        # -------------------------------------------------

        travel_time = route.travelTime
        walking_distance = self.WALKING_DISTANCE[parking_area]

        reward = (
            -(travel_time / 60.0)
            -(walking_distance / 100.0)
        )

        # Nächstes neues Fahrzeug suchen
        observation, info = self._get_next_vehicle_state()

        terminated = info["simulation_finished"]

        truncated = False

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    # -----------------------------------------------------
    # Zum nächsten neuen Fahrzeug laufen
    # -----------------------------------------------------

    def _get_next_vehicle_state(self):

        while True:

            # Aktuelle Fahrzeuge
            current_vehicles = set(
                traci.vehicle.getIDList()
            )

            # Noch nicht bekannte Fahrzeuge
            new_vehicles = (
                current_vehicles - self.known_vehicles
            )

            # Störverkehr entfernen
            new_vehicles = {
                vehicle
                for vehicle in new_vehicles
                if not vehicle.startswith("Störverkehr")
            }

            # Bekannte Fahrzeuge aktualisieren
            self.known_vehicles.update(
                current_vehicles
            )

            # Gibt es ein neues normales Fahrzeug?
            if new_vehicles:

                self.current_vehicle = sorted(
                    new_vehicles
                )[0]

                observation = self._build_observation(
                    self.current_vehicle
                )

                info = {
                    "vehicle": self.current_vehicle,
                    "simulation_time": traci.simulation.getTime(),
                    "simulation_finished": False
                }

                self.current_state = observation

                return observation, info

            # Kein neues Fahrzeug:
            # Simulation einen Schritt weiterführen
            traci.simulationStep()

            # Prüfen, ob Simulation beendet ist
            if traci.simulation.getMinExpectedNumber() <= 0:

                observation = np.zeros(
                    self.observation_space.shape,
                    dtype=np.float32
                )

                info = {
                    "vehicle": None,
                    "simulation_time": traci.simulation.getTime(),
                    "simulation_finished": True
                }

                return observation, info

    # -----------------------------------------------------
    # State erzeugen
    # -----------------------------------------------------

    def _build_observation(self, vehicle):

        current_edge = traci.vehicle.getRoadID(
            vehicle
        )

        observation = []

        for parking_area in self.PARKING_AREAS:

            parking_lane = self.PARKING_LANES[parking_area]
            parking_edge = traci.lane.getEdgeID(parking_lane)

            # Belegung
            occupied = traci.parkingarea.getVehicleCount(
                parking_area
            )

            capacity = self.PARKING_CAPACITY[
                parking_area
            ]

            occupancy_normalized = (
                occupied / capacity
            )

            # Route berechnen
            route = traci.simulation.findRoute(
                current_edge,
                parking_edge,
                vType="car"
            )

            # Normalisierte Entfernung
            distance_normalized = min(
                route.length / 1000.0,
                1.0
            )

            # Normalisierte Fahrzeit
            travel_time_normalized = min(
                route.travelTime / 300.0,
                1.0
            )

            # Normalisierte Gehstrecke
            walking_normalized = min(
                self.WALKING_DISTANCE[parking_area]
                / 500.0,
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

    # -----------------------------------------------------
    # SUMO schließen
    # -----------------------------------------------------

    def close(self):

        if traci.isLoaded():
            traci.close()