import numpy as np
import traci

from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from reward_functions import calculate_reward


class ParkingMultiAgentEnv2(MultiAgentEnv):
    """
    Direkte RLlib-Multi-Agent-Environment für das
    Traffic-Aware-Parking-Szenario.

    Wichtige Architektur:
    - Jedes normale SUMO-Fahrzeug erhält dynamisch einen Agenten.
    - Alle Fahrzeug-Agenten verwenden später dieselbe Policy.
    - Es ist immer nur der Agent entscheidungsbereit, dessen
      Fahrzeug gerade in der decision_queue steht.
    - Pending-Fahrzeuge fahren währenddessen weiter.
    - Ein später entstehender Reward wird direkt dem
      verursachenden Agenten zugeordnet.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "traffic_aware_parking_v2",
    }

    # =====================================================
    # SUMO / Szenario
    # =====================================================

    SUMO_CONFIG = (
        r"C:\Users\User\Desktop\TrafficAwareParking_constructed_4parking"
        r"\TrafficAwareParking_constructed\sumo\config\constructed.sumocfg"
    )

    SIMULATION_END = 500.0

    # =====================================================
    # Parkplätze
    # =====================================================

    PARKING_AREAS = ["P1", "P2", "P3"]

    PARKING_LANES = {
        "P1": "HochfahrtLinks_0",
        "P2": "LinksLinks_0",
        "P3": "HochfahrtRechts_0",
    }

    PARKING_CAPACITY = {
        "P1": 40,
        "P2": 40,
        "P3": 40,
    }

    WALKING_DISTANCE = {
        "P1": 300,
        "P2": 250,
        "P3": 50,
    }

    # =====================================================
    # Initialisierung
    # =====================================================

    def __init__(
        self,
        max_agent_slots=2000,
        render_mode=None,
    ):

        super().__init__()

        self.render_mode = render_mode

        # -------------------------------------------------
        # Alle möglichen Agenten.
        # Diese Liste bleibt während der Laufzeit konstant.
        # -------------------------------------------------

        self.possible_agents = [
            f"vehicle_agent_{i}"
            for i in range(max_agent_slots)
        ]

        self._agent_ids = set(
            self.possible_agents
        )

        # -------------------------------------------------
        # Für jeden möglichen Agenten sind die Spaces
        # von Anfang an bekannt.
        # -------------------------------------------------

        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(12,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(3)
            for agent in self.possible_agents
        }

        # -------------------------------------------------
        # Reward-Gewichte
        # -------------------------------------------------

        self.W_DRIVE = 1.0
        self.W_WAIT = 1.0
        self.W_WALK = 0.1
        self.W_FAILURE = 10.0

        # -------------------------------------------------
        # Laufzeitvariablen
        # -------------------------------------------------

        self.agents = []

        self.known_vehicles = set()

        # Agent <-> SUMO-Fahrzeug
        self.agent_to_vehicle = {}
        self.vehicle_to_agent = {}

        # Fahrzeuge, die noch keine Aktion erhalten haben
        self.decision_queue = []

        # Agenten, deren Entscheidung bereits getroffen wurde
        # und deren Ergebnis noch aussteht.
        #
        # Struktur:
        # {
        #     "vehicle_agent_0": {
        #         "vehicle_id": "...",
        #         "action": 1,
        #         "parking_area": "P2",
        #         "start_time": ...,
        #         "walking_distance": 250,
        #         "waiting_time": 0.0,
        #         "parking_failure": 0
        #     }
        # }
        self.pending_vehicles = {}

        # Aktuell entscheidungsbereite Agenten.
        # In dieser ersten Version normalerweise maximal einer.
        self.ready_agents = []

        # =================================================
        # Zustände / RLlib-Ausgaben
        # =================================================

        self.observations = {}
        self.infos = {}

        self.current_rewards = {}

        self.terminations = {}
        self.truncations = {}

        self.episode_terminated = False

        # Nächster freier Agenten-Slot
        self.next_agent_index = 0

    # =====================================================
    # Spaces
    # =====================================================

    def get_observation_space(self, agent_id):

        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):

        return self.action_spaces[agent_id]

    # =====================================================
    # Reset
    # =====================================================

    def reset(self, *, seed=None, options=None):

        # -------------------------------------------------
        # Alte SUMO-Verbindung schließen
        # -------------------------------------------------

        if traci.isLoaded():
            traci.close()

        # -------------------------------------------------
        # SUMO neu starten
        # -------------------------------------------------

        traci.start([
            "sumo",
            "-c",
            self.SUMO_CONFIG,
        ])

        # -------------------------------------------------
        # Interne Daten zurücksetzen
        # -------------------------------------------------

        self.agents = []

        self.known_vehicles = set()

        self.agent_to_vehicle = {}
        self.vehicle_to_agent = {}

        self.decision_queue = []
        self.pending_vehicles = {}
        self.ready_agents = []

        self.observations = {}
        self.infos = {}

        self.current_rewards = {}

        self.terminations = {}
        self.truncations = {}

        self.episode_terminated = False

        self.next_agent_index = 0

        # -------------------------------------------------
        # Bis zum ersten entscheidungsbereiten Fahrzeug
        # laufen
        # -------------------------------------------------

        observations, _, _, _, infos = (
            self._advance_until_decision()
        )

        return observations, infos

    # =====================================================
    # RLlib Step
    # =====================================================

    def step(self, action_dict):
        print("\n----- ENV STEP -----")
        print(
            f"Simulation: "
            f"{traci.simulation.getTime():.1f}s"
        )

        print(
            f"Aktionen von RLlib: {action_dict}"
        )
        # -------------------------------------------------
        # Falls die Episode bereits beendet wurde
        # -------------------------------------------------

        if self.episode_terminated:

            return (
                {},
                {},
                {"__all__": True},
                {"__all__": False},
                {},
            )

        # -------------------------------------------------
        # In unserer ersten Version ist genau ein Agent
        # entscheidungsbereit.
        # -------------------------------------------------

        if not self.ready_agents:

            raise RuntimeError(
                "step() wurde aufgerufen, obwohl kein "
                "Agent entscheidungsbereit ist."
            )

        current_agent = self.ready_agents.pop(0)

        # -------------------------------------------------
        # RLlib muss für diesen Agenten eine Aktion liefern
        # -------------------------------------------------

        if current_agent not in action_dict:

            raise ValueError(
                f"Keine Aktion für "
                f"{current_agent} erhalten."
            )

        action = int(
            action_dict[current_agent]
        )

        vehicle = self.agent_to_vehicle[
            current_agent
        ]
        print(vehicle, action)
        # -------------------------------------------------
        # Aktion auf Gültigkeit prüfen
        # -------------------------------------------------

        if action not in range(
            len(self.PARKING_AREAS)
        ):

            raise ValueError(
                f"Ungültige Aktion {action} "
                f"für {current_agent}."
            )

        parking_area = self.PARKING_AREAS[
            action
        ]

        # -------------------------------------------------
        # Aktuelle Fahrzeugposition
        # -------------------------------------------------

        if vehicle not in traci.vehicle.getIDList():

            raise RuntimeError(
                f"Fahrzeug {vehicle} ist nicht mehr "
                f"in SUMO vorhanden."
            )

        current_edge = traci.vehicle.getRoadID(
            vehicle
        )

        # -------------------------------------------------
        # Parking Lane -> Parking Edge
        # -------------------------------------------------

        parking_lane = self.PARKING_LANES[
            parking_area
        ]

        parking_edge = traci.lane.getEdgeID(
            parking_lane
        )

        # -------------------------------------------------
        # Route berechnen
        # -------------------------------------------------

        route = traci.simulation.findRoute(
            current_edge,
            parking_edge,
            vType="car",
        )

        # =================================================
        # Keine Route
        # =================================================

        if not route.edges:

            self.current_rewards[
                current_agent
            ] = -100.0

            self.terminations[
                current_agent
            ] = True

            self.truncations[
                current_agent
            ] = False

            self.infos[
                current_agent
            ]["reason"] = "no_route"

            # Agent ist fertig.
            self._remove_agent(
                current_agent
            )

            # Simulation für den nächsten Agenten fortsetzen
            return self._advance_until_decision(
                include_rewards=True
            )

        # =================================================
        # Route setzen
        # =================================================

        traci.vehicle.setRoute(
            vehicle,
            list(route.edges),
        )

        # =================================================
        # Parking Stop setzen
        # =================================================

        traci.vehicle.setParkingAreaStop(
            vehicle,
            parking_area,
            duration=1200,
        )

        # =================================================
        # Pending-Daten speichern
        # =================================================

        self.pending_vehicles[
            current_agent
        ] = {

            "vehicle_id": vehicle,

            "action": action,

            "parking_area": parking_area,

            "start_time":
                traci.simulation.getTime(),

            "walking_distance":
                self.WALKING_DISTANCE[
                    parking_area
                ],

            "waiting_time": 0.0,

            "parking_failure": 0,
        }

        # -------------------------------------------------
        # Informationen zur Entscheidung speichern
        # -------------------------------------------------

        self.infos[
            current_agent
        ] = {

            "vehicle_id": vehicle,

            "action": action,

            "parking_area": parking_area,

            "decision_time":
                traci.simulation.getTime(),
        }

        # -------------------------------------------------
        # Agent bleibt aktiv, weil sein Reward erst später
        # bekannt wird.
        # -------------------------------------------------

        # Simulation bis zum nächsten Ereignis fortsetzen.
        return self._advance_until_decision(
            include_rewards=True
        )

    # =====================================================
    # Simulation fortsetzen
    # =====================================================

    def _advance_until_decision(
        self,
        include_rewards=False,
    ):

        rewards = {}

        while True:

            # -------------------------------------------------
            # Falls neue Entscheidungsagenten vorhanden sind
            # -------------------------------------------------

            if self.decision_queue:

                agent = self.decision_queue.pop(0)

                self.ready_agents.append(
                    agent
                )

                observation = (
                    self.observations[
                        agent
                    ]
                )

                info = self.infos[
                    agent
                ]

                self.agents = [
                    a
                    for a in self.agents
                    if not (
                        self.terminations.get(a, False)
                        or self.truncations.get(a, False)
                    )
                ]

                return (
                    {agent: observation},
                    rewards,
                    {
                        "__all__": False
                    },
                    {
                        "__all__": False
                    },
                    {
                        agent: info
                    },
                )

            # -------------------------------------------------
            # Wenn keine Entscheidung bereit ist:
            # einen SUMO-Schritt durchführen
            # -------------------------------------------------

            traci.simulationStep()

            current_time = (
                traci.simulation.getTime()
            )

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

            for vehicle in sorted(new_vehicles):

                if vehicle.startswith(
                    "Störverkehr"
                ):
                    continue

                self._create_agent(
                    vehicle,
                    current_time
                )

            self.known_vehicles.update(
                current_vehicles
            )

            # -------------------------------------------------
            # Pending-Fahrzeuge aktualisieren
            # -------------------------------------------------

            completed_agents = []

            for agent in list(
                self.pending_vehicles.keys()
            ):

                data = self.pending_vehicles[
                    agent
                ]

                vehicle = data[
                    "vehicle_id"
                ]

                # -------------------------------------------------
                # Fahrzeug noch vorhanden?
                # -------------------------------------------------

                if vehicle in current_vehicles:

                    # -------------------------------------------------
                    # Parkzustand
                    # -------------------------------------------------

                    is_parking = (
                        traci.vehicle.isStoppedParking(
                            vehicle
                        )
                    )

                    # -------------------------------------------------
                    # Noch nicht geparkt:
                    # Wartezeit + Parking Failure
                    # -------------------------------------------------

                    if not is_parking:

                        speed = traci.vehicle.getSpeed(
                            vehicle
                        )

                        if speed < 0.1:

                            data[
                                "waiting_time"
                            ] += 1.0

                        parking_area = data[
                            "parking_area"
                        ]

                        parking_lane = (
                            self.PARKING_LANES[
                                parking_area
                            ]
                        )

                        parking_edge = (
                            traci.lane.getEdgeID(
                                parking_lane
                            )
                        )

                        current_edge = (
                            traci.vehicle.getRoadID(
                                vehicle
                            )
                        )

                        if (
                            current_edge
                            == parking_edge
                        ):

                            occupied = (
                                traci.parkingarea.getVehicleCount(
                                    parking_area
                                )
                            )

                            capacity = (
                                self.PARKING_CAPACITY[
                                    parking_area
                                ]
                            )

                            if occupied >= capacity:

                                data[
                                    "parking_failure"
                                ] += 1

                    # -------------------------------------------------
                    # Fahrzeug parkt
                    # -------------------------------------------------

                    if is_parking:

                        completed_agents.append(
                            agent
                        )

                # -------------------------------------------------
                # Fahrzeug ist verschwunden
                # -------------------------------------------------

                else:

                    completed_agents.append(
                        agent
                    )

            # -------------------------------------------------
            # Abgeschlossene Agenten auswerten
            # -------------------------------------------------

            for agent in completed_agents:

                if agent not in self.pending_vehicles:
                    continue

                data = self.pending_vehicles[
                    agent
                ]

                vehicle = data[
                    "vehicle_id"
                ]

                reward = (
                    self._calculate_agent_reward(
                        vehicle,
                        data,
                        current_time
                    )
                )

                print(vehicle, reward)

                rewards[agent] = reward

                self.current_rewards[
                    agent
                ] = reward

                self.terminations[
                    agent
                ] = True

                self.truncations[
                    agent
                ] = False

                self.infos[
                    agent
                ]["finish_time"] = (
                    current_time
                )

                self.infos[
                    agent
                ]["parking_failure"] = (
                    data["parking_failure"]
                )

                self.infos[
                    agent
                ]["waiting_time"] = (
                    data["waiting_time"]
                )

                # Pending entfernen
                del self.pending_vehicles[
                    agent
                ]

                # Agent aus aktiven Agenten entfernen
                # RLlib bekommt die Terminierung über
                # terminateds trotzdem mit.
                if agent in self.agents:
                    self.agents.remove(agent)

            # -------------------------------------------------
            # Falls mindestens ein neues Fahrzeug jetzt
            # entscheidungsbereit ist
            # -------------------------------------------------

            if self.decision_queue:

                agent = self.decision_queue.pop(0)

                self.ready_agents.append(
                    agent
                )

                observations = {
                    agent:
                        self.observations[
                            agent
                        ]
                }

                infos = {
                    agent:
                        self.infos[
                            agent
                        ]
                }

                terminateds = {
                    "__all__": False
                }

                truncateds = {
                    "__all__": False
                }

                if rewards:

                    return (
                        observations,
                        rewards,
                        terminateds,
                        truncateds,
                        infos,
                    )

                if include_rewards:

                    return (
                        observations,
                        {},
                        terminateds,
                        truncateds,
                        infos,
                    )

                return (
                    observations,
                    {},
                    terminateds,
                    truncateds,
                    infos,
                )

            # -------------------------------------------------
            # Prüfen, ob die gesamte Episode beendet ist
            # -------------------------------------------------

            expected = (
                traci.simulation.getMinExpectedNumber()
            )

            no_future_vehicles = (
                expected <= 0
            )

            no_pending = (
                len(self.pending_vehicles) == 0
            )

            no_decisions = (
                len(self.decision_queue) == 0
                and len(self.ready_agents) == 0
            )

            if (
                no_future_vehicles
                and no_pending
                and no_decisions
            ):

                self.episode_terminated = True

                terminateds = {
                    "__all__": True
                }

                truncateds = {
                    "__all__": False
                }

                # Eventuelle zuletzt entstandene Rewards
                # trotzdem zurückgeben.
                return (
                    {},
                    rewards,
                    terminateds,
                    truncateds,
                    {},
                )

    # =====================================================
    # Agent erzeugen
    # =====================================================

    def _create_agent(
        self,
        vehicle,
        current_time,
    ):

        # -------------------------------------------------
        # Freien Agenten-Slot auswählen
        # -------------------------------------------------

        if (
            self.next_agent_index
            >= len(self.possible_agents)
        ):

            raise RuntimeError(
                "Maximale Anzahl möglicher "
                "Agenten erreicht."
            )

        agent = self.possible_agents[
            self.next_agent_index
        ]

        self.next_agent_index += 1

        # -------------------------------------------------
        # Mapping
        # -------------------------------------------------

        self.agent_to_vehicle[
            agent
        ] = vehicle

        self.vehicle_to_agent[
            vehicle
        ] = agent

        # -------------------------------------------------
        # Agent aktivieren
        # -------------------------------------------------

        if agent not in self.agents:

            self.agents.append(
                agent
            )

        self.terminations[
            agent
        ] = False

        self.truncations[
            agent
        ] = False

        self.current_rewards[
            agent
        ] = 0.0

        # -------------------------------------------------
        # Observation für das Fahrzeug erzeugen
        # -------------------------------------------------

        observation = (
            self._build_observation(
                vehicle
            )
        )

        self.observations[
            agent
        ] = observation

        self.infos[
            agent
        ] = {

            "vehicle_id": vehicle,

            "spawn_time": current_time,

            "simulation_finished": False
        }

        # -------------------------------------------------
        # Fahrzeug benötigt eine Entscheidung
        # -------------------------------------------------

        self.decision_queue.append(
            agent
        )

    # =====================================================
    # Reward berechnen
    # =====================================================

    def _calculate_agent_reward(
        self,
        vehicle,
        data,
        current_time,
    ):

        return calculate_reward(

            vehicle,

            data,

            current_time,

            self.W_DRIVE,

            self.W_WAIT,

            self.W_WALK,

            self.W_FAILURE,
        )

    # =====================================================
    # Agent entfernen
    # =====================================================

    def _remove_agent(
        self,
        agent,
    ):

        vehicle = self.agent_to_vehicle.pop(
            agent,
            None
        )

        if vehicle is not None:

            self.vehicle_to_agent.pop(
                vehicle,
                None
            )

        if agent in self.agents:

            self.agents.remove(
                agent
            )

    # =====================================================
    # Observation erzeugen
    #
    # Erste Version:
    # 12 Werte wie in der bisherigen Environment.
    #
    # Pro Parkplatz:
    #   Belegung
    #   Routendistanz
    #   Fahrzeit
    #   Fußweg
    # =====================================================

    def _build_observation(
        self,
        vehicle,
    ):

        if vehicle not in traci.vehicle.getIDList():

            return np.zeros(
                (12,),
                dtype=np.float32
            )

        current_edge = traci.vehicle.getRoadID(
            vehicle
        )

        observation = []

        for parking_area in self.PARKING_AREAS:

            # -------------------------------------------------
            # Belegung
            # -------------------------------------------------

            occupied = (
                traci.parkingarea.getVehicleCount(
                    parking_area
                )
            )

            capacity = (
                self.PARKING_CAPACITY[
                    parking_area
                ]
            )

            occupancy_normalized = (
                occupied / capacity
            )

            # -------------------------------------------------
            # Parking Lane -> Edge
            # -------------------------------------------------

            parking_lane = (
                self.PARKING_LANES[
                    parking_area
                ]
            )

            parking_edge = (
                traci.lane.getEdgeID(
                    parking_lane
                )
            )

            # -------------------------------------------------
            # Route
            # -------------------------------------------------

            route = (
                traci.simulation.findRoute(
                    current_edge,
                    parking_edge,
                    vType="car",
                )
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
            #
            # Bleibt in dieser Version noch im State.
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
                walking_normalized,
            ])

        return np.array(
            observation,
            dtype=np.float32
        )

    # =====================================================
    # Render
    # =====================================================

    def render(self):

        if self.render_mode != "human":
            return

        print(
            f"Aktive Agenten: {self.agents}"
        )

        print(
            f"Pending: "
            f"{list(self.pending_vehicles.keys())}"
        )

    # =====================================================
    # Environment schließen
    # =====================================================

    def close(self):

        if traci.isLoaded():
            traci.close()
