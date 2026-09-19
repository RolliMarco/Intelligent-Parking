import functools

import gymnasium
from gymnasium import spaces
import numpy as np
import traci

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from reward_functions import calculate_reward


class ParkingMultiAgentEnv(MultiAgentEnv):

    metadata = {
        "render_modes": ["human"],
        "name": "traffic_aware_parking_v0"
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

    def __init__(
        self,
        max_agent_slots=2000,
        render_mode=None
    ):

        super().__init__()

        self.render_mode = render_mode

        # -------------------------------------------------
        # Mögliche Agenten
        #
        # Diese Liste bleibt während des gesamten
        # Programms unverändert.
        # -------------------------------------------------

        self.possible_agents = [
            f"vehicle_agent_{i}"
            for i in range(max_agent_slots)
        ]

        # -------------------------------------------------
        # Action Space
        #
        # 0 -> P1
        # 1 -> P2
        # 2 -> P3
        # -------------------------------------------------

        self._action_space = spaces.Discrete(3)

        # -------------------------------------------------
        # Observation Space
        #
        # Erste Version:
        #
        # 3 Parkplätze * 4 Werte = 12
        #
        # Belegung
        # Entfernung
        # Fahrzeit
        # Fußweg
        # -------------------------------------------------

        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self._observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(12,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self._action_spaces = {
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
        # Wird in reset() initialisiert
        # -------------------------------------------------

        self.agents = []

        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.observations = {}

        self.known_vehicles = set()

        # Mapping zwischen PettingZoo-Agent und
        # tatsächlicher SUMO-Fahrzeug-ID
        self.agent_to_vehicle = {}
        self.vehicle_to_agent = {}

        # Fahrzeuge, die noch eine Parkplatzentscheidung
        # benötigen
        self.decision_queue = []

        # Fahrzeuge mit bereits getroffener Entscheidung,
        # deren Ergebnis noch aussteht
        self.pending_vehicles = {}

        # Fahrzeuge, die fertig geparkt haben und jetzt
        # ihren letzten PettingZoo-Schritt mit action=None
        # erhalten müssen
        self.completion_queue = []

        self.agent_selection = None

        # Welcher mögliche Agenten-Slot wird als nächstes
        # verwendet?
        self.next_agent_index = 0

    # =====================================================
    # Observation Space
    # =====================================================

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    # =====================================================
    # Action Space
    # =====================================================

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    # =====================================================
    # Neue Episode
    # =====================================================

    def reset(self, seed=None, options=None):

        # -------------------------------------------------
        # Alte SUMO-Verbindung schließen
        # -------------------------------------------------

        if traci.isLoaded():
            traci.close()

        # -------------------------------------------------
        # SUMO starten
        # -------------------------------------------------

        traci.start([
            "sumo",
            "-c",
            self.SUMO_CONFIG
        ])

        # -------------------------------------------------
        # PettingZoo-Zustand zurücksetzen
        # -------------------------------------------------

        self.agents = []

        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.observations = {}

        self.known_vehicles = set()

        self.agent_to_vehicle = {}
        self.vehicle_to_agent = {}

        self.decision_queue = []
        self.pending_vehicles = {}
        self.completion_queue = []

        self.agent_selection = None

        self.next_agent_index = 0

        # -------------------------------------------------
        # Bis zum ersten neuen Fahrzeug laufen
        # -------------------------------------------------

        self._advance_until_ready()

        # -------------------------------------------------
        # Sicherheitsprüfung
        # -------------------------------------------------

        if not self.agents:

            raise RuntimeError(
                "Keine normalen Fahrzeuge in der Episode gefunden."
            )

    # =====================================================
    # RL-Schritt des aktuell ausgewählten Agenten
    # =====================================================

    def step(self, action):

        # =================================================
        # Fall 1:
        # Agent ist bereits fertig und muss mit None
        # aus der Umgebung entfernt werden.
        # =================================================

        agent = self.agent_selection

        if (
            self.terminations[agent]
            or self.truncations[agent]
        ):

            if action is not None:

                raise ValueError(
                    "Ein terminierter Agent darf nur "
                    "action=None erhalten."
                )

            self._remove_completed_agent(agent)

            self._clear_rewards()

            # Nächsten Agenten auswählen
            self._select_next_agent()

            return

        # =================================================
        # Fall 2:
        # Normaler Entscheidungs-Schritt
        # =================================================

        # Der bisher angesammelte Reward dieses Agenten
        # wurde gerade mit last() abgeholt.
        self._cumulative_rewards[agent] = 0.0

        # -------------------------------------------------
        # Zugehöriges SUMO-Fahrzeug
        # -------------------------------------------------

        vehicle = self.agent_to_vehicle[agent]

        # -------------------------------------------------
        # Action -> Parkplatz
        # -------------------------------------------------

        parking_area = self.PARKING_AREAS[
            int(action)
        ]

        # -------------------------------------------------
        # Aktuelle Edge des Fahrzeugs
        # -------------------------------------------------

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
            vType="car"
        )

        # =================================================
        # Keine Route gefunden
        # =================================================

        if not route.edges:

            # Wir geben dem Agenten einen starken
            # negativen Reward.
            self.rewards[agent] = -100.0

            self.terminations[agent] = True

            self.infos[agent]["reason"] = (
                "no_route"
            )

            self.completion_queue.append(
                agent
            )

            self._select_next_agent()

            self._accumulate_rewards()
            self._clear_rewards()

            return

        # =================================================
        # Route setzen
        # =================================================

        traci.vehicle.setRoute(
            vehicle,
            list(route.edges)
        )

        # =================================================
        # Parking Stop setzen
        # =================================================

        traci.vehicle.setParkingAreaStop(
            vehicle,
            parking_area,
            duration=1200
        )

        # =================================================
        # Fahrzeug als pending speichern
        # =================================================

        self.pending_vehicles[agent] = {

            "vehicle_id": vehicle,

            "action": int(action),

            "parking_area": parking_area,

            "start_time":
                traci.simulation.getTime(),

            "walking_distance":
                self.WALKING_DISTANCE[
                    parking_area
                ],

            "waiting_time": 0.0,

            "parking_failure": 0
        }

        # =================================================
        # Der Agent hat seine einzige Entscheidung
        # getroffen.
        #
        # Er bleibt aber in self.agents, bis sein
        # Ergebnis bekannt ist.
        # =================================================

        self.infos[agent]["action"] = int(action)

        self.infos[agent]["parking_area"] = (
            parking_area
        )

        self.infos[agent]["decision_time"] = (
            traci.simulation.getTime()
        )

        # =================================================
        # Simulation weiterlaufen lassen
        # =================================================

        self._advance_until_ready()

        # =================================================
        # Reward-Zwischenspeicher an alle Agenten verteilen
        # =================================================

        self._accumulate_rewards()
        self._clear_rewards()

    # =====================================================
    # Simulation weiterführen
    #
    # Läuft bis:
    #
    # 1. neuer Agent eine Entscheidung benötigt
    # oder
    # 2. bestehender Agent abgeschlossen ist
    # =====================================================

    def _advance_until_ready(self):

        while True:

            # =================================================
            # Ein fertig gewordener Agent hat Priorität
            # =================================================

            if self.completion_queue:

                self.agent_selection = (
                    self.completion_queue.pop(0)
                )

                return

            # =================================================
            # Neue Fahrzeuge haben eine Entscheidung nötig
            # =================================================

            if self.decision_queue:

                self.agent_selection = (
                    self.decision_queue.pop(0)
                )

                return

            # =================================================
            # Einen SUMO-Schritt durchführen
            # =================================================

            traci.simulationStep()

            current_time = (
                traci.simulation.getTime()
            )

            # =================================================
            # Aktuelle Fahrzeuge
            # =================================================

            current_vehicles = set(
                traci.vehicle.getIDList()
            )

            # =================================================
            # Neue Fahrzeuge erkennen
            # =================================================

            new_vehicles = (
                current_vehicles
                - self.known_vehicles
            )

            # =================================================
            # Neue normale Fahrzeuge als Agenten erzeugen
            # =================================================

            for vehicle in sorted(new_vehicles):

                if vehicle.startswith(
                    "Störverkehr"
                ):
                    continue

                self._create_agent_for_vehicle(
                    vehicle,
                    current_time
                )

            # Bekannte Fahrzeuge aktualisieren
            self.known_vehicles.update(
                current_vehicles
            )

            # =================================================
            # Pending Fahrzeuge bearbeiten
            # =================================================

            for agent in list(
                self.pending_vehicles.keys()
            ):

                data = self.pending_vehicles[
                    agent
                ]

                vehicle = data[
                    "vehicle_id"
                ]

                # ---------------------------------------------
                # Fahrzeug ist noch in SUMO
                # ---------------------------------------------

                if vehicle in current_vehicles:

                    # -----------------------------------------
                    # Aktuellen Parkzustand prüfen
                    # -----------------------------------------

                    is_parking = (
                        traci.vehicle.isStoppedParking(
                            vehicle
                        )
                    )

                    # -----------------------------------------
                    # Fahrzeug fährt noch / parkt noch nicht
                    # -----------------------------------------

                    if not is_parking:

                        speed = (
                            traci.vehicle.getSpeed(
                                vehicle
                            )
                        )

                        # Wartezeit
                        if speed < 0.1:

                            data["waiting_time"] += 1.0

                        # -------------------------------------
                        # Parking Failure prüfen
                        # -------------------------------------

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

                        # Fahrzeug ist an der
                        # Parkplatz-Edge angekommen
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

                            # Parkplatz ist voll
                            if occupied >= capacity:

                                data[
                                    "parking_failure"
                                ] += 1

                    # -----------------------------------------
                    # Fahrzeug hat erfolgreich geparkt
                    # -----------------------------------------

                    if is_parking:

                        self._finish_agent(
                            agent,
                            current_time,
                            reason="parked"
                        )

                # ---------------------------------------------
                # Fahrzeug existiert nicht mehr
                # ---------------------------------------------

                else:

                    self._finish_agent(
                        agent,
                        current_time,
                        reason="vehicle_left_simulation"
                    )

            # =================================================
            # Gibt es jetzt einen fertigen oder neuen Agenten?
            # =================================================

            if self.completion_queue:

                self.agent_selection = (
                    self.completion_queue.pop(0)
                )

                return

            if self.decision_queue:

                self.agent_selection = (
                    self.decision_queue.pop(0)
                )

                return

            # =================================================
            # Ende der Episode
            # =================================================

            if (
                current_time >= self.SIMULATION_END
                and not self.pending_vehicles
                and not self.decision_queue
                and not self.completion_queue
            ):

                self.agent_selection = None

                return

    # =====================================================
    # Einen neuen PettingZoo-Agenten erzeugen
    # =====================================================

    def _create_agent_for_vehicle(
        self,
        vehicle,
        current_time
    ):

        # -------------------------------------------------
        # Prüfen, ob noch Agenten-Slots verfügbar sind
        # -------------------------------------------------

        if (
            self.next_agent_index
            >= len(self.possible_agents)
        ):

            raise RuntimeError(
                "MAX_AGENT_SLOTS erreicht. "
                "Erhöhe max_agent_slots."
            )

        # -------------------------------------------------
        # Einen neuen Agenten-Slot verwenden
        # -------------------------------------------------

        agent = self.possible_agents[
            self.next_agent_index
        ]

        self.next_agent_index += 1

        # -------------------------------------------------
        # Mapping erstellen
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

        self.agents.append(agent)

        self.rewards[agent] = 0.0

        self._cumulative_rewards[
            agent
        ] = 0.0

        self.terminations[
            agent
        ] = False

        self.truncations[
            agent
        ] = False

        # -------------------------------------------------
        # State des neu erschienenen Fahrzeugs
        # -------------------------------------------------

        self.observations[
            agent
        ] = self._build_observation(
            vehicle
        )

        # -------------------------------------------------
        # Informationen
        # -------------------------------------------------

        self.infos[
            agent
        ] = {

            "vehicle_id": vehicle,

            "spawn_time": current_time,

            "simulation_finished": False
        }

        # -------------------------------------------------
        # Agent benötigt Entscheidung
        # -------------------------------------------------

        self.decision_queue.append(
            agent
        )

    # =====================================================
    # Fahrzeug abschließen
    # =====================================================

    def _finish_agent(
        self,
        agent,
        current_time,
        reason
    ):

        if agent not in self.pending_vehicles:
            return

        data = self.pending_vehicles[
            agent
        ]

        # -------------------------------------------------
        # Tatsächlichen Reward berechnen
        # -------------------------------------------------

        reward = calculate_reward(

            data["vehicle_id"],

            data,

            current_time,

            self.W_DRIVE,
            self.W_WAIT,
            self.W_WALK,
            self.W_FAILURE
        )

        # -------------------------------------------------
        # Reward diesem Agenten zuordnen
        # -------------------------------------------------

        self.rewards[
            agent
        ] += reward

        # -------------------------------------------------
        # Agent ist fertig
        # -------------------------------------------------

        self.terminations[
            agent
        ] = True

        # -------------------------------------------------
        # Zusatzinformationen
        # -------------------------------------------------

        self.infos[
            agent
        ]["finish_time"] = (
            current_time
        )

        self.infos[
            agent
        ]["reason"] = reason

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

        # -------------------------------------------------
        # Pending-Eintrag entfernen
        # -------------------------------------------------

        del self.pending_vehicles[
            agent
        ]

        # -------------------------------------------------
        # Agent muss noch seinen finalen
        # step(None) bekommen.
        # -------------------------------------------------

        self.completion_queue.append(
            agent
        )

    # =====================================================
    # Fertigen Agenten aus self.agents entfernen
    # =====================================================

    def _remove_completed_agent(
        self,
        agent
    ):

        vehicle = self.agent_to_vehicle.get(
            agent
        )

        # Mapping entfernen
        if vehicle is not None:

            self.vehicle_to_agent.pop(
                vehicle,
                None
            )

        self.agent_to_vehicle.pop(
            agent,
            None
        )

        # Agent aus aktiver Agentenliste entfernen
        if agent in self.agents:

            self.agents.remove(
                agent
            )

        # PettingZoo-Dictionaries bereinigen
        self.rewards.pop(
            agent,
            None
        )

        self._cumulative_rewards.pop(
            agent,
            None
        )

        self.terminations.pop(
            agent,
            None
        )

        self.truncations.pop(
            agent,
            None
        )

        self.infos.pop(
            agent,
            None
        )

        self.observations.pop(
            agent,
            None
        )

    # =====================================================
    # Nächsten Agenten auswählen
    # =====================================================

    def _select_next_agent(self):

        # Fertige Agenten zuerst
        if self.completion_queue:

            self.agent_selection = (
                self.completion_queue.pop(0)
            )

            return

        # Danach neue Fahrzeuge
        if self.decision_queue:

            self.agent_selection = (
                self.decision_queue.pop(0)
            )

            return

        # Keine sofortige Entscheidung:
        # Simulation bis zum nächsten Ereignis laufen lassen
        self._advance_until_ready()

    # =====================================================
    # State erzeugen
    #
    # ERSTE VERSION:
    # Noch dein bisheriger 12-dimensionaler State.
    # =====================================================

    def _build_observation(
        self,
        vehicle
    ):

        current_edge = traci.vehicle.getRoadID(
            vehicle
        )

        observation = []

        for parking_area in self.PARKING_AREAS:

            # ---------------------------------------------
            # Belegung
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Parking Lane -> Edge
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Route
            # ---------------------------------------------

            route = (
                traci.simulation.findRoute(
                    current_edge,
                    parking_edge,
                    vType="car"
                )
            )

            # ---------------------------------------------
            # Entfernung
            # ---------------------------------------------

            distance_normalized = min(
                route.length / 1000.0,
                1.0
            )

            # ---------------------------------------------
            # Fahrzeit
            # ---------------------------------------------

            travel_time_normalized = min(
                route.travelTime / 300.0,
                1.0
            )

            # ---------------------------------------------
            # Fußweg
            #
            # Bleibt in dieser ersten Version
            # noch im State.
            # ---------------------------------------------

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
    # Globale Beobachtung
    # =====================================================

    def observe(self, agent):

        return self.observations[
            agent
        ]

    # =====================================================
    # Globaler State
    # =====================================================

    def state(self):

        if not self.agents:

            return np.array(
                [],
                dtype=np.float32
            )

        return np.concatenate([
            self.observations[agent]
            for agent in self.agents
        ])

    # =====================================================
    # Render
    # =====================================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.agent_selection is None:
            print("Keine aktive Agentenentscheidung.")
            return

        agent = self.agent_selection

        vehicle = self.agent_to_vehicle.get(
            agent
        )

        print(
            f"Agent: {agent} | "
            f"SUMO-Fahrzeug: {vehicle}"
        )

    # =====================================================
    # Environment schließen
    # =====================================================

    def close(self):

        if traci.isLoaded():
            traci.close()


# =========================================================
# PettingZoo Environment Factory
# =========================================================

def env(
    render_mode=None,
    max_agent_slots=2000
):

    environment = ParkingMultiAgentEnv(
        max_agent_slots=max_agent_slots,
        render_mode=render_mode
    )

    environment = (
        wrappers.AssertOutOfBoundsWrapper(
            environment
        )
    )

    environment = (
        wrappers.OrderEnforcingWrapper(
            environment
        )
    )

    return environment