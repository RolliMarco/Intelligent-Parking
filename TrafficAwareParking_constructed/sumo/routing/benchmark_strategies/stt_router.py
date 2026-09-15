import traci

import xml.etree.ElementTree as ET

# ---------------------------------------------------------
# SUMO-Konfiguration
# ---------------------------------------------------------

SUMO_CONFIG = (
    r"C:\Users\User\Desktop\TrafficAwareParking_constructed_4parking"
    r"\TrafficAwareParking_constructed\sumo\config\constructed.sumocfg"
)


# ---------------------------------------------------------
# Parkplätze
# ---------------------------------------------------------

parkplaetze = ["P1", "P2", "P3"]

walking_distance = {
    "P1": 300,
    "P2": 250,
    "P3": 50
}

parkplatz_lanes = {
    "P1": "HochfahrtLinks_0",
    "P2": "LinksLinks_0",
    "P3": "HochfahrtRechts_0"
}

gesamter_fussweg = 0
anzahl_fahrzeuge = 0


# ---------------------------------------------------------
# Bereits bekannte Fahrzeuge
# ---------------------------------------------------------

bekannte_fahrzeuge = set()


# ---------------------------------------------------------
# SUMO starten
# ---------------------------------------------------------

traci.start([
    "sumo-gui",
    "-c", SUMO_CONFIG
])

print("TraCI-Verbindung hergestellt.")


# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------

parking_failures = 0
bereits_gemeldete_failures = set()
zugewiesene_parkplaetze = {}


for step in range(2000):

    traci.simulationStep()

    # -----------------------------------------------------
    # Prüfen, ob zugewiesene Fahrzeuge ihren Parkplatz
    # erreicht haben und ob ein Parking Failure vorliegt
    # -----------------------------------------------------

    for fahrzeug, parkplatz in zugewiesene_parkplaetze.items():

        # Fahrzeug existiert nicht mehr in der Simulation
        if fahrzeug not in traci.vehicle.getIDList():
            continue

        aktuelle_edge = traci.vehicle.getRoadID(fahrzeug)

        # Die Lane des Parkplatzes holen
        parkplatz_lane = parkplatz_lanes[parkplatz]

        # Aus der Lane die zugehörige Edge bestimmen
        parkplatz_edge = traci.lane.getEdgeID(
            parkplatz_lane
        )

        # Wir prüfen erst, wenn das Fahrzeug die
        # Parkplatz-Edge erreicht hat
        if aktuelle_edge != parkplatz_edge:
            continue

        # -------------------------------------------------
        # Aktuelle Belegung der Parking Area
        # -------------------------------------------------

        belegung = traci.parkingarea.getVehicleCount(
            parkplatz
        )

        # Kapazität der Parking Area
        kapazitaet = int(
            traci.simulation.getParameter(
                parkplatz,
                "parkingArea.capacity"
            )
        )

        # -------------------------------------------------
        # Parking Failure
        # -------------------------------------------------

        if belegung >= kapazitaet:

            if fahrzeug not in bereits_gemeldete_failures:

                parking_failures += 1
                bereits_gemeldete_failures.add(
                    fahrzeug
                )


    # -----------------------------------------------------
    # Aktuelle Fahrzeuge
    # -----------------------------------------------------

    aktuelle_fahrzeuge = set(
        traci.vehicle.getIDList()
    )

    # Nur Fahrzeuge, die seit dem letzten Schritt
    # neu erschienen sind
    neue_fahrzeuge = (
        aktuelle_fahrzeuge - bekannte_fahrzeuge
    )


    # -----------------------------------------------------
    # Für jedes neue Fahrzeug:
    # schnellste Route zu P1, P2 und P3 berechnen
    # -----------------------------------------------------

    for fahrzeug in neue_fahrzeuge:

        if fahrzeug.startswith("Störverkehr"):
            continue

        aktuelle_edge = traci.vehicle.getRoadID(
            fahrzeug
        )

        beste_route = None
        bester_parkplatz = None
        beste_fahrzeit = float("inf")


        # -------------------------------------------------
        # Alle Parkplätze prüfen
        # -------------------------------------------------

        for parkplatz in parkplaetze:

            # ---------------------------------------------
            # Aktuelle Belegung
            # ---------------------------------------------

            belegung = traci.parkingarea.getVehicleCount(
                parkplatz
            )

            # Kapazität direkt aus SUMO auslesen
            kapazitaet = int(
                traci.simulation.getParameter(
                    parkplatz,
                    "parkingArea.capacity"
                )
            )

            # Parkplatz überspringen, wenn er voll ist
            if belegung >= kapazitaet:
                continue


            # ---------------------------------------------
            # Lane des Parkplatzes
            # ---------------------------------------------

            parkplatz_lane = parkplatz_lanes[
                parkplatz
            ]

            # Zugehörige Edge der Lane bestimmen
            parkplatz_edge = traci.lane.getEdgeID(
                parkplatz_lane
            )


            # ---------------------------------------------
            # Schnellste Route vom Fahrzeug zur
            # Parkplatz-Edge berechnen
            # ---------------------------------------------

            route = traci.simulation.findRoute(
                aktuelle_edge,
                parkplatz_edge,
                vType="car"
            )


            # Keine verwendbare Route gefunden
            if not route.edges:
                continue


            # ---------------------------------------------
            # Route mit der geringsten Fahrzeit merken
            # ---------------------------------------------

            if route.travelTime < beste_fahrzeit:

                beste_fahrzeit = route.travelTime
                beste_route = route
                bester_parkplatz = parkplatz


        # -------------------------------------------------
        # Beste Route einmalig setzen
        # -------------------------------------------------

        if beste_route is not None:

            gesamter_fussweg += walking_distance[
                bester_parkplatz
            ]

            anzahl_fahrzeuge += 1


            # Route setzen
            traci.vehicle.setRoute(
                fahrzeug,
                list(beste_route.edges)
            )


            # Fahrzeug zum Parking Area Stop schicken
            traci.vehicle.setParkingAreaStop(
                fahrzeug,
                bester_parkplatz,
                duration=1200
            )


            # Parkplatz-Zuweisung speichern
            zugewiesene_parkplaetze[
                fahrzeug
            ] = bester_parkplatz


    # -----------------------------------------------------
    # Fahrzeuge für die nächsten Schritte speichern
    # -----------------------------------------------------

    bekannte_fahrzeuge.update(
        aktuelle_fahrzeuge
    )

# ---------------------------------------------------------
# Simulation beenden
# ---------------------------------------------------------

traci.close()

tripinfo_file = (
    r"C:\Users\User\Desktop\TrafficAwareParking_constructed_4parking"
    r"\TrafficAwareParking_constructed\sumo\output\tripinfo.xml"
)

tree = ET.parse(tripinfo_file)
root = tree.getroot()

fahrzeiten = []
wartezeiten = []

for tripinfo in root.findall("tripinfo"):
    duration = float(tripinfo.get("duration"))
    waiting_time = float(tripinfo.get("waitingTime"))

    fahrzeiten.append(duration)
    wartezeiten.append(waiting_time)

durchschnitt = sum(fahrzeiten) / len(fahrzeiten)
durchschnitt_wartezeit = sum(wartezeiten) / len(wartezeiten)
durchschnittlicher_fussweg = (gesamter_fussweg / anzahl_fahrzeuge)

print(
    f"Durchschnittliche Fahrzeit: "
    f"{durchschnitt:.2f} s"
)

print(
    f"Durchschnittliche Wartezeit: "
    f"{durchschnitt_wartezeit:.2f} s"
)

print(
    f"Durchschnittlicher Fußweg: "
    f"{durchschnittlicher_fussweg:.2f} m"
)

print(
    f"Parking Failures: {parking_failures}"
)

print("Simulation beendet.")