# TrafficAwareParking – konstruiertes SUMO-Szenario

Dieses Szenario ist absichtlich synthetisch und übersichtlich:
- 1 Krankenhaus im Zentrum
- genau 4 Parkplätze: P1, P2, P3, P4
- vier Hauptzufahrten
- einfache symmetrische Straßenstruktur
- 20 Stellplätze je Parkplatz als Startwert
- Demo-Verkehr nur zum technischen Funktionstest

## Start
`sumo/build_and_run.bat` doppelklicken.

Alternativ in PowerShell:
`cd <Ordner>\sumo`
`./build_and_run.bat`

## Dateien
- network/constructed.nod.xml – Knoten
- network/constructed.edg.xml – Straßen
- parking/constructed.add.xml – Krankenhaus + 4 Parking Areas
- routes/demo.rou.xml – einfacher Testverkehr
- config/constructed.sumocfg – Simulation
- build_and_run.bat – Netz bauen und SUMO-GUI starten

Die Zahlen sind Modellannahmen. Für die eigentliche Untersuchung werden wir danach Verkehrsnachfrage, Parkplatzkapazitäten und die Zielfunktion sauber definieren.
