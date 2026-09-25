"""Offline road graph and configurable traffic-to-road cost mapping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Road:
    source: str
    target: str
    distance: float
    base_travel_time: float
    congestion_level: str = "LOW"
    status: str = "OPEN"
    predicted_traffic: float = 0.0

    @property
    def congestion_cost(self) -> float:
        return CONGESTION_COSTS[self.congestion_level]


CONGESTION_COSTS = {"LOW": 1.0, "MEDIUM": 4.0, "HIGH": 10.0}


class RoadNetwork:
    def __init__(self):
        self.roads: dict[tuple[str, str], Road] = {}
        self.nodes: set[str] = set()

    def add_road(self, source: str, target: str, distance: float, time: float,
                 congestion: str = "LOW", status: str = "OPEN") -> None:
        road = Road(source, target, distance, time, congestion, status)
        self.roads[source, target] = road
        self.nodes.update((source, target))

    def neighbors(self, node: str) -> Iterator[str]:
        return (v for (u, v), road in self.roads.items() if u == node and road.status != "BLOCKED")

    def get_road(self, source: str, target: str) -> Road:
        return self.roads[source, target]

    def all_routes(self, start: str, destination: str, max_hops: int = 12) -> list[list[str]]:
        if start not in self.nodes or destination not in self.nodes:
            raise ValueError(f"Unknown route endpoint: {start} or {destination}")
        routes: list[list[str]] = []
        def walk(node: str, path: list[str]) -> None:
            if node == destination:
                routes.append(path.copy())
                return
            if len(path) > max_hops:
                return
            for nxt in self.neighbors(node):
                if nxt not in path:
                    walk(nxt, path + [nxt])
        walk(start, [start])
        if not routes:
            raise ValueError(f"No open route from {start} to {destination}")
        return routes

    def set_congestion(self, source: str, target: str, level: str, predicted_traffic: float = 0.0):
        road = self.get_road(source, target)
        level = level.upper()
        if level not in CONGESTION_COSTS:
            raise ValueError(f"Invalid congestion level: {level}")
        road.congestion_level, road.predicted_traffic = level, predicted_traffic

    def block_road(self, source: str, target: str):
        self.get_road(source, target).status = "BLOCKED"


def build_demo_network() -> RoadNetwork:
    graph = RoadNetwork()
    rows = [
        ("A","B",4,5),("A","C",3,5),("B","D",4,5),("C","D",2,3),
        ("C","E",5,6),("B","E",6,8),("D","F",3,4),("E","F",3,4),
        ("B","G",5,7),("G","F",4,5),("D","H",4,5),("H","F",3,4),
        ("C","I",4,6),("I","H",3,4),("E","J",4,5),("J","F",4,5),
        ("G","H",2,3),("I","J",3,4),
    ]
    for u,v,d,t in rows:
        graph.add_road(u,v,d,t)
    return graph
