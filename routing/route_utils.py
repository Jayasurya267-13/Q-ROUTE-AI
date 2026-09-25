"""Shared route cost calculation and AI congestion integration."""
from __future__ import annotations
from dataclasses import dataclass
from .road_network import RoadNetwork

@dataclass
class RouteMetrics:
    distance: float
    travel_time: float
    congestion_cost: float
    fitness: float

DEFAULT_WEIGHTS = {"time": 1.0, "distance": 0.15, "congestion": 1.0, "risk": 0.0}

def apply_ai_prediction(graph: RoadNetwork, prediction: dict, edges=None):
    """Apply predicted level to selected demonstration roads (all if unspecified)."""
    level = prediction["congestion"]
    traffic = float(prediction["predicted_traffic"])
    for key in edges or list(graph.roads):
        graph.set_congestion(*key, level, traffic)

def route_metrics(graph: RoadNetwork, route, weights=None) -> RouteMetrics:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    roads = [graph.get_road(u,v) for u,v in zip(route, route[1:])]
    if not roads or any(r.status == "BLOCKED" for r in roads):
        raise ValueError("Route is empty or uses a blocked road")
    distance = sum(r.distance for r in roads)
    congestion = sum(r.congestion_cost for r in roads)
    travel_time = sum(r.base_travel_time * (1 + 0.15 * (r.congestion_cost - 1)) for r in roads)
    fitness = weights["time"]*travel_time + weights["distance"]*distance + weights["congestion"]*congestion
    return RouteMetrics(distance, travel_time, congestion, fitness)
