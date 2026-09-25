"""Multi-vehicle route planning with a light shared-road congestion penalty."""
from __future__ import annotations
from qpso.optimizer import QPSOOptimizer
from .road_network import RoadNetwork
from .route_utils import route_metrics

DEFAULT_VEHICLES = [{"vehicle":"Vehicle 1","start":"A","destination":"F"},
                    {"vehicle":"Vehicle 2","start":"A","destination":"E"},
                    {"vehicle":"Vehicle 3","start":"B","destination":"F"}]

def optimize_vehicles(graph: RoadNetwork, vehicles=None, optimizer=None):
    optimizer = optimizer or QPSOOptimizer()
    results=[]
    usage: dict[tuple[str,str],int]={}
    for item in vehicles or DEFAULT_VEHICLES:
        routes=graph.all_routes(item["start"], item["destination"])
        def score(route):
            metrics=route_metrics(graph,route)
            return metrics.fitness + sum(usage.get(edge,0)*0.5 for edge in zip(route,route[1:]))
        route, fitness=optimizer.optimize(routes,fitness_fn=score)
        metrics=route_metrics(graph,route)
        for edge in zip(route,route[1:]): usage[edge]=usage.get(edge,0)+1
        results.append({"vehicle":item.get("vehicle",f"Vehicle {len(results)+1}"),"start":item["start"],
                        "destination":item["destination"],"route":route,"distance":metrics.distance,
                        "travel_time":metrics.travel_time,"congestion_cost":metrics.congestion_cost,
                        "fitness":fitness,"status":"ROUTED"})
    return results
