"""Incident simulation and QPSO rerouting."""
from qpso.optimizer import QPSOOptimizer
from .road_network import RoadNetwork
from .route_utils import route_metrics

def reroute_after_incident(graph: RoadNetwork, start="A", destination="F", edge=("D","F"),
                           congestion="HIGH", blocked=False, optimizer=None):
    before_routes=graph.all_routes(start,destination)
    optimizer=optimizer or QPSOOptimizer()
    old_route,old_fit=optimizer.optimize(before_routes,fitness_fn=lambda r:route_metrics(graph,r).fitness)
    old_cost=route_metrics(graph,old_route).fitness
    if blocked: graph.block_road(*edge)
    else: graph.set_congestion(*edge,congestion)
    routes=graph.all_routes(start,destination)
    new_route,new_fit=optimizer.optimize(routes,fitness_fn=lambda r:route_metrics(graph,r).fitness)
    return {"old_route":old_route,"new_route":new_route,"old_cost":old_cost,
            "new_cost":route_metrics(graph,new_route).fitness,"cost_change":route_metrics(graph,new_route).fitness-old_cost,
            "incident_edge":edge,"blocked":blocked,"fitness":new_fit}
