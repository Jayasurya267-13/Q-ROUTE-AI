"""Run the offline Q-ROUTE AI end-to-end demonstration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ai.inference import TrafficPredictor
from qpso.optimizer import QPSOOptimizer
from routing.road_network import build_demo_network
from routing.route_utils import apply_ai_prediction,route_metrics
from routing.multi_vehicle import optimize_vehicles
from routing.dynamic_routing import reroute_after_incident

def main():
    predictor=TrafficPredictor()
    prediction=predictor.predict(2500,288.5,0,0,40,8,2,10)
    graph=build_demo_network(); apply_ai_prediction(graph,prediction)
    optimizer=QPSOOptimizer(particles=36,iterations=80,alpha=.75,seed=7)
    routes=graph.all_routes("A","F")
    route,fitness=optimizer.optimize(routes,fitness_fn=lambda r:route_metrics(graph,r).fitness)
    m=route_metrics(graph,route)
    print("Q-ROUTE AI OFFLINE DEMO")
    print(f"AI prediction: {prediction['predicted_traffic']:.1f} vehicles/hour ({prediction['congestion']})")
    print(f"QPSO route: {' -> '.join(route)} | fitness={fitness:.2f} | distance={m.distance:.1f} | ETA={m.travel_time:.1f} | congestion={m.congestion_cost:.1f}")
    print("Multi-vehicle routes:")
    for r in optimize_vehicles(graph,optimizer=optimizer):
        print(f"  {r['vehicle']}: {' -> '.join(r['route'])} | {r['distance']:.1f} km | {r['travel_time']:.1f} min")
    change=reroute_after_incident(graph,edge=("D","F"),congestion="HIGH",optimizer=optimizer)
    print(f"Incident D->F HIGH: {' -> '.join(change['old_route'])} => {' -> '.join(change['new_route'])}; cost {change['old_cost']:.2f} -> {change['new_cost']:.2f} (delta {change['cost_change']:+.2f})")

if __name__=="__main__": main()
