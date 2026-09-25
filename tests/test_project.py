from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ai.inference import TrafficPredictor
from qpso.optimizer import QPSOOptimizer
from routing.road_network import build_demo_network
from routing.route_utils import route_metrics
from routing.multi_vehicle import optimize_vehicles
from routing.dynamic_routing import reroute_after_incident

def test_ai_model_and_inference():
    result=TrafficPredictor().predict(2500,288.5,0,0,40,8,2,10)
    assert result['predicted_traffic'] > 0
    assert result['congestion'] in {'LOW','MEDIUM','HIGH'}

def test_qpso_produces_feasible_path():
    g=build_demo_network(); routes=g.all_routes('A','F')
    route,score=QPSOOptimizer(20,40,seed=3).optimize(routes,fitness_fn=lambda r:route_metrics(g,r).fitness)
    assert route[0]=='A' and route[-1]=='F' and route in routes and score >= 0

def test_multi_vehicle_and_rerouting():
    g=build_demo_network();results=optimize_vehicles(g)
    assert len(results)==3 and all(r['route'][0]==r['start'] for r in results)
    changed=reroute_after_incident(build_demo_network(),edge=('D','F'),blocked=True)
    assert changed['new_route'][0]=='A' and changed['new_route'][-1]=='F'
    assert ('D','F') not in list(zip(changed['new_route'],changed['new_route'][1:]))

def test_dashboard_importable():
    import ast
    ast.parse((ROOT/'app/dashboard.py').read_text(encoding='utf-8-sig'))
