"""Repeated offline route-method comparison; writes measured runs to CSV."""
from pathlib import Path
import sys,time,csv
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ai.inference import TrafficPredictor
from qpso.optimizer import QPSOOptimizer
from routing.road_network import build_demo_network
from routing.route_utils import apply_ai_prediction,route_metrics

def main(repeats=10):
    p=TrafficPredictor().predict(2500,288.5,0,0,40,8,2,10)
    rows=[]
    for method in ("Shortest Path","QPSO","AI + QPSO"):
        for run in range(repeats):
            g=build_demo_network()
            if method=="AI + QPSO": apply_ai_prediction(g,p)
            candidates=g.all_routes("A","F")
            t=time.perf_counter()
            if method=="Shortest Path":
                route=min(candidates,key=lambda r:sum(g.get_road(u,v).base_travel_time for u,v in zip(r,r[1:])))
            else:
                route,_=QPSOOptimizer(30,80,.75,seed=run).optimize(candidates,fitness_fn=lambda r:route_metrics(g,r).fitness)
            elapsed=(time.perf_counter()-t)*1000; m=route_metrics(g,route)
            rows.append({"method":method,"run":run+1,"route":" -> ".join(route),"distance":m.distance,
                         "travel_time":m.travel_time,"congestion_cost":m.congestion_cost,"route_fitness":m.fitness,"runtime_ms":elapsed})
    out=ROOT/"benchmarks"/"results.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f"Wrote {len(rows)} measured runs to {out}")
    for method in ("Shortest Path","QPSO","AI + QPSO"):
        subset=[r for r in rows if r["method"]==method]
        print(f"{method}: mean runtime {sum(r['runtime_ms'] for r in subset)/len(subset):.3f} ms; mean fitness {sum(r['route_fitness'] for r in subset)/len(subset):.3f}")
if __name__=="__main__":main()
