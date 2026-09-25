"""Interactive dashboard backed by project inference and routing modules."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from ai.inference import TrafficPredictor
from qpso.optimizer import QPSOOptimizer
from routing.road_network import build_demo_network
from routing.route_utils import apply_ai_prediction, route_metrics
from routing.multi_vehicle import optimize_vehicles
from routing.dynamic_routing import reroute_after_incident

st.set_page_config(page_title="Q-ROUTE AI", page_icon="🚗", layout="wide")
st.title("Q-ROUTE AI")
st.subheader("On-Device Intelligent Traffic Prediction and Quantum-Inspired Route Optimization")
st.caption("AI predicts congestion. QPSO searches feasible routes on classical hardware.")

@st.cache_resource
def model(): return TrafficPredictor()

try:
    prediction=model().predict(2500,288.5,0,0,40,8,2,10)
except Exception as exc:
    st.error(f"Traffic model could not be loaded: {exc}")
    st.stop()
graph=build_demo_network()
apply_ai_prediction(graph,prediction)
optimizer=QPSOOptimizer(36,80,.75,seed=7)
route,fitness=optimizer.optimize(graph.all_routes("A","F"),fitness_fn=lambda r:route_metrics(graph,r).fitness)
metrics=route_metrics(graph,route)

st.header("Traffic Intelligence")
a,b,c,d=st.columns(4)
a.metric("Current traffic","2,500 vehicles/hour")
b.metric("Predicted next hour",f"{prediction['predicted_traffic']:,.0f} vehicles/hour")
c.metric("Congestion",prediction["congestion"])
d.metric("AI model","Random Forest · local")

st.header("Road Network")
fig,ax=plt.subplots(figsize=(10,4))
layout={"A":(0,1),"B":(1,2),"C":(1,0),"D":(2,2),"E":(2,0),"F":(4,1),"G":(2,3),"H":(3,2),"I":(2,-1),"J":(3,0)}
ng=nx.DiGraph();ng.add_nodes_from(graph.nodes);ng.add_edges_from(graph.roads)
nx.draw_networkx_nodes(ng,layout,node_color="#d9efff",node_size=700,ax=ax)
nx.draw_networkx_labels(ng,layout,font_weight="bold",ax=ax)
chosen=set(zip(route,route[1:]));colors=["#d62728" if e in chosen else "#999999" for e in ng.edges]
nx.draw_networkx_edges(ng,layout,edge_color=colors,width=[3 if e in chosen else 1.3 for e in ng.edges],arrows=True,arrowsize=16,ax=ax)
ax.axis("off");st.pyplot(fig);plt.close(fig)
st.caption("Red edges show the QPSO route on the offline demonstration road graph.")

st.header("QPSO Optimization")
cols=st.columns(5)
for col,label,value in zip(cols,["Particles","Iterations","Best fitness","Route","Distance / ETA"],[36,80,f"{fitness:.2f}"," → ".join(route),f"{metrics.distance:.1f} km / {metrics.travel_time:.1f} min"]): col.metric(label,value)
st.metric("Congestion cost",f"{metrics.congestion_cost:.1f}")
st.header("Multi-Vehicle Routing")
st.dataframe([{**r,"route":" → ".join(r["route"])} for r in optimize_vehicles(graph,optimizer=optimizer)],use_container_width=True,hide_index=True)
st.header("Dynamic Re-Routing")
if st.button("Simulate Traffic Incident",type="primary"):
    changed=reroute_after_incident(graph,edge=("D","F"),congestion="HIGH",optimizer=optimizer)
    x,y,z=st.columns(3)
    x.write("Old route: "+" → ".join(changed["old_route"]))
    y.write("New route: "+" → ".join(changed["new_route"]))
    z.metric("Cost change",f"{changed['cost_change']:+.2f}")
st.header("AI + QPSO Pipeline")
st.info("Traffic Data  ↓  AI Prediction  ↓  Congestion  ↓  Road Cost  ↓  QPSO  ↓  Optimized Route  ↓  Dynamic Re-Routing")
st.caption("Snapdragon/NPU execution is the target deployment configuration; current development and validation were performed on the available development PC.")
