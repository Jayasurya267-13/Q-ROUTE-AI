# Competition demo (3–5 minutes)

1. Start the dashboard with `streamlit run app/dashboard.py`.
2. Explain the I-94 traffic input and show the local Random Forest next-hour prediction and its LOW/MEDIUM/HIGH class.
3. Point to the offline road graph and state that the sample graph is a demonstration network, not a live map.
4. Show the QPSO route, distance, estimated travel time, congestion cost, particles, and iterations.
5. Review the three vehicle routes and explain that shared roads receive a small congestion penalty during planning.
6. Click **Simulate Traffic Incident**. Show the old route, the new route, and the recalculated cost after D→F becomes highly congested.
7. State the core principle: AI predicts congestion; QPSO searches feasible routes. QPSO is quantum-inspired and runs on classical hardware.
8. Describe Snapdragon/AI Hub as the intended on-device deployment target. The current numbers are local PC results; do not present them as Snapdragon or NPU results.
