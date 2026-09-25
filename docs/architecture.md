# Architecture

Traffic data → Random Forest next-hour prediction → congestion classification → offline directed road graph → configurable route fitness → discrete quantum-inspired QPSO → vehicle plans → incident-driven re-optimization.

QPSO is a classical stochastic optimizer inspired by quantum-behaved particle swarm updates. It does not use quantum hardware. Candidate particles are positions over an enumerated finite set of feasible routes and are decoded only to valid paths.
