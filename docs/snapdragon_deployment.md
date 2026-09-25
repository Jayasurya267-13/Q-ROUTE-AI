# Snapdragon deployment status

## Implemented

The current prototype uses a scikit-learn RandomForestRegressor serialized with joblib. Inference runs locally on the development PC and supplies a congestion class to the route-cost model. No Qualcomm AI Hub conversion, Snapdragon runtime, or NPU execution has been performed.

## Target path

The Random Forest object is not a Qualcomm AI Hub deployable artifact by itself. For a practical portable path, train an additional small neural model in a framework supported by the chosen Qualcomm AI Hub workflow, export to ONNX, validate numerical parity against the existing Random Forest baseline, then compile/optimize through AI Hub and integrate the generated runtime artifact on a supported Snapdragon PC. Alternatively, investigate ONNX export of the tree ensemble and validate operator/runtime support before selecting it. Keep the current Random Forest as the baseline until parity and device performance are measured.

Snapdragon/NPU execution is the target deployment configuration; current development and validation were performed on the available development PC. Benchmark claims must be collected on actual target hardware and reported separately.
