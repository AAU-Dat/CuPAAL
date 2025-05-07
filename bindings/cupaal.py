import sys
sys.path.append("/workspaces/CuPAAL/build")
import libcupaal_bindings

states = ["state1", "state2", "state3"]
labels = ["label1", "label2", "label3"]
observations = [["label1", "label2"], ["label2", "label3"], ["label1", "label3"]]
initial_distribution = [0.5, 0.3, 0.2]
transitions = [0.5, 0.3, 0.2, 0.2, 0.5, 0.3, 0.3, 0.2, 0.5]
emissions = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

libcupaal_bindings.bw_wrapping_function(
    states,
    labels,
    observations,
    initial_distribution,
    transitions,
    emissions
)