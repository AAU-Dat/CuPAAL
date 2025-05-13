import sys
sys.path.append("/workspaces/CuPAAL/build")
import libcupaal_bindings
import jajapy as ja
import numpy as np
from decimal import Decimal 

original_model = ja.loadMC("leader_sync_jajapy_run0.txt")

# We generate 1000 sequences of 10 observations for each set
training_set = ja.loadSet("leader_sync_observations_jajapy.txt")

def jajapymodel_to_cupaal(model, training_set):
    labels = list(set(model.labelling))
    labels.remove("empty")
    labels.remove("init")
    labels.append("init")
    labels.append("empty")
    initial_states = model.initial_state
    transitions = model.matrix.flatten().tolist()
    print("Transitions:", transitions[0])
    num_states = len(model.matrix)
    states = make_states(num_states)
    observations = jajapyobservation_to_cupaal(training_set)
    emissions = make_emission_matrix_to_mc(model) 

    libcupaal_bindings.bw_wrapping_function(
    states,
    labels,
    observations,
    initial_states,
    transitions,
    emissions)

def jajapyobservation_to_cupaal(training_set):
    full_list = []
    for seq, count in zip(training_set.sequences, training_set.times):
        full_list.extend([seq] * count)
    return full_list

def make_emission_matrix_to_hmm(model):
    unique_labels = list(set(model.labelling))
    unique_labels.remove("empty")
    unique_labels.remove("init")
    unique_labels.append("init")
    unique_labels.append("empty")

    emission_matrix = np.zeros((model.nb_states, len(unique_labels)))
    for j in range(len(unique_labels)):
        for i in range(model.nb_states):
            if model.labelling[i] == unique_labels[j]:
                emission_matrix[i][j] = 1
    
    return emission_matrix.flatten().tolist()

def make_emission_matrix_to_mc(model):
    labels = model.labelling
    unique_labels = list(set(model.labelling))
    unique_labels.remove("empty")
    unique_labels.remove("init")
    unique_labels.append("init")
    unique_labels.append("empty")
    label_to_index = {}

    num_states = len(labels)
    num_labels = len(unique_labels)

    # Initialize emission matrix with zeros
    emission_matrix = np.zeros((num_labels, num_states))

    # Set deterministic 1.0 entries
    for j in range(len(unique_labels)):
        for i in range(num_states):
            if labels[i] == unique_labels[j]:
                emission_matrix[j][i] = 1
    return emission_matrix.flatten().tolist()

def make_states(num_states):
    states = []
    for i in range(num_states):
        states.append(f"state{i}")
    return states

    # set = (observastions, times )
    # set = ([["label1", "label2"], ["label2", "label3"], ["label1", "label3"]], [1, 4, 5])


# states = ["state1", "state2", "state3"]
# labels = ["label1", "label2", "label3"]
# observations = [["label1", "label2"], ["label2", "label3"], ["label1", "label3"]]
# initial_distribution = [0.5, 0.3, 0.2]
# transitions = [0.5, 0.3, 0.2, 0.2, 0.5, 0.3, 0.3, 0.2, 0.5]
# emissions = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

jajapymodel_to_cupaal(original_model, training_set)