import sys
import jajapy
import stormpy
import numpy
import pandas

def semi_randomize(model, sseed=None):
    rng = numpy.random.default_rng(sseed)
    matrix = []
    array = random_probabilities(model.nb_states)
    for _ in range(model.nb_states):
        copy = numpy.copy(array)
        rng.shuffle(copy)
        matrix.append((copy))
    matrix = numpy.array(matrix)
    model.matrix = matrix

    return model

def randomize(model, sseed=None):
    if sseed is not None:
        numpy.random.seed(sseed)
    matrix = []
    for _ in range(model.nb_states):
        matrix.append((random_probabilities(model.nb_states)))
    matrix = numpy.array(matrix)
    model.matrix = matrix

    return model


def random_probabilities(length):
    random_numbers = numpy.random.rand(length)
    normalized_probabilities = random_numbers / random_numbers.sum()

    return normalized_probabilities


def save_jajapy_results(results, path):
    model, info = results
    dataframe = pandas.DataFrame([info])
    dataframe.to_csv(path, index=False)


def save_as_cupaal(model, training_set, model_path, training_set_path):
    with open(model_path, "w") as file:
        file.write('states\n')
        for i in range(model.nb_states):
            file.write(str(i) + ' ')
        file.write("\nlabels\n")

        unique_labels = list(set(model.labelling))
        for label in unique_labels:
            file.write(label + ' ')
        file.write("\ninitial\n")

        initial_state_str = ' '.join(map(str, model.initial_state.tolist()))
        file.write(initial_state_str)
        file.write("\ntransitions\n")

        # normalize the transition matrix to each row to sum to 1
        for i in range(model.nb_states):
            row_sum = sum(model.matrix[i])
            for j in range(model.nb_states):
                model.matrix[i][j] = model.matrix[i][j] / row_sum
        for i in range(model.nb_states):
            for j in range(model.nb_states):
                file.write(f"{model.matrix[i][j]} ")
            file.write("\n")
        file.write("emissions\n")

        for j in range(len(unique_labels)):
            for i in range(model.nb_states):
                if model.labelling[i] == unique_labels[j]:
                    file.write("1 ")
                else:
                    file.write("0 ")
            file.write("\n")

    with open(training_set_path, "w") as file:
        for time, sequence in zip(training_set.times, training_set.sequences):
            for i in range(time):
                file.write(' '.join(sequence) + '\n')


def convert_model_with_rewards(jajapy_model, reward_models):
    state_labelling = stormpy.storage.StateLabeling(jajapy_model.nb_states)
    for o in jajapy_model.getAlphabet():
        state_labelling.add_label(o)
    for s in range(jajapy_model.nb_states):
        state_labelling.add_label_to_state(jajapy_model.labelling[s],s)

    transition_matrix = jajapy_model.matrix
    transition_matrix =  stormpy.build_sparse_matrix(transition_matrix)
    components = stormpy.SparseModelComponents(transition_matrix=transition_matrix,
                                               state_labeling=state_labelling,
                                               reward_models=reward_models)
    mc = stormpy.storage.SparseDtmc(components)
    return mc


def semi_random_experiment():
    prism_file_path = sys.argv[1]
    save_path = sys.argv[2]
    iterations = int(sys.argv[3])
    epsilon = float(sys.argv[4])
    observation_count = int(sys.argv[5])
    observation_length = int(sys.argv[6])

    prism_model = jajapy.loadPrism(prism_file_path)
    # prism_model.save(f"{save_path}_prism_model")

    # training_set = prism_model.generateSet(observation_count, observation_length)
    # training_set.save(f"{save_path}_jajapy_training_set.txt")
    training_set = jajapy.loadSet(f"{save_path}_jajapy_training_set.txt")

    jajapy_model = semi_randomize(prism_model, sseed=42)
    jajapy_model.save(f"{save_path}_jajapy_model-semi.txt")
    save_as_cupaal(jajapy_model, training_set, f"{save_path}_cupaal_model-semi.txt", f"{save_path}_cupaal_training_set.txt")

    result = jajapy.BW().fit(training_set=training_set, initial_model=jajapy_model, max_it=iterations, epsilon=epsilon,
                             nb_states=jajapy_model.nb_states, verbose=3, return_data=True, processes=1)
    save_jajapy_results(result, f"{save_path}_jajapy_results-semi.csv")
    jajapy_model.save(f"{save_path}_jajapy_model_learned-semi.txt")


def random_experiment():
    prism_file_path = sys.argv[1]
    save_path = sys.argv[2]
    iterations = int(sys.argv[3])
    epsilon = float(sys.argv[4])
    observation_count = int(sys.argv[5])
    observation_length = int(sys.argv[6])

    prism_model = jajapy.loadPrism(prism_file_path)
    prism_model.save(f"{save_path}_prism_model")

    training_set = prism_model.generateSet(observation_count, observation_length)
    training_set.save(f"{save_path}_jajapy_training_set.txt")

    # jajapy_model = jajapy.MC_random(nb_states=prism_model.nb_states - 1, labelling=prism_model.labelling[1:],
    #                                 random_initial_state=True, sseed=0)
    jajapy_model = randomize(prism_model, sseed=42)
    jajapy_model.save(f"{save_path}_jajapy_model.txt")
    save_as_cupaal(jajapy_model, training_set, f"{save_path}_cupaal_model.txt", f"{save_path}_cupaal_training_set.txt")

    training_set = jajapy.loadSet(f"{save_path}_jajapy_training_set.txt")
    result = jajapy.BW().fit(training_set=training_set, initial_model=jajapy_model, max_it=iterations, epsilon=epsilon,
                             nb_states=jajapy_model.nb_states, verbose=3, return_data=True, processes=1)
    save_jajapy_results(result, f"{save_path}_jajapy_results.csv")
    jajapy_model.save(f"{save_path}_jajapy_model_learned.txt")


def accuracy_experiment():
    prism_file_path = sys.argv[1]
    save_path = sys.argv[2]

    formulae = 'P>=1 [ F "elected" ]; R{"num_rounds"}=? [ F "elected" ]'
    program = stormpy.parse_prism_program(prism_file_path)
    properties = stormpy.parse_properties(formulae, program, None)
    model = stormpy.build_model(program, properties)

    result1 = stormpy.model_checking(model, properties[0])
    result2 = stormpy.model_checking(model, properties[1])

    jajapy_learned_model = jajapy.loadMC(f"{save_path}_jajapy_model_learned.txt")
    stormpy_learned_model = convert_model_with_rewards(jajapy_learned_model, model.reward_models)

    jajapy_result1 = stormpy.model_checking(stormpy_learned_model, properties[0])
    jajapy_result2 = stormpy.model_checking(stormpy_learned_model, properties[1])

    # model, property1 prism, property2 prism, property1 learned, property2 learned, error
    with open(f"{save_path}_model_check_results.txt", "w") as file:
        file.write("model,P>=1 [ F \"elected\" ]; R{\"num_rounds\"}=? [ F \"elected\" ]\n")
        file.write(f"prism,{result1},{result2}\n")
        file.write(f"jajapy,{jajapy_result1},{jajapy_result2}\n")


if __name__ == "__main__":
    random_experiment()
    semi_random_experiment()
    accuracy_experiment()
