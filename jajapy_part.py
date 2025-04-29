import sys
import jajapy
import stormpy
import numpy
import pandas

def randomize(model):
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

        #normalize the transition matrix to each row to sum to 1
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

if __name__ == "__main__" :

    prism_file_path = sys.argv[1]
    iterations = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    prism_model = jajapy.loadPrism(prism_file_path)
    prism_model.save("../results/prism_model")

    training_set = prism_model.generateSet(100, 30)
    # training_set.save("../results/jajapy_training_set.txt")

    jajapy_model = jajapy.MC_random(nb_states=prism_model.nb_states - 1, labelling=prism_model.labelling[1:], random_initial_state=True, sseed=0)
    # jajapy_model = randomize(prism_model)

    jajapy_model.save("../results/jajapy_model.txt")

    save_as_cupaal(jajapy_model, training_set, "../results/cupaal_model.txt", "../results/cupaal_training_set.txt")

    training_set = jajapy.loadSet("../results/jajapy_training_set.txt")
    result = jajapy.BW().fit(training_set=training_set, initial_model=jajapy_model, max_it=iterations, epsilon=epsilon, nb_states=jajapy_model.nb_states, verbose=3, return_data=True)
    # print(info["training_set_loglikelihood"])

    save_jajapy_results(result, "../results/jajapy_results.csv")

    jajapy_model.save("../results/jajapy_model_learned.txt")
