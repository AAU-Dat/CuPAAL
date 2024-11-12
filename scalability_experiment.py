import sys
from jajapy import loadSet, loadPrism, BW
from numpy import zeros
import pandas as pd
from datetime import datetime

# Define parameters
min_c = 3

def experiment3(args):
    max_c = 25
    steps = 1
    nb_rep = 10

    if len(args) > 1:
        if args[1] == '--quick':
            steps = 2
            nb_rep = 5
        elif args[1] == '--fastest':
            steps = 7
            nb_rep = 3
            max_c = 19

    results = execute(max_c, steps, nb_rep)
    results.to_csv("scalability_results.csv", index=False)
    print(results)


def generateSet(c, timed=True):
    m = loadPrism("examples/materials/tandem_3.sm")
    m.instantiate(["mu1a", "mu1b", "mu2", "kappa"], [0.2, 1.8, 2.0, 4.0])
    return m.generateSet(100, 30, timed=timed)


def execute(max_c, steps, nb_rep):
    # DataFrame to store results
    results = pd.DataFrame(columns=["implementation", "c_value", "num_states", "num_nonzero", "learning_time"])

    # Experiment over multiple values of c
    for c in range(min_c, max_c, steps):
        # Load model and get state info
        m = loadPrism("examples/materials/tandem_3.sm")
        num_states = m.nb_states - 1  # Adjust for counting from 0
        nn_z = str((m.matrix.flatten() != 0).sum() - 1 ) #number of non-zero transitions

        # Loop over `timed` and `non-timed` settings
        for timed in [True, False]:
            training_set = generateSet(c, timed=timed)
            implementations = [BW.ComputeAlphaBetaHow.SYMBOLIC_LOG_SEMIRING, BW.ComputeAlphaBetaHow.CLASSIC, #new implementation
            ]

            # Repeat each experiment `nb_rep` times for each implementation
            for impl in implementations:
                for rep in range(nb_rep):
                    print(f"c: {c}, timed: {timed}, implementation: {impl.name}, repetition: {rep + 1}")

                    # if impl == new implementation:
                        # do this here
                    else:
                        # Track the start time for learning
                        start_time = datetime.now()
                        # Perform learning with BW algorithm on non-instantiated parameters
                        output_val, learning_data = BW().fit_nonInstantiatedParameters(
                        training_set, m, min_val=0.1, max_val=5.0, return_data=True, 
                        compute_alpha_beta_how=impl
                        )

                        # Calculate learning time in seconds
                        learning_time = (datetime.now() - start_time).total_seconds()

                    # Append results to the DataFrame
                    results = results.append({
                        "implementation": impl.name,
                        "c_value": c,
                        "num_states": num_states,
                        "num_nonzero": nn_z,
                        "learning_time": learning_time
                    }, ignore_index=True)

    return results


if __name__ == '__main__':
    experiment3(sys.argv)
