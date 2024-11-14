# -----------------------------------------------------------------------------
# This is a do over of the experiments in our paper, where the models will reuse the same parameters and observations.
# The goal is to make the results correctly, and not randomly, as is done in the paper.
# -----------------------------------------------------------------------------
# The idea is to have a script that will do the following:
# 1. Generate a model with the parameters we want to test.
# 2. Generate a set of observations.
# 3. Run the three algorithms on the observations.
# 4. Repeat 1, 2 and 3 several times.
# 5. Plot the results.
# -----------------------------------------------------------------------------

from random import uniform

import pandas as pd

from jajapy import loadPrism, BW
from jajapy.base.BW import ComputeAlphaBetaHow

NUMBER_OF_RUNS = 10


def experiment() -> pd.DataFrame:
    results = pd.DataFrame(
        columns=['model', 'run', 'timed', 'implementation', 'initial_parameters', 'final_parameters', 'time'])

    # generate random parameters for the models
    for run in range(0, NUMBER_OF_RUNS):
        # polling_baseline_model = loadPrism("examples/materials/polling.sm")
        # polling_baseline_model.instantiate(["mu", "gammax"], [1, 200])

        # cluster_baseline_model = loadPrism("examples/materials/cluster.sm")
        # cluster_baseline_model.instantiate(["ws_fail", "switch_fail", "line_fail"], [1 / 500, 1 / 4000, 1 / 5000])

        # tandem_baseline_model = loadPrism("examples/materials/tandem_3.sm")
        # tandem_baseline_model.instantiate(["mu1a", "mu1b", "mu2", "kappa"], [0.2, 1.8, 2.0, 4.0])

        philosophers_baseline_model = loadPrism("examples/materials/philosophers1.sm")
        philosophers_baseline_model.instantiate(["alpha", "betax", "zetax"], [0.3, 0.1, 0.2])

        # initial_polling_parameter_values = [uniform(0.00025, 0.0025), uniform(0.00025, 0.0025)]
        # initial_cluster_parameter_values = [uniform(1 / 5000, 1 / 500), uniform(1 / 5000, 1 / 500),
        #                                     uniform(1 / 5000, 1 / 500)]
        # initial_tandem_parameter_values = [uniform(0.1, 5.0), uniform(0.1, 5.0), uniform(0.1, 5.0), uniform(0.1, 5.0)]

        initial_philosophers_parameter_values = [uniform(0.00025, 0.0025), uniform(0.00025, 0.0025), uniform(0.00025, 0.0025)]

        for timed in [True, False]:
            # polling_observations = polling_baseline_model.generateSet(100, 30, timed=timed)
            # cluster_observations = cluster_baseline_model.generateSet(100, 30, timed=timed)
            # tandem_observations = tandem_baseline_model.generateSet(100, 30, timed=timed)
            philosophers_observations = philosophers_baseline_model.generateSet(100, 30, timed=timed)

            for impl in [ComputeAlphaBetaHow.SYMBOLIC_LOG_SEMIRING, ComputeAlphaBetaHow.MATRIX_NUMERIC,
                         ComputeAlphaBetaHow.NUMERIC_C]:
                print("run: " + str(run) + " timed: " + str(timed) + " impl: " + str(impl))
                # polling_model = loadPrism("examples/materials/polling.sm")
                # polling_model.instantiate(["mu", "gammax"], initial_polling_parameter_values)

                # cluster_model = loadPrism("examples/materials/cluster.sm")
                # cluster_model.instantiate(["ws_fail", "switch_fail", "line_fail"], initial_cluster_parameter_values)

                # tandem_model = loadPrism("examples/materials/tandem_3.sm")
                # tandem_model.instantiate(["mu1a", "mu1b", "mu2", "kappa"], initial_tandem_parameter_values)

                philosophers_model = loadPrism("examples/materials/philosophers1.sm")
                philosophers_model.instantiate(["alpha", "betax", "zetax"], initial_philosophers_parameter_values)

                # (polling_parameters, polling_result) = BW().fit_parameters(polling_observations, polling_model,
                #                                                            ["mu", "gammax"], return_data=True,
                #                                                            compute_alpha_beta_how=impl)
                # (cluster_parameters, cluster_result) = BW().fit_parameters(cluster_observations, cluster_model,
                #                                                            ["ws_fail", "switch_fail", "line_fail"],
                #                                                            return_data=True,
                #                                                            compute_alpha_beta_how=impl)
                # (tandem_parameters, tandem_result) = BW().fit_parameters(tandem_observations, tandem_model,
                #                                                          ["mu1a", "mu1b", "mu2", "kappa"],
                #                                                          return_data=True, compute_alpha_beta_how=impl)


                (philosophers_parameters, philosophers_result) = BW().fit_parameters(philosophers_observations, philosophers_model,
                                                                                     ["alpha", "betax", "zetax"],
                                                                                     return_data=True, compute_alpha_beta_how=impl)

                # polling_dataframe = pd.DataFrame([
                #     {"model": "Polling", "run": run, "timed": timed, "implementation": impl.name,
                #      "initial_parameters": initial_polling_parameter_values, "final_parameters": polling_parameters,
                #      "learning_time": polling_result['learning_time'],
                #      "learning_rounds": polling_result['learning_rounds'],
                #      "training_set_loglikelihood": polling_result['training_set_loglikelihood']}])

                # cluster_dataframe = pd.DataFrame([
                #     {"model": "Cluster", "run": run, "timed": timed, "implementation": impl.name,
                #      "initial_parameters": initial_cluster_parameter_values, "final_parameters": cluster_parameters,
                #      "learning_time": cluster_result['learning_time'],
                #      "learning_rounds": cluster_result['learning_rounds'],
                #      "training_set_loglikelihood": cluster_result['training_set_loglikelihood']}])

                # tandem_dataframe = pd.DataFrame([
                #     {"model": "Tandem", "run": run, "timed": timed, "implementation": impl.name,
                #      "initial_parameters": initial_tandem_parameter_values, "final_parameters": tandem_parameters,
                #      "learning_time": tandem_result['learning_time'],
                #      "learning_rounds": tandem_result['learning_rounds'],
                #      "training_set_loglikelihood": tandem_result['training_set_loglikelihood']}])

                philosophers_dataframe = pd.DataFrame([
                    {"model": "Philosophers", "run": run, "timed": timed, "implementation": impl.name,
                     "initial_parameters": initial_philosophers_parameter_values, "final_parameters": philosophers_parameters,
                     "learning_time": philosophers_result['learning_time'],
                     "learning_rounds": philosophers_result['learning_rounds'],
                     "training_set_loglikelihood": philosophers_result['training_set_loglikelihood']}])

                results = pd.concat([results, #polling_dataframe,
                                     #cluster_dataframe,
                                     #tandem_dataframe,
                                     philosophers_dataframe],
                                    ignore_index=True, axis=0)

    return results


if __name__ == '__main__':
    result = experiment()
    result.to_csv("redo-results.csv")
    print(result)
