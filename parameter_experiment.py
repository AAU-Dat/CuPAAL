# -----------------------------------------------------------------------------
# This script replicates the experiments from our paper, ensuring the results
# are consistent and not random by reusing the same parameters and observations.
# -----------------------------------------------------------------------------
# Procedure:
# 1. Generate models with desired parameters.
# 2. Generate observations.
# 3. Run algorithms on observations.
# 4. Repeat the steps multiple times.
# 5. Plot and save the results.
# -----------------------------------------------------------------------------

from random import uniform
import pandas as pd
from jajapy import loadPrism, BW
from jajapy.base.BW import ComputeAlphaBetaHow

NUMBER_OF_RUNS = 10

if __name__ == '__main__':
    result_df = experiment()
    result_df.to_csv("parameter-results.csv", index=False)
    print(result_df)

def experiment() -> pd.DataFrame:
    results = []

    model_configs = {
        "Polling": {"path": "examples/materials/polling.sm", 
                    "params": ["mu", "gammax"],
                    "param_vals": [1, 200],
                    "init_ranges": [(0.00025, 0.0025), (0.00025, 0.0025)]},
        
        "Cluster": {"path": "examples/materials/cluster.sm", 
                    "params": ["ws_fail", "switch_fail", "line_fail"],
                    "param_vals": [1 / 500, 1 / 4000, 1 / 5000],
                    "init_ranges": [(1 / 5000, 1 / 500)] * 3},
        
        "Tandem": {"path": "examples/materials/tandem_3.sm", 
                   "params": ["mu1a", "mu1b", "mu2", "kappa"],
                   "param_vals": [0.2, 1.8, 2.0, 4.0],
                   "init_ranges": [(0.1, 5.0)] * 4},
        
        "Philosophers": {"path": "examples/materials/philosophers1.sm", 
                         "params": ["alpha", "betax", "zetax"],
                         "param_vals": [0.3, 0.1, 0.2],
                         "init_ranges": [(0.00025, 0.0025)] * 3},

        "philosophers2": {"path": "examples/materials/philosophers-gamma.sm", 
                         "params": ["alpha", "betax", "gammax", "zetax"],
                         "param_vals": [0.3, 0.1, 1.0, 0.2],
                         "init_ranges": [(0.00025, 2.0)] * 4}
    }

    # Load baseline models
    baseline_models = {name: load_and_instantiate_model(config["path"], config["params"], config["param_vals"])
                       for name, config in model_configs.items()}
    
    # Generate observations for each model with timed and untimed settings
    observations = generate_observations(baseline_models, count=100, length=30, timed_options=[True, False])

    for run in range(NUMBER_OF_RUNS):
        # Generate new initial parameters
        initial_values = {name: generate_initial_values(config["init_ranges"]) for name, config in model_configs.items()}
        # Run experiment for each configuration
        run_experiment(model_configs, observations, initial_values, results, run)
    
    return pd.DataFrame(results)

def run_experiment(models, observations, initial_values, results, run):
    for timed, impl in [(timed, impl) 
                        for timed in [True, False] 
                        for impl in [ComputeAlphaBetaHow.SYMBOLIC_LOG_SEMIRING, ComputeAlphaBetaHow.CLASSIC, #new implementation
                        ]]:
        
        for model_name, (model, init_vals) in models.items():
            obs = observations[model_name][timed]
            instantiated_model = load_and_instantiate_model(model["path"], model["params"], init_vals)
            
            print(f"Run: {run}, Model: {model_name}, Timed: {timed}, Impl: {impl}")
            if (new_implementation) :
                #run new implementation
            else:
                (final_params, result) = BW().fit_parameters(obs, instantiated_model, 
                                                         model["params"], return_data=True, 
                                                         compute_alpha_beta_how=impl)
            
            results.append({
                "model": model_name,
                "run": run,
                "timed": timed,
                "implementation": impl.name,
                "initial_parameters": init_vals,
                "final_parameters": final_params,
                "learning_time": result['learning_time'],
                "learning_rounds": result['learning_rounds'],
                "training_set_loglikelihood": result['training_set_loglikelihood']
            })

def load_and_instantiate_model(filepath, params, values):
    model = loadPrism(filepath)
    model.instantiate(params, values)
    return model

def generate_initial_values(param_ranges):
    return [uniform(*r) for r in param_ranges]

def generate_observations(models, count, length, timed_options):
    return {model_name: {timed: model.generateSet(count, length, timed=timed)
                         for timed in timed_options} 
            for model_name, model in models.items()}






