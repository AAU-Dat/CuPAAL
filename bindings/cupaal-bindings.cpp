#include </usr/include/python3.12/Python.h>
#include </usr/include/python3.12/pyconfig.h>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>
#include <string>
#include <iostream>
#include "cupaal/baum_welch.h"

namespace py = pybind11; 

struct cupaal_markov_model {
    std::vector<double> initial_distribution;
    std::vector<double> transitions;
    std::vector<double> emissions;
};

cupaal_markov_model bw_wrapping_function(
    const std::vector<std::string>& states,
    const std::vector<std::string>& labels, 
    const std::vector<std::vector<std::string>>& observations, 
    const std::vector<double>& initial_distribution, 
    const std::vector<double>& transitions, 
    const std::vector<double>& emissions, 
    unsigned int max_iterations = 100, 
    double epsilon = 1e-2,
    const std::string& outputPath = "",
    const std::string& resultPath = ""){
    const auto program_start = std::chrono::steady_clock::now();

    cupaal::MarkovModel model(states, labels, initial_distribution, transitions, emissions, observations);
    cupaal_markov_model model_data;
    std::chrono::seconds time = std::chrono::seconds(3600);

    
    if (model.observations.size() > 1) {
        model.baum_welch_multiple_observations(max_iterations, epsilon, time);
    } else {
        model.baum_welch(max_iterations, epsilon, time);
    }

    if (!outputPath.empty()) {
        std::cout << "Saving model to: " << outputPath << std::endl;
        model.export_to_file(outputPath);
    }

    if (!resultPath.empty()) {
        std::cout << "Saving iteration details to: " << resultPath << std::endl;
        model.save_experiment_to_csv(resultPath);
    }

    model.clean_up_cudd();
    const auto program_end = std::chrono::steady_clock::now();
    const auto elapsed_time = program_end - program_start;
    std::cout << "Total time spent(s): " << std::chrono::duration_cast<std::chrono::seconds>(elapsed_time) << std::endl;

    model_data.initial_distribution = model.initial_distribution;
    model_data.transitions = model.transitions;
    model_data.emissions = model.emissions;

    Cudd_Quit(model.manager);
    return model_data;
}

PYBIND11_MODULE(libcupaal_bindings, m) {
    py::class_<cupaal_markov_model>(m, "cupaal_markov_model")
        .def_readwrite("initial_distribution", &cupaal_markov_model::initial_distribution)
        .def_readwrite("transitions", &cupaal_markov_model::transitions)
        .def_readwrite("emissions", &cupaal_markov_model::emissions);
    m.def("cupaal_bw_symbolic", &bw_wrapping_function, "A wrapper for the Baum-Welch algorithm.");
}