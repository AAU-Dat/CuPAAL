#include <iostream>
#include <filesystem>
#include <storm/utility/initialize.h>
#include <cuddInt.h>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"

void manual_algebraic_decision_diagram_variant() {
    DdManager *dd_manager = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    Cudd_SetEpsilon(dd_manager, 0);

    double transition_function[] = {
        0.6, 0.4,
        0.3, 0.7
    };

    double labelling_function[] = {
        0.3, 0.4,
        0.4, 0.3,
        0.3, 0.3
    };

    double initial_distribution[] = {0.7, 0.3};

    cupaal::MarkovModel_ADD model;
    model.manager = dd_manager;
    model.states = {0, 1};
    model.labels = {"r", "w", "b"};
    model.observations.push_back({
        "r", "w", "b", "b", "b", "b", "r", "w", "b", "b", "b", "b", "r", "w", "b", "b", "b", "b", "r", "w", "b", "b",
        "b", "b"
    });

    model.initialize_helpers();
    model.initialize_probabilities(transition_function, labelling_function, initial_distribution);
    Cudd_PrintDebug(dd_manager, model.transition_add, 2, 4);

    auto [iterations, microseconds, log_likelihood] = model.baum_welch_add(1000);
    Cudd_PrintDebug(dd_manager, model.transition_add, 2, 4);

    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Microseconds: " << microseconds << std::endl;
    std::cout << "Log likelihood: " << log_likelihood << std::endl;

    std::cout << "Remaining references (expecting 0): " << Cudd_CheckZeroRef(dd_manager) << std::endl;
    Cudd_Quit(dd_manager);
}

int main(int argc, char *argv[]) {
    // storm::utility::setUp();
    // storm::settings::initializeAll("CuPAAL", "CuPAAL");
    const std::filesystem::path dir(getenv("HOME"));
    const std::filesystem::path full_path = dir / "thesis" / "CuPAAL" ;
    DdManager *dd_manager = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    Cudd_SetEpsilon(dd_manager, 0);

    cupaal::MarkovModel_HMM model;
    model.manager = dd_manager;

    model.initialize_from_file("../example-model.txt");
    model.export_to_file("../example-model-output.txt");
    cupaal::write_dd_to_dot(model.manager, model.tau, (full_path / "tau.dot").c_str());

    model.add_observation({"r",  "g", "b", "b" , "b", "g", "b", "r"});
    model.baum_welch(1);

    // Added for debugging purposes
    model.clean_up_cudd();

    std::cout << "Remaining references (expecting 0): " << Cudd_CheckZeroRef(dd_manager) << std::endl;
    Cudd_Quit(dd_manager);
    exit(EXIT_SUCCESS);
}
