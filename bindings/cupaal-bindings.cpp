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
#include <src/cupaal/baum_welch.h>
#include <cuddObj.hh>
namespace py = pybind11; 

void helloworld() {
    std::cout << "Hello, World!" << std::endl;
}

PYBIND11_MODULE(cupaal_bindings, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("bw_wrapping_function", &bw_wrapping_function, "A function that wraps the Baum-Welch algorithm")
}



// Example function
void bw_wrapping_function(const std::vector<std::string>& labels, const std::vector<std::vector<std::string>>& observations, const std::vector<std::double>& initial_distribution, const std::vector<std::double>& transitions, const std::vector<std::double>& emissions, unsigned int max_iterations =100,double epsilon = 1e-6,std::chrono::seconds time = std::chrono::seconds(3600), const std::string& outputPath = "", const std::string& resultPath = "") {
    DdManager *dd_manager = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0); 
    Cudd_SetEpsilon(dd_manager, 0);

    cupaal::MarkovModel model;
    model.manager = dd_manager;

    model.labels = labels;
    model.observations = observations;
    model.initial_distribution = initial_distribution;
    model.transitions = transitions;
    model.emissions = emissions;

    make_adds(model);

    if (model.observations.size() > 1) {
        model.baum_welch_multiple_observations(iterations, epsilon, time);
    } else {
        model.baum_welch(iterations, epsilon, time);
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
    std::cout << "Remaining references (expecting 0): " << Cudd_CheckZeroRef(dd_manager) << std::endl;
    Cudd_Quit(dd_manager);
    const auto program_end = std::chrono::steady_clock::now();
    const auto elapsed_time = program_end - program_start;
    std::cout << "Total time spent(s): " << std::chrono::duration_cast<std::chrono::seconds>(elapsed_time) << std::endl;
    exit(EXIT_SUCCESS);
}

void cupaal::MarkovModel::make_adds(const MarkovModel &model) {
    number_of_states = static_cast<int>(model.states.size());
    number_of_labels = static_cast<int>(model.labels.size());
    dump_n_rows = number_of_states;
    dump_n_cols = number_of_states;
    n_row_vars = 0;
    n_col_vars = 0;
    n_vars = ceil(log2(number_of_states));
    row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    comp_row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    comp_col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    omega = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), number_of_labels));

    for (int i = 0; i < number_of_labels; i++) {
        label_index_map[labels.at(i)] = i;
    }

    // read transition into ADD form
    Sudd_addRead(
        model.transitions.data(),
        number_of_states,
        number_of_states,
        manager,
        &tau,
        &row_vars,
        &col_vars,
        &comp_row_vars,
        &comp_col_vars,
        &n_row_vars,
        &n_col_vars,
        &dump_n_rows,
        &dump_n_cols,
        ROW_VAR_INDEX_OFFSET,
        ROW_VAR_INDEX_MULTIPLIER,
        COL_VAR_INDEX_OFFSET,
        COL_VAR_INDEX_MULTIPLIER);

    Sudd_addRead(
        model.initial_distribution.data(),
        number_of_states,
        1,
        manager,
        &pi,
        &row_vars,
        &col_vars,
        &comp_row_vars,
        &comp_col_vars,
        &n_row_vars,
        &n_col_vars,
        &dump_n_rows,
        &dump_n_cols,
        ROW_VAR_INDEX_OFFSET,
        ROW_VAR_INDEX_MULTIPLIER,
        COL_VAR_INDEX_OFFSET,
        COL_VAR_INDEX_MULTIPLIER);

    for (int l = 0; l < number_of_labels; l++) {
        Sudd_addRead(
            &model.labels[l * number_of_states],
            number_of_states,
            1,
            manager,
            &omega[l],
            &row_vars,
            &col_vars,
            &comp_row_vars,
            &comp_col_vars,
            &n_row_vars,
            &n_col_vars,
            &dump_n_rows,
            &dump_n_cols,
            ROW_VAR_INDEX_OFFSET,
            ROW_VAR_INDEX_MULTIPLIER,
            COL_VAR_INDEX_OFFSET,
            COL_VAR_INDEX_MULTIPLIER);
    }

    auto cube_array = new int[n_vars];
    for (int i = 0; i < n_vars; i++) {
        cube_array[i] = 1;
    }
    row_cube = Cudd_addComputeCube(manager, row_vars, cube_array, n_vars);
    Cudd_Ref(row_cube);
    delete[] cube_array;
}
