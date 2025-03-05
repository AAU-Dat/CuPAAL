#include <filesystem>
#include <fstream>
#include <chrono>
#include <ranges>

#include "baum_welch.h"
#include "cudd_extensions.h"
#include "helpers.h"

#define ROW_VAR_INDEX_OFFSET 0
#define ROW_VAR_INDEX_MULTIPLIER 2
#define COL_VAR_INDEX_OFFSET 1
#define COL_VAR_INDEX_MULTIPLIER 2

const std::set<std::string> MODEL_ELEMENTS = {"states", "labels", "initial", "transitions", "emissions"};

void cupaal::MarkovModel_HMM::baum_welch(unsigned int max_iterations) {
    unsigned int current_iteration = 1;
    double epsilon = Cudd_ReadEpsilon(manager);

    while (current_iteration <= max_iterations) {
        const auto alpha = forward();
        std::cout << "hello" << std::endl;
        current_iteration++;
        Cudd_RecursiveDeref(manager, alpha[0]);
        Cudd_RecursiveDeref(manager, alpha[1]);
        Cudd_RecursiveDeref(manager, alpha[2]);
        Cudd_RecursiveDeref(manager, alpha[3]);
        Cudd_RecursiveDeref(manager, alpha[4]);
        Cudd_RecursiveDeref(manager, alpha[5]);
        Cudd_RecursiveDeref(manager, alpha[6]);
        Cudd_RecursiveDeref(manager, alpha[7]);
        free(alpha);
    }
    // double previous_log_likelihood = -std::numeric_limits<double>::infinity();
    // double current_log_likelihood = -std::numeric_limits<double>::infinity();
    //
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // while (current_iteration <= max_iterations && !(current_log_likelihood - previous_log_likelihood < epsilon)) {
    //     previous_log_likelihood = current_log_likelihood;
    //     const auto alpha = forward_add();
    //     const auto beta = backward_add();
    //     const auto gamma = gamma_add(alpha, beta);
    //     const auto xi = xi_add(alpha, beta);
    //     update_model_parameters_add(gamma, xi);
    //     const auto logs = Cudd_addMonadicApply(manager, addLog, alpha[observations[0].size() - 1]);
    //     Cudd_Ref(logs);
    //     const auto log_likelihood = Cudd_addExistAbstract(manager, logs, row_cube);
    //     Cudd_Ref(log_likelihood);
    //     current_log_likelihood = Cudd_V(log_likelihood);
    //     Cudd_RecursiveDeref(manager, logs);
    //     Cudd_RecursiveDeref(manager, log_likelihood);
    //
    //     current_iteration++;
    // }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //
    // report.iterations = current_iteration - 1;
    // report.microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // report.log_likelihood = current_log_likelihood;
}

DdNode ** cupaal::MarkovModel_HMM::forward() const {
    const auto alpha = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), mapped_observations[0].size()));
    alpha[0] = Cudd_addApply(manager, Cudd_addTimes, pi, omega[mapped_observations[0][0]]);
    Cudd_Ref(alpha[0]);

    for (int t = 1; t < mapped_observations[0].size(); t++) {
        DdNode *alpha_temp = Cudd_addMatrixMultiply(manager, tau, alpha[t - 1], row_vars, n_vars);
        Cudd_Ref(alpha_temp);
        DdNode *alpha_temp2 = Cudd_addSwapVariables(manager, alpha_temp, row_vars, col_vars, n_vars);
        Cudd_Ref(alpha_temp2);

        alpha[t] = Cudd_addApply(manager, Cudd_addTimes, omega[mapped_observations[0][t]],
                                 alpha_temp2);
        Cudd_Ref(alpha[t]);
        Cudd_RecursiveDeref(manager, alpha_temp);
        Cudd_RecursiveDeref(manager, alpha_temp2);
    }
    return alpha;
}

void cupaal::MarkovModel_HMM::initialize_from_file(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return;
    }

    std::string parsing;
    std::string line;
    while (std::getline(file, line)) {
        std::string word;
        std::stringstream line_stream(line);
        while (std::getline(line_stream, word, ' ')) {
            if (MODEL_ELEMENTS.contains(word)) {
                parsing = word;
                continue;
            }
            if (parsing == "states") states.push_back(word);
            if (parsing == "labels") labels.push_back(word);
            if (parsing == "initial") initial_distribution.push_back(stod(word));
            if (parsing == "transitions") transitions.push_back(stod(word));
            if (parsing == "emissions") emissions.push_back(stod(word));
        }
    }

    number_of_states = states.size();
    number_of_labels = labels.size();
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
        transitions.data(),
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
        COL_VAR_INDEX_MULTIPLIER
    );

    Sudd_addRead(
        initial_distribution.data(),
        states.size(),
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
            &emissions[l * number_of_states],
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
            COL_VAR_INDEX_MULTIPLIER
        );
    }

    auto cube_array = new int[n_vars];
    for (int i = 0; i < n_vars; i++) {
        cube_array[i] = 1;
    }
    row_cube = Cudd_addComputeCube(manager, row_vars, cube_array, n_vars);
    Cudd_Ref(row_cube);
    delete cube_array;

    // Validate?
    // Emissions = |S| x |L|
    // Transitions = |S| x |S|
    // Initial distribution = |S|
}

void cupaal::MarkovModel_HMM::add_observation(std::vector<std::string> observation) {
    std::vector<int> mapped_observation;
    observations.push_back(observation);
    for (std::string s : observation) {
        mapped_observation.push_back(label_index_map[s]);
    }
    mapped_observations.push_back(mapped_observation);
}

void cupaal::MarkovModel_HMM::export_to_file(const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    file << "states\n";
    for (const auto &s: states) {
        file << s << " ";
    }
    file << std::endl;

    file << "labels\n";
    for (const auto &l: labels) {
        file << l << " ";
    }
    file << std::endl;

    file << "initial\n";
    for (int i = 0; i < number_of_states; i++) {
        int assignment[n_vars];
        for (int j = 0; j < (1 << n_vars); j++) {
            assignment[j] = (i >> (n_vars - j - 1)) & 1;
        }
        file << Cudd_Eval(manager, pi, assignment) << " ";
    }
    file << std::endl;

    file << "transitions\n";
    for (int row = 0; row < number_of_states; row++) {
        for (int col = 0; col < number_of_states; col++) {
            int assignment[2 * n_vars];

            for (int i = 0; i < n_vars; i++) {
                assignment[2 * i] = (row >> (n_vars - i - 1)) & 1; // Row bits
                assignment[2 * i + 1] = (col >> (n_vars - i - 1)) & 1; // Column bits
            }
            file << Cudd_Eval(manager, tau, assignment) << " ";
        }
        file << std::endl;
    }

    file << "emissions\n";
    for (int row = 0; row < number_of_labels; row++) {
        int assignment[n_vars];

        // Iterate over all possible binary assignments (columns)
        for (int col = 0; col < (1 << n_vars); col++) {
            // Generate binary assignment
            for (int j = 0; j < (1 << n_vars); j++) {
                assignment[j] = (col >> (n_vars - j - 1)) & 1;
            }
            // Evaluate the current ADD for this column assignment
            file << Cudd_Eval(manager, omega[row], assignment) << " ";
        }
        file << std::endl;
    }
}

void cupaal::MarkovModel_HMM::clean_up_cudd() const {
    Cudd_RecursiveDeref(manager, tau);
    Cudd_RecursiveDeref(manager, pi);
    Cudd_RecursiveDeref(manager, row_cube);

    for (int l = 0; l < number_of_labels; l++) {
        Cudd_RecursiveDeref(manager, omega[l]);
    }

    for (int i = 0; i < n_vars; i++) {
        Cudd_RecursiveDeref(manager, row_vars[i]);
        Cudd_RecursiveDeref(manager, col_vars[i]);
        Cudd_RecursiveDeref(manager, comp_row_vars[i]);
        Cudd_RecursiveDeref(manager, comp_col_vars[i]);
    }
}

void cupaal::MarkovModel_Matrix::initialize_model_parameters_randomly(const int seed) {
    auto index = 0;
    for (const auto &label: labels) {
        label_index_map[label] = index;
        index++;
    }

    initial_distribution_vector = static_cast<probability *>(safe_malloc(
        sizeof(probability), states.size()));

    transition_matrix = static_cast<probability *>(safe_malloc(
        sizeof(probability), states.size() * states.size()));

    labelling_matrix = static_cast<probability *>(safe_malloc(
        sizeof(probability), states.size() * labels.size()));

    const std::vector<double> initial_distribution = generate_stochastic_probabilities(
        states.size(), seed);

    for (int s = 0; s < states.size(); s++) {
        initial_distribution_vector[s] = initial_distribution[s];
    }

    for (int s = 0; s < states.size(); s++) {
        std::vector<double> transition_probabilities = generate_stochastic_probabilities(
            states.size(), seed);
        for (int s_prime = 0; s_prime < states.size(); s_prime++) {
            transition_matrix[s * states.size() + s_prime] = transition_probabilities[s_prime];
        }
    }

    for (int s = 0; s < states.size(); s++) {
        std::vector<double> labelling_probabilities = generate_stochastic_probabilities(
            labels.size(), seed);
        for (int l = 0; l < labels.size(); l++) {
            labelling_matrix[s * labels.size() + l] = labelling_probabilities[l];
        }
    }
    if (!observations.empty() && labelling_matrix != nullptr) {
        calculate_omega();
    }
}

void cupaal::MarkovModel_Matrix::print_model_parameters() const {
    std::cout << "Model Details:" << std::endl;
    std::cout << "Number of States: " << states.size() << std::endl;

    std::cout << "Initial Distribution Vector:" << std::endl;
    for (int s = 0; s < states.size(); s++) {
        std::cout << "state: " << s << " ";
        std::cout << "probability: " << initial_distribution_vector[s] << std::endl;
    }

    std::cout << "Transition Matrix:" << std::endl;
    for (int s = 0; s < states.size(); s++) {
        for (int s_prime = 0; s_prime < states.size(); s_prime++) {
            std::cout << "state " << s << " -> " << s_prime << " ";
            std::cout << transition_matrix[s * states.size() + s_prime] << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "Labelling Matrix:" << std::endl;
    for (int s = 0; s < states.size(); s++) {
        std::cout << "state: " << s << " ";
        for (int l = 0; l < labels.size(); l++) {
            std::cout << "label: " << labels[l] << " ";
            std::cout << "probability: " << labelling_matrix[s * labels.size() + l] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Omega Matrix:" << std::endl;
    for (int s = 0; s < states.size(); s++) {
        for (int l = 0; l < observations[0].size(); l++) {
            std::cout << omega_matrix[s * observations[0].size() + l] << " ";
        }
        std::cout << std::endl;
    }
}

void cupaal::MarkovModel_Matrix::calculate_omega() {
    if (observations.empty()) {
        std::cout << "You should supply some observations before trying to calculate omega." << std::endl;
        return;
    }

    omega_matrix = static_cast<probability *>(safe_malloc(
        sizeof(probability), states.size() * observations[0].size()));

    for (const auto &trace: observations) {
        for (int s = 0; s < states.size(); s++) {
            for (int l = 0; l < observations[0].size(); l++) {
                const auto label = trace[l];
                const auto label_index = label_index_map[label];
                omega_matrix[s * observations[0].size() + l] = labelling_matrix[label_index + s * labels.size()];
            }
        }
    }
}

void cupaal::MarkovModel_ADD::initialize_probabilities(probability *transition_function,
                                                       probability *labelling_function,
                                                       probability *initial_distribution) {
    Sudd_addRead(
        transition_function,
        states.size(),
        states.size(),
        manager,
        &transition_add,
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
        COL_VAR_INDEX_MULTIPLIER
    );

    Sudd_addRead(
        initial_distribution,
        states.size(),
        1,
        manager,
        &initial_distribution_add,
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

    // for (int s = 0; s < states.size(); s++) {
    //     std::vector<double> labelling_probabilities = generate_stochastic_probabilities(
    //         labels.size(), seed);
    //     for (int l = 0; l < labels.size(); l++) {
    //         labelling_matrix[s * labels.size() + l] = labelling_probabilities[l];
    //     }
    // }

    for (int l = 0; l < labels.size(); l++) {
        Sudd_addRead(
            &labelling_function[l * states.size()],
            states.size(),
            1,
            manager,
            &labelling_add[l],
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
            COL_VAR_INDEX_MULTIPLIER
        );
    }
    const auto cube_array = new int[n_vars];
    for (int i = 0; i < n_vars; i++) {
        cube_array[i] = 1;
    }
    row_cube = Cudd_addComputeCube(manager, row_vars, cube_array, n_vars);
    Cudd_Ref(row_cube);
    delete[] cube_array;
}

void cupaal::MarkovModel_ADD::initialize_helpers() {
    auto index = 0;
    for (const auto &label: labels) {
        label_index_map[label] = index;
        index++;
    }
    dump_n_rows = states.size();
    dump_n_cols = states.size();
    n_row_vars = 0;
    n_col_vars = 0;
    n_vars = ceil(log2(states.size()));
    row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    comp_row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    comp_col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    labelling_add = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), labels.size()));
}


void cupaal::MarkovModel_ADD::initialize_model_parameters_randomly(std::mt19937 generator) {
    auto index = 0;
    for (const auto &label: labels) {
        label_index_map[label] = index;
        index++;
    }

    auto initial_distribution_vector = static_cast<probability *>(safe_malloc(
        sizeof(probability), states.size()));

    auto transition_matrix = static_cast<probability *>(safe_malloc(
        sizeof(probability), states.size() * states.size()));

    auto labelling_matrix = static_cast<probability *>(safe_malloc(
        sizeof(probability), states.size() * labels.size()));

    const std::vector<double> initial_distribution = generate_stochastic_probabilities(
        states.size(), generator);
    generator.discard(states.size());

    for (int s = 0; s < states.size(); s++) {
        initial_distribution_vector[s] = initial_distribution[s];
    }

    std::cout << "Initial Distribution Vector:" << std::endl;
    for (int s = 0; s < states.size(); s++) {
        std::cout << "state: " << s << " ";
        std::cout << "probability: " << initial_distribution_vector[s] << std::endl;
    }

    for (int s = 0; s < states.size(); s++) {
        std::vector<double> transition_probabilities = generate_stochastic_probabilities(
            states.size(), generator);
        generator.discard(states.size());
        for (int s_prime = 0; s_prime < states.size(); s_prime++) {
            transition_matrix[s * states.size() + s_prime] = transition_probabilities[s_prime];
        }
    }

    std::cout << "Transition Matrix:" << std::endl;
    for (int s = 0; s < states.size(); s++) {
        for (int s_prime = 0; s_prime < states.size(); s_prime++) {
            std::cout << "state " << s << " -> " << s_prime << " ";
            std::cout << transition_matrix[s * states.size() + s_prime] << ", ";
        }
        std::cout << std::endl;
    }

    for (int s = 0; s < states.size(); s++) {
        std::vector<double> labelling_probabilities = generate_stochastic_probabilities(
            labels.size(), generator);
        generator.discard(states.size());
        for (int l = 0; l < labels.size(); l++) {
            labelling_matrix[s * labels.size() + l] = labelling_probabilities[l];
        }
    }

    std::cout << "Labelling Matrix:" << std::endl;
    for (int s = 0; s < states.size(); s++) {
        std::cout << "state: " << s << " ";
        for (int l = 0; l < labels.size(); l++) {
            std::cout << "label: " << labels[l] << " ";
            std::cout << "probability: " << labelling_matrix[s * labels.size() + l] << " ";
        }
        std::cout << std::endl;
    }

    int dump_n_rows = states.size();
    int dump_n_cols = states.size();
    int n_row_vars = 0;
    int n_col_vars = 0;
    int n_vars = ceil(log2(states.size()));
    auto row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    auto col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));

    Sudd_addRead(
        initial_distribution_vector,
        states.size(),
        1,
        manager,
        &initial_distribution_add,
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
        transition_matrix,
        states.size(),
        states.size(),
        manager,
        &transition_add,
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
        COL_VAR_INDEX_MULTIPLIER
    );

    for (int l = 0; l < labels.size(); l++) {
        Sudd_addRead(
            &labelling_matrix[l * states.size()],
            states.size(),
            1,
            manager,
            &labelling_add[l],
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
            COL_VAR_INDEX_MULTIPLIER
        );
    }
}

void cupaal::MarkovModel_ADD::print_model_parameters() const {
    const std::filesystem::path dir(getenv("HOME"));
    const std::filesystem::path full_path = dir / "CuPAAL";

    std::cout << "Model Details:" << std::endl;
    std::cout << "Number of States: " << states.size() << std::endl;

    std::cout << "Transition ADD:" << std::endl;
    Cudd_PrintDebug(manager, transition_add, 4, 4);

    std::cout << "Initial distribution ADD:" << std::endl;
    Cudd_PrintDebug(manager, initial_distribution_add, 2, 4);

    std::cout << "labelling ADD:" << std::endl;
    Cudd_PrintDebug(manager, labelling_add[0], 2, 4);
    Cudd_PrintDebug(manager, labelling_add[1], 2, 4);
    Cudd_PrintDebug(manager, labelling_add[2], 2, 4);
    // write_dd_to_dot(manager, labelling_add, (full_path / "B_labelling.dot").c_str());
    // write_dd_to_dot(manager, transition_add, (full_path / "A_transition.dot").c_str());
    // write_dd_to_dot(manager, initial_distribution_add, (full_path / "C_initial_distribution.dot").c_str());
}

DdNode **cupaal::MarkovModel_ADD::forward_add() const {
    const auto alpha = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    alpha[0] = Cudd_addApply(manager, Cudd_addTimes, initial_distribution_add, labelling_add[0]);
    Cudd_Ref(alpha[0]);

    for (int t = 1; t < observations[0].size(); t++) {
        DdNode *alpha_temp = Cudd_addMatrixMultiply(manager, transition_add, alpha[t - 1], row_vars, n_vars);
        Cudd_Ref(alpha_temp);
        DdNode *alpha_temp2 = Cudd_addSwapVariables(manager, alpha_temp, row_vars, col_vars, n_vars);
        Cudd_Ref(alpha_temp2);

        alpha[t] = Cudd_addApply(manager, Cudd_addTimes, labelling_add[label_index_map.at(observations[0][t])],
                                 alpha_temp2);
        Cudd_Ref(alpha[t]);
        Cudd_RecursiveDeref(manager, alpha_temp);
        Cudd_RecursiveDeref(manager, alpha_temp2);
    }
    return alpha;
}

DdNode **cupaal::MarkovModel_ADD::backward_add() const {
    const auto beta = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    beta[observations[0].size() - 1] = Cudd_ReadOne(manager);
    DdNode *temporary_transition_add = Cudd_addSwapVariables(manager, transition_add, col_vars, row_vars, n_vars);
    Cudd_Ref(temporary_transition_add);

    for (int t = observations[0].size() - 2; t >= 0; t--) {
        DdNode *beta_temp = Cudd_addApply(manager, Cudd_addTimes, beta[t + 1],
                                          labelling_add[label_index_map.at(observations[0][t + 1])]);
        Cudd_Ref(beta_temp);
        DdNode *beta_temp2 = Cudd_addMatrixMultiply(manager, temporary_transition_add, beta_temp, row_vars, n_vars);
        Cudd_Ref(beta_temp2);
        auto beta_value = Cudd_addSwapVariables(manager, beta_temp2, row_vars, col_vars, n_vars);
        beta[t] = beta_value;
        Cudd_Ref(beta[t]);
        Cudd_RecursiveDeref(manager, beta_temp);
        Cudd_RecursiveDeref(manager, beta_temp2);
    }
    Cudd_RecursiveDeref(manager, temporary_transition_add);
    return beta;
}

DdNode **cupaal::MarkovModel_ADD::gamma_add(DdNode **alpha, DdNode **beta) const {
    const auto gamma = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    DdNode *scalar = Cudd_addExistAbstract(manager, alpha[observations[0].size() - 1], row_cube);
    Cudd_Ref(scalar);

    for (unsigned long t = 0; t < observations[0].size(); t++) {
        DdNode *gamma_temp = Cudd_addApply(manager, Cudd_addTimes, alpha[t], beta[t]);
        Cudd_Ref(gamma_temp);
        gamma[t] = Cudd_addApply(manager, Cudd_addDivide, gamma_temp, scalar);
        Cudd_Ref(gamma[t]);
        Cudd_RecursiveDeref(manager, gamma_temp);
    }
    Cudd_RecursiveDeref(manager, scalar);
    return gamma;
}

DdNode **cupaal::MarkovModel_ADD::xi_add(DdNode **alpha, DdNode **beta) const {
    const auto xi = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    DdNode *scalar = Cudd_addExistAbstract(manager, alpha[observations[0].size() - 1], row_cube);
    Cudd_Ref(scalar);
    DdNode *temporary_transition_add = Cudd_addSwapVariables(manager, transition_add, col_vars, row_vars, n_vars);
    Cudd_Ref(temporary_transition_add);
    DdNode *scaled_transition_add = Cudd_addApply(manager, Cudd_addDivide, temporary_transition_add, scalar);
    Cudd_Ref(scaled_transition_add);

    for (unsigned long t = 0; t < observations[0].size() - 1; t++) {
        DdNode *beta_temp = Cudd_addApply(manager, Cudd_addTimes, beta[t + 1],
                                          labelling_add[label_index_map.at(observations[0][t + 1])]);
        Cudd_Ref(beta_temp);
        auto alpha_temp = Cudd_addSwapVariables(manager, alpha[t], row_vars, col_vars, n_vars);
        Cudd_Ref(alpha_temp);
        auto kronecker_product = Cudd_addApply(manager, Cudd_addTimes, alpha_temp, beta_temp);
        Cudd_Ref(kronecker_product);
        auto xi_value = Cudd_addApply(manager, Cudd_addTimes, scaled_transition_add, kronecker_product);
        Cudd_Ref(xi_value);
        xi[t] = Cudd_addSwapVariables(manager, xi_value, row_vars, col_vars, n_vars);
        Cudd_Ref(xi[t]);
        Cudd_RecursiveDeref(manager, beta_temp);
        Cudd_RecursiveDeref(manager, alpha_temp);
        Cudd_RecursiveDeref(manager, kronecker_product);
        Cudd_RecursiveDeref(manager, xi_value);
    }
    Cudd_RecursiveDeref(manager, scalar);
    Cudd_RecursiveDeref(manager, temporary_transition_add);
    Cudd_RecursiveDeref(manager, scaled_transition_add);
    return xi;
}

void cupaal::MarkovModel_ADD::update_model_parameters_add(DdNode **gamma, DdNode **xi) {
    if (!gamma) { return; }
    DdNode *temporary_gamma_sum = gamma[0];
    DdNode *temporary_xi_sum = Cudd_ReadZero(manager);
    for (int t = 1; t < observations[0].size(); t++) {
        temporary_gamma_sum = Cudd_addApply(manager, Cudd_addPlus, temporary_gamma_sum, gamma[t]);
        temporary_xi_sum = Cudd_addApply(manager, Cudd_addPlus, temporary_xi_sum, xi[t - 1]);
    }
    // Update transitions
    transition_add = Cudd_addApply(manager, Cudd_addDivide, temporary_xi_sum, temporary_gamma_sum);
    Cudd_Ref(transition_add);
    // Update labelling
    for (int l = 0; l < labels.size(); l++) {
        labelling_add[l] = Cudd_ReadZero(manager);
    }
    for (int t = 0; t < observations[0].size(); t++) {
        labelling_add[label_index_map.at(observations[0][t])] = Cudd_addApply(
            manager, Cudd_addPlus, labelling_add[label_index_map.at(observations[0][t])], gamma[t]);
    }
    for (int l = 0; l < labels.size(); l++) {
        labelling_add[l] = Cudd_addApply(manager, Cudd_addDivide, labelling_add[l], temporary_gamma_sum);
        Cudd_Ref(labelling_add[l]);
    }
    // Update initial distribution
    initial_distribution_add = gamma[0];
    Cudd_Ref(gamma[0]);
}

cupaal::report cupaal::MarkovModel_ADD::baum_welch_add(const unsigned int max_iterations) {
    report report{};
    unsigned int current_iteration = 1;
    double epsilon = Cudd_ReadEpsilon(manager);
    double previous_log_likelihood = -std::numeric_limits<double>::infinity();
    double current_log_likelihood = -std::numeric_limits<double>::infinity();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (current_iteration <= max_iterations && !(current_log_likelihood - previous_log_likelihood < epsilon)) {
        previous_log_likelihood = current_log_likelihood;
        const auto alpha = forward_add();
        const auto beta = backward_add();
        const auto gamma = gamma_add(alpha, beta);
        const auto xi = xi_add(alpha, beta);
        update_model_parameters_add(gamma, xi);
        const auto logs = Cudd_addMonadicApply(manager, addLog, alpha[observations[0].size() - 1]);
        Cudd_Ref(logs);
        const auto log_likelihood = Cudd_addExistAbstract(manager, logs, row_cube);
        Cudd_Ref(log_likelihood);
        current_log_likelihood = Cudd_V(log_likelihood);
        Cudd_RecursiveDeref(manager, logs);
        Cudd_RecursiveDeref(manager, log_likelihood);

        current_iteration++;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    report.iterations = current_iteration - 1;
    report.microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    report.log_likelihood = current_log_likelihood;

    return report;
}

std::vector<probability> cupaal::forward_matrix(const MarkovModel_Matrix &model) {
    const unsigned long number_of_states = model.states.size();
    // Allocate alpha
    std::vector alpha(number_of_states * model.observations[0].size(), 0.0);

    // Base case: t = 0
    for (int s = 0; s < number_of_states; s++) {
        alpha[s] = model.initial_distribution_vector[s] *
                   model.labelling_matrix[s * model.labels.size() + model.label_index_map.at(model.observations[0][0])];
    }

    // Case: 0 < t <= n_obs
    for (int t = 1; t < model.observations[0].size(); t++) {
        for (int s = 0; s < number_of_states; s++) {
            double temporary_sum = 0;
            for (int s_prime = 0; s_prime < number_of_states; s_prime++) {
                temporary_sum += alpha[(t - 1) * number_of_states + s_prime] * model.transition_matrix[
                    s_prime * number_of_states + s];
            }
            alpha[t * number_of_states + s] = model.labelling_matrix[
                                                  s * model.labels.size() + model.label_index_map.at(
                                                      model.observations[0][t])] * temporary_sum;
        }
    }
    return alpha;
}

std::vector<probability> cupaal::backward_matrix(const MarkovModel_Matrix &model) {
    const unsigned long number_of_states = model.states.size();
    // Allocate beta + Base case: t = n_obs
    std::vector beta(number_of_states * model.observations[0].size(), 1.0);

    // Case: 0 <= t < n_obs
    for (int t = model.observations[0].size() - 2; t >= 0; t--) {
        for (int s = 0; s < number_of_states; s++) {
            double temporary_sum = 0.0;
            for (int s_prime = 0; s_prime < number_of_states; s_prime++) {
                // auto label_result = model.labelling_matrix[s_prime * model.labels.size() + model.label_index_map.at(model.observations[0][t])];
                temporary_sum += beta[(t + 1) * number_of_states + s_prime] *
                        model.transition_matrix[s * number_of_states + s_prime] *
                        model.labelling_matrix[s_prime * model.labels.size() + model.label_index_map.at(
                                                   model.observations[0][t + 1])];
            }
            // beta[(t - 1) * number_of_states + s] = model.omega_matrix[(t - 1) * number_of_states + s] * temp;
            beta[t * number_of_states + s] = temporary_sum;
        }
    }
    return beta;
}

std::vector<probability> cupaal::gamma_matrix(const MarkovModel_Matrix &model, const std::vector<probability> &alpha,
                                              const std::vector<probability> &beta) {
    const unsigned long number_of_states = model.states.size();
    const unsigned long number_of_obs = model.observations[0].size();

    // Allocate gamma
    std::vector<double> gamma(number_of_states * number_of_obs, 0.0);

    for (unsigned long t = 0; t < number_of_obs; t++) {
        double normalization = 0.0;

        // Compute normalization factor for time t
        for (unsigned long s = 0; s < number_of_states; s++) {
            normalization += alpha[t * number_of_states + s] * beta[t * number_of_states + s];
        }

        // Compute gamma values for time t
        for (unsigned long s = 0; s < number_of_states; s++) {
            gamma[t * number_of_states + s] =
                    (alpha[t * number_of_states + s] * beta[t * number_of_states + s]) / normalization;
        }
    }
    return gamma;
}

std::vector<probability> cupaal::xi_matrix(const MarkovModel_Matrix &model, const std::vector<probability> &alpha,
                                           const std::vector<probability> &beta) {
    const unsigned long number_of_states = model.states.size();
    const unsigned long number_of_obs = model.observations[0].size();
    std::vector<double> xi(number_of_states * number_of_states * (number_of_obs - 1), 0.0);

    for (unsigned long t = 0; t < (number_of_obs - 1); t++) {
        double normalization = 0.0;

        // Compute normalization factor for time t
        for (unsigned long i = 0; i < number_of_states; i++) {
            normalization += alpha[t * number_of_states + i] * beta[t * number_of_states + i];
        }

        // Compute xi values for time t
        for (unsigned long i = 0; i < number_of_states; i++) {
            for (unsigned long j = 0; j < number_of_states; j++) {
                xi[t * model.states.size() * model.states.size() + i * model.states.size() + j] =
                        alpha[t * number_of_states + i] *
                        model.transition_matrix[i * number_of_states + j] *
                        model.labelling_matrix[j * model.labels.size() + model.label_index_map.at(
                                                   model.observations[0][t + 1])] *
                        beta[(t + 1) * number_of_states + j] / normalization;
            }
        }
    }
    return xi;
}

cupaal::MarkovModel_Matrix cupaal::baum_welch_matrix(const MarkovModel_Matrix &model) {
    const auto alpha = forward_matrix(model);
    const auto beta = backward_matrix(model);
    const auto gamma = gamma_matrix(model, alpha, beta);
    const auto xi = xi_matrix(model, alpha, beta);

    if (model.print_calculations) {
        std::cout << "Alpha Matrix:" << std::endl;
        for (int t = 0; t < model.observations[0].size(); t++) {
            // Rows: time steps
            for (int s = 0; s < model.states.size(); s++) {
                // Columns: states
                std::cout << alpha[t * model.states.size() + s] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Beta Matrix:" << std::endl;
        for (int t = 0; t < model.observations[0].size(); t++) {
            // Rows: time steps
            for (int s = 0; s < model.states.size(); s++) {
                // Columns: states
                std::cout << beta[t * model.states.size() + s] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Gamma Matrix:" << std::endl;
        for (int t = 0; t < model.observations[0].size(); t++) {
            // Rows: time steps
            for (int s = 0; s < model.states.size(); s++) {
                // Columns: states
                std::cout << gamma[t * model.states.size() + s] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Xi Matrix:" << std::endl;
        for (int t = 0; t < model.observations[0].size() - 1; t++) {
            std::cout << "Time step " << t << ":" << std::endl;
            for (int i = 0; i < model.states.size(); i++) {
                // Rows: current state
                for (int j = 0; j < model.states.size(); j++) {
                    // Columns: next state
                    std::cout << xi[t * model.states.size() * model.states.size() + i * model.states.size() + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    return model;
}
