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

DdNode **cupaal::MarkovModel::calculate_alpha(const std::vector<int> &observation) const {
    const auto alpha = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observation.size()));
    alpha[0] = Cudd_addApply(manager, Cudd_addTimes, pi, omega[observation[0]]);
    Cudd_Ref(alpha[0]);

    for (unsigned int t = 1; t < observation.size(); t++) {
        DdNode *alpha_temp = Cudd_addMatrixMultiply(manager, tau, alpha[t - 1], row_vars, n_vars);
        Cudd_Ref(alpha_temp);
        DdNode *alpha_temp2 = Cudd_addSwapVariables(manager, alpha_temp, row_vars, col_vars, n_vars);
        Cudd_Ref(alpha_temp2);

        alpha[t] = Cudd_addApply(manager, Cudd_addTimes, omega[observation[t]], alpha_temp2);
        Cudd_Ref(alpha[t]);
        Cudd_RecursiveDeref(manager, alpha_temp);
        Cudd_RecursiveDeref(manager, alpha_temp2);
    }
    return alpha;
}

DdNode **cupaal::MarkovModel::calculate_beta(const std::vector<int> &observation) const {
    const auto beta = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observation.size()));
    beta[observation.size() - 1] = Cudd_ReadOne(manager);
    Cudd_Ref(beta[observation.size() - 1]);
    DdNode *temporary_transition_add = Cudd_addSwapVariables(manager, tau, col_vars, row_vars, n_vars);
    Cudd_Ref(temporary_transition_add);

    for (int t = static_cast<int>(observation.size() - 2); t >= 0; t--) {
        DdNode *beta_temp = Cudd_addApply(manager, Cudd_addTimes, beta[t + 1], omega[observation[t + 1]]);
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

DdNode **cupaal::MarkovModel::calculate_gamma(DdNode **alpha, DdNode **beta) const {
    const auto gamma = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    DdNode *scalar = Cudd_addExistAbstract(manager, alpha[observations[0].size() - 1], row_cube);
    Cudd_Ref(scalar);

    for (unsigned int t = 0; t < observations[0].size(); t++) {
        DdNode *gamma_temp = Cudd_addApply(manager, Cudd_addTimes, alpha[t], beta[t]);
        Cudd_Ref(gamma_temp);
        gamma[t] = Cudd_addApply(manager, Cudd_addDivide, gamma_temp, scalar);
        Cudd_Ref(gamma[t]);
        Cudd_RecursiveDeref(manager, gamma_temp);
    }
    Cudd_RecursiveDeref(manager, scalar);
    return gamma;
}

DdNode **cupaal::MarkovModel::calculate_xi(DdNode **alpha, DdNode **beta, const std::vector<int> &observation) const {
    const auto xi = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observation.size()));
    DdNode *scalar = Cudd_addExistAbstract(manager, alpha[observation.size() - 1], row_cube);
    Cudd_Ref(scalar);
    DdNode *temporary_transition_add = Cudd_addSwapVariables(manager, tau, col_vars, row_vars, n_vars);
    Cudd_Ref(temporary_transition_add);
    DdNode *scaled_transition_add = Cudd_addApply(manager, Cudd_addDivide, temporary_transition_add, scalar);
    Cudd_Ref(scaled_transition_add);

    for (unsigned long t = 0; t < observation.size() - 1; t++) {
        DdNode *beta_temp = Cudd_addApply(manager, Cudd_addTimes, beta[t + 1], omega[observation[t + 1]]);
        Cudd_Ref(beta_temp);
        const auto alpha_temp = Cudd_addSwapVariables(manager, alpha[t], row_vars, col_vars, n_vars);
        Cudd_Ref(alpha_temp);
        const auto kronecker_product = Cudd_addApply(manager, Cudd_addTimes, alpha_temp, beta_temp);
        Cudd_Ref(kronecker_product);
        const auto xi_value = Cudd_addApply(manager, Cudd_addTimes, scaled_transition_add, kronecker_product);
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

void cupaal::MarkovModel::update_model_parameters(DdNode **gamma, DdNode **xi) {
    if (!gamma || !xi) { return; }
    DdNode *temporary_gamma_sum = Cudd_ReadZero(manager);
    Cudd_Ref(temporary_gamma_sum);
    DdNode *temporary_xi_sum = Cudd_ReadZero(manager);
    Cudd_Ref(temporary_xi_sum);
    for (unsigned int t = 0; t < mapped_observations[0].size() - 1; t++) {
        DdNode *temporary_gamma_sum2 = Cudd_addApply(manager, Cudd_addPlus, temporary_gamma_sum, gamma[t]);
        Cudd_Ref(temporary_gamma_sum2);
        Cudd_RecursiveDeref(manager, temporary_gamma_sum);
        temporary_gamma_sum = temporary_gamma_sum2;
        DdNode *temporary_xi_sum2 = Cudd_addApply(manager, Cudd_addPlus, temporary_xi_sum, xi[t]);
        Cudd_Ref(temporary_xi_sum2);
        Cudd_RecursiveDeref(manager, temporary_xi_sum);
        temporary_xi_sum = temporary_xi_sum2;
    }
    // Update transitions
    Cudd_RecursiveDeref(manager, tau);
    tau = Cudd_addApply(manager, Cudd_addDivide, temporary_xi_sum, temporary_gamma_sum);
    Cudd_Ref(tau);

    DdNode *temporary_gamma_sum2 = Cudd_addApply(manager, Cudd_addPlus, temporary_gamma_sum,
                                                 gamma[mapped_observations[0].size() - 1]);
    Cudd_Ref(temporary_gamma_sum2);
    Cudd_RecursiveDeref(manager, temporary_gamma_sum);
    temporary_gamma_sum = temporary_gamma_sum2;
    // Update labelling
    for (unsigned int l = 0; l < labels.size(); l++) {
        Cudd_RecursiveDeref(manager, omega[l]);
        omega[l] = Cudd_ReadZero(manager);
        Cudd_Ref(omega[l]);
    }
    for (unsigned int t = 0; t < mapped_observations[0].size(); t++) {
        DdNode *temp_omega = Cudd_addApply(manager, Cudd_addPlus, omega[mapped_observations[0][t]], gamma[t]);
        Cudd_Ref(temp_omega);
        Cudd_RecursiveDeref(manager, omega[mapped_observations[0][t]]);
        omega[mapped_observations[0][t]] = temp_omega;
    }
    for (unsigned int l = 0; l < labels.size(); l++) {
        DdNode *temp_omega = Cudd_addApply(manager, Cudd_addDivide, omega[l], temporary_gamma_sum);
        Cudd_Ref(temp_omega);
        Cudd_RecursiveDeref(manager, omega[l]);
        omega[l] = temp_omega;
    }

    // update initial
    Cudd_RecursiveDeref(manager, pi);
    pi = gamma[0];
    Cudd_Ref(pi);
    Cudd_RecursiveDeref(manager, temporary_gamma_sum);
    Cudd_RecursiveDeref(manager, temporary_xi_sum);
}


void cupaal::MarkovModel::update_model_parameters_multiple_observations(const std::vector<DdNode **> &gammas,
                                                                        const std::vector<DdNode **> &xis,
                                                                        const std::map<std::vector<int>, int> &
                                                                        observation_map) {
    std::vector<DdNode *> counts;
    for (const auto &amount: observation_map | std::views::values) {
        DdNode *count = Cudd_addConst(manager, amount);
        Cudd_Ref(count);
        counts.push_back(count);
    }

    // mutiply each gamma and xi value with the number of counts for that observation
    for (unsigned int o = 0; o < observation_map.size(); o++) {
        for (unsigned int t = 0; t < mapped_observations[0].size() - 1; t++) {
            DdNode *temp_gamma = Cudd_addApply(manager, Cudd_addTimes, gammas[o][t], counts[o]);
            Cudd_Ref(temp_gamma);
            Cudd_RecursiveDeref(manager, gammas[o][t]);
            gammas[o][t] = temp_gamma;

            DdNode *temp_xi = Cudd_addApply(manager, Cudd_addTimes, xis[o][t], counts[o]);
            Cudd_Ref(temp_xi);
            Cudd_RecursiveDeref(manager, xis[o][t]);
            xis[o][t] = temp_xi;
        }
    }

    for (unsigned int o = 0; o < observation_map.size(); o++) {
        DdNode *temp_gamma = Cudd_addApply(manager, Cudd_addTimes, gammas[o][mapped_observations[0].size() - 1],
                                           counts[o]);
        Cudd_Ref(temp_gamma);
        Cudd_RecursiveDeref(manager, gammas[o][mapped_observations[0].size() - 1]);
        gammas[o][mapped_observations[0].size() - 1] = temp_gamma;
    }

    DdNode **gamma = gammas[0];
    DdNode **xi = xis[0];
    DdNode *number_of_observations = Cudd_addConst(manager, static_cast<double>(mapped_observations.size()));
    Cudd_Ref(number_of_observations);

    // precaculate common values: sum the gamma and xi vectors over each observation
    for (unsigned int o = 1; o < observation_map.size(); o++) {
        for (unsigned int t = 0; t < mapped_observations[0].size() - 1; t++) {
            DdNode *temp_gamma = Cudd_addApply(manager, Cudd_addPlus, gamma[t], gammas[o][t]);
            Cudd_Ref(temp_gamma);
            Cudd_RecursiveDeref(manager, gamma[t]);
            gamma[t] = temp_gamma;

            DdNode *temp_xi = Cudd_addApply(manager, Cudd_addPlus, xi[t], xis[o][t]);
            Cudd_Ref(temp_xi);
            Cudd_RecursiveDeref(manager, xi[t]);
            xi[t] = temp_xi;
        }
    }

    // precalculate common values: sum the vector values
    DdNode *temporary_gamma_sum = Cudd_ReadZero(manager);
    Cudd_Ref(temporary_gamma_sum);
    DdNode *temporary_xi_sum = Cudd_ReadZero(manager);
    Cudd_Ref(temporary_xi_sum);
    for (unsigned int t = 0; t < mapped_observations[0].size() - 1; t++) {
        DdNode *temporary_gamma_sum2 = Cudd_addApply(manager, Cudd_addPlus, temporary_gamma_sum, gamma[t]);
        Cudd_Ref(temporary_gamma_sum2);
        Cudd_RecursiveDeref(manager, temporary_gamma_sum);
        temporary_gamma_sum = temporary_gamma_sum2;
        DdNode *temporary_xi_sum2 = Cudd_addApply(manager, Cudd_addPlus, temporary_xi_sum, xi[t]);
        Cudd_Ref(temporary_xi_sum2);
        Cudd_RecursiveDeref(manager, temporary_xi_sum);
        temporary_xi_sum = temporary_xi_sum2;
    }

    // update pi (initial)
    Cudd_RecursiveDeref(manager, pi);
    pi = Cudd_addApply(manager, Cudd_addDivide, gamma[0], number_of_observations);
    Cudd_Ref(pi);
    // update tau (transitions)
    Cudd_RecursiveDeref(manager, tau);
    tau = Cudd_addApply(manager, Cudd_addDivide, temporary_xi_sum, temporary_gamma_sum);
    Cudd_Ref(tau);
    // update omega (labelling)
    DdNode *temporary_gamma_sum2 = Cudd_addApply(manager, Cudd_addPlus, temporary_gamma_sum,
                                                 gamma[mapped_observations[0].size() - 1]);
    Cudd_Ref(temporary_gamma_sum2);
    Cudd_RecursiveDeref(manager, temporary_gamma_sum);
    temporary_gamma_sum = temporary_gamma_sum2;

    for (unsigned int l = 0; l < labels.size(); l++) {
        Cudd_RecursiveDeref(manager, omega[l]);
        omega[l] = Cudd_ReadZero(manager);
        Cudd_Ref(omega[l]);
    }

    for (unsigned int o = 0; o < observation_map.size(); o++) {
        for (unsigned int t = 0; t < mapped_observations[0].size(); t++) {
            DdNode *temp_omega = Cudd_addApply(manager, Cudd_addPlus, omega[mapped_observations[o][t]], gamma[t]);
            Cudd_Ref(temp_omega);
            Cudd_RecursiveDeref(manager, omega[mapped_observations[o][t]]);
            omega[mapped_observations[o][t]] = temp_omega;
        }
    }

    for (unsigned int l = 0; l < labels.size(); l++) {
        DdNode *scaling_factor = Cudd_addConst(manager, static_cast<double>(observation_map.size()));
        Cudd_Ref(scaling_factor);

        DdNode *scaled_gamma_sum = Cudd_addApply(manager, Cudd_addTimes, temporary_gamma_sum, scaling_factor);
        Cudd_Ref(scaled_gamma_sum);

        DdNode *temp_omega = Cudd_addApply(manager, Cudd_addDivide, omega[l], scaled_gamma_sum);
        Cudd_Ref(temp_omega);
        Cudd_RecursiveDeref(manager, scaling_factor);
        Cudd_RecursiveDeref(manager, scaled_gamma_sum);
        Cudd_RecursiveDeref(manager, omega[l]);
        omega[l] = temp_omega;
    }

    for (const auto count: counts) {
        Cudd_RecursiveDeref(manager, count);
    }
    Cudd_RecursiveDeref(manager, temporary_gamma_sum);
    Cudd_RecursiveDeref(manager, temporary_xi_sum);
    Cudd_RecursiveDeref(manager, number_of_observations);
}

void cupaal::MarkovModel::baum_welch(const unsigned int max_iterations,
                                     const double epsilon,
                                     const std::chrono::seconds time) {
    using namespace std::chrono_literals;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time);
    unsigned int current_iteration = 1;
    double prev_log_likelihood = -std::numeric_limits<double>::infinity();
    double log_likelihood = 0.0;

    while (microseconds > 0ms &&
           current_iteration <= max_iterations &&
           std::abs(log_likelihood - prev_log_likelihood) >= epsilon) {
        auto iteration_start = std::chrono::system_clock::now();
        iterationReport report{};
        prev_log_likelihood = log_likelihood;
        log_likelihood = 0;
        const auto alpha = calculate_alpha(mapped_observations[0]);
        const auto beta = calculate_beta(mapped_observations[0]);
        const auto gamma = calculate_gamma(alpha, beta);
        const auto xi = calculate_xi(alpha, beta, mapped_observations[0]);
        log_likelihood += calculate_log_likelihood(alpha);
        update_model_parameters(gamma, xi);

        for (unsigned int t = 0; t < mapped_observations[0].size() - 1; t++) {
            Cudd_RecursiveDeref(manager, alpha[t]);
            Cudd_RecursiveDeref(manager, beta[t]);
            Cudd_RecursiveDeref(manager, gamma[t]);
            Cudd_RecursiveDeref(manager, xi[t]);
        }
        Cudd_RecursiveDeref(manager, alpha[observations[0].size() - 1]);
        Cudd_RecursiveDeref(manager, beta[observations[0].size() - 1]);
        Cudd_RecursiveDeref(manager, gamma[observations[0].size() - 1]);
        free(alpha);
        free(beta);
        free(gamma);
        free(xi);

        report.iteration_number = current_iteration;
        report.log_likelihood = log_likelihood;
        report.running_time_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - iteration_start);
        iteration_reports.push_back(report);
        std::cout << "Iteration: " << report.iteration_number
                << ", Time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    report.running_time_microseconds).
                count()
                << ", Log likelihood: " << report.log_likelihood << std::endl;
        microseconds -= report.running_time_microseconds;
        current_iteration++;
    }
}

void cupaal::MarkovModel::baum_welch_multiple_observations(const unsigned int max_iterations,
                                                           const double epsilon,
                                                           const std::chrono::seconds time) {
    using namespace std::chrono_literals;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time);
    unsigned int current_iteration = 1;
    double prev_log_likelihood = -std::numeric_limits<double>::infinity();
    double log_likelihood = 0.0;
    
    std::map<std::vector<int>, int> observation_counts;
    for (const auto &observation: mapped_observations) {
        observation_counts[observation]++;
    }
    
    while (microseconds > 0ms &&
        current_iteration <= max_iterations &&
        std::abs(log_likelihood - prev_log_likelihood) >= epsilon) {
            auto iteration_start = std::chrono::system_clock::now();
            iterationReport report{};
            prev_log_likelihood = log_likelihood;
            log_likelihood = 0;
            std::vector<DdNode **> gammas;
            std::vector<DdNode **> xis;
            
        for (const auto &[observation, count]: observation_counts) {
            std::cout << "Calculating alpha for iteration: " << current_iteration << std::endl;
            const auto alpha = calculate_alpha(observation);
            const auto beta = calculate_beta(observation);
            const auto gamma = calculate_gamma(alpha, beta);
            const auto xi = calculate_xi(alpha, beta, observation);

            gammas.push_back(gamma);
            xis.push_back(xi);

            log_likelihood += calculate_log_likelihood(alpha) * count;

            for (unsigned int t = 0; t < observation.size(); t++) {
                Cudd_RecursiveDeref(manager, alpha[t]);
                Cudd_RecursiveDeref(manager, beta[t]);
            }
            free(alpha);
            free(beta);
        }

        update_model_parameters_multiple_observations(gammas, xis, observation_counts);

        for (unsigned int o = 0; o < gammas.size(); o++) {
            for (unsigned int t = 0; t < observations[0].size() - 1; t++) {
                Cudd_RecursiveDeref(manager, gammas[o][t]);
                Cudd_RecursiveDeref(manager, xis[o][t]);
            }
            Cudd_RecursiveDeref(manager, gammas[o][observations[0].size() - 1]);
            free(gammas[o]);
            free(xis[o]);
        }
        report.iteration_number = current_iteration;
        report.log_likelihood = log_likelihood;
        report.running_time_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - iteration_start);
        iteration_reports.push_back(report);
        std::cout << "Iteration: " << report.iteration_number
                << ", Time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    report.running_time_microseconds).
                count()
                << ", Log likelihood: " << report.log_likelihood << std::endl;
        microseconds -= report.running_time_microseconds;
        current_iteration++;
    }
}

void cupaal::MarkovModel::initialize_adds(){
    number_of_states = static_cast<int>(states.size());
    number_of_labels = static_cast<int>(labels.size());
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
        COL_VAR_INDEX_MULTIPLIER);

    Sudd_addRead(
        initial_distribution.data(),
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

void cupaal::MarkovModel::initialize_from_file(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return;
    }

    std::string currently_parsing;
    std::string line;
    while (std::getline(file, line)) {
        std::string word;
        std::stringstream line_stream(line);
        while (std::getline(line_stream, word, ' ')) {
            if (MODEL_ELEMENTS.contains(word)) {
                currently_parsing = word;
                continue;
            }
            if (currently_parsing == "states") states.push_back(word);
            if (currently_parsing == "labels") labels.push_back(word);
            if (currently_parsing == "initial") initial_distribution.push_back(stod(word));
            if (currently_parsing == "transitions") transitions.push_back(stod(word));
            if (currently_parsing == "emissions") emissions.push_back(stod(word));
        }
    }

    initialize_adds();
}

void cupaal::MarkovModel::add_observation(const std::vector<std::string> &observation) {
    std::vector<int> mapped_observation;
    observations.push_back(observation);
    for (const std::string &s: observation) {
        mapped_observation.push_back(label_index_map[s]);
    }
    mapped_observations.push_back(mapped_observation);
}


void cupaal::MarkovModel::add_observation_from_file(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> observation;
        std::string word;
        std::stringstream line_stream(line);
        while (std::getline(line_stream, word, ' ')) {
            observation.push_back(word);
        }
        add_observation(observation);
    }
}

void cupaal::MarkovModel::export_to_file(const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;

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
        int assignment[2 * n_vars];
        for (int j = 0; j < n_vars; j++) {
            assignment[j * 2] = (i >> (n_vars - j - 1)) & 1;
        }
        file << Cudd_Eval(manager, pi, assignment) << " ";
    }
    file << std::endl;

    file << "transitions\n";
    for (int row = 0; row < number_of_states; row++) {
        for (int col = 0; col < number_of_states; col++) {
            int assignment[2 * n_vars];
            for (int i = 0; i < n_vars; i++) {
                assignment[2 * i] = (row >> (n_vars - i - 1)) & 1;
                assignment[2 * i + 1] = (col >> (n_vars - i - 1)) & 1;
            }
            file << Cudd_Eval(manager, tau, assignment) << " ";
        }
        file << std::endl;
    }

    file << "emissions\n";
    for (int row = 0; row < number_of_labels; row++) {
        for (int col = 0; col < number_of_states; col++) {
            int assignment[2 * n_vars];
            for (int j = 0; j < n_vars; j++) {
                assignment[2 * j] = (col >> (n_vars - j - 1)) & 1;
            }
            file << Cudd_Eval(manager, omega[row], assignment) << " ";
        }
        file << std::endl;
    }
    file.close();
}

void cupaal::MarkovModel::save_experiment_to_csv(const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    file << "iteration,milliseconds,log_likelihood\n";
    for (auto [iteration_number, running_time_microseconds, log_likelihood]: iteration_reports) {
        file << iteration_number << ","
                << std::chrono::duration_cast<std::chrono::milliseconds>(running_time_microseconds).count() << ","
                << log_likelihood << std::endl;
    }
    file.close();
}

void cupaal::MarkovModel::clean_up_cudd() const {
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

double cupaal::MarkovModel::calculate_log_likelihood(DdNode **alpha) const {
    const auto log_values = Cudd_addMonadicApply(manager, addLog, alpha[observations[0].size() - 1]);
    Cudd_Ref(log_values);
    const auto log_likelihood_add = Cudd_addExistAbstract(manager, log_values, row_cube);
    Cudd_Ref(log_likelihood_add);
    const double log_likelihood = Cudd_V(log_likelihood_add);
    Cudd_RecursiveDeref(manager, log_values);
    Cudd_RecursiveDeref(manager, log_likelihood_add);
    return log_likelihood;
}
