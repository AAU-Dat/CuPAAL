#include <filesystem>

#include "baum_welch.h"
#include "cudd_extensions.h"
#include "helpers.h"

#define ROW_VAR_INDEX_OFFSET 0
#define ROW_VAR_INDEX_MULTIPLIER 2
#define COL_VAR_INDEX_OFFSET 1
#define COL_VAR_INDEX_MULTIPLIER 2

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

    Sudd_addRead(
        labelling_matrix,
        states.size(),
        labels.size(),
        manager,
        labelling_add,
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

    for (int t = 1; t < observations[0].size(); t++) {
        DdNode *alpha_temp = Cudd_addMatrixMultiply(manager, transition_add, alpha[t - 1], row_vars, n_vars);
        alpha_temp = Cudd_addSwapVariables(manager, alpha_temp, row_vars, col_vars, n_vars);
        Cudd_Ref(alpha_temp);

        alpha[t] = Cudd_addApply(manager, Cudd_addTimes, labelling_add[label_index_map.at(observations[0][t])],
                                 alpha_temp);
        Cudd_Ref(alpha[t]);
        Cudd_RecursiveDeref(manager, alpha_temp);
    }
    return alpha;
}

DdNode **cupaal::MarkovModel_ADD::backward_add() const {
    const auto beta = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    beta[observations[0].size() - 1] = Cudd_ReadOne(manager);

    DdNode *temporary_transition_add = Cudd_addSwapVariables(manager, transition_add, col_vars, row_vars, n_vars);

    for (int t = observations[0].size() - 2; t >= 0; t--) {
        DdNode *beta_temp = Cudd_addApply(manager, Cudd_addTimes, beta[t + 1],
                                          labelling_add[label_index_map.at(observations[0][t + 1])]);
        Cudd_Ref(beta_temp);
        DdNode *beta_value = Cudd_addMatrixMultiply(manager, temporary_transition_add, beta_temp, row_vars, n_vars);
        beta_value = Cudd_addSwapVariables(manager, beta_value, row_vars, col_vars, n_vars);
        beta[t] = beta_value;
        Cudd_Ref(beta[t]);
        Cudd_RecursiveDeref(manager, beta_temp);
    }
    return beta;
}

DdNode **cupaal::MarkovModel_ADD::gamma_add(DdNode **alpha, DdNode **beta) const {
    const auto gamma = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    DdNode *scalar = Cudd_addExistAbstract(manager, alpha[observations[0].size() - 1], *row_vars);
    Cudd_Ref(scalar);

    for (unsigned long t = 0; t < observations[0].size(); t++) {
        DdNode *gamma_temp = Cudd_addApply(manager, Cudd_addTimes, alpha[t], beta[t]);
        gamma[t] = Cudd_addApply(manager, Cudd_addDivide, gamma_temp, scalar);
        Cudd_Ref(gamma[t]);
    }
    Cudd_RecursiveDeref(manager, scalar);
    return gamma;
}

DdNode **cupaal::MarkovModel_ADD::xi_add(DdNode **alpha, DdNode **beta) const {
    const auto xi = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), observations[0].size()));
    DdNode *scalar = Cudd_addExistAbstract(manager, alpha[observations[0].size() - 1], *row_vars);
    DdNode *temporary_transition_add = Cudd_addSwapVariables(manager, transition_add, col_vars, row_vars, n_vars);
    DdNode *scaled_transition_add = Cudd_addApply(manager, Cudd_addDivide, temporary_transition_add, scalar);
    Cudd_Ref(scaled_transition_add);

    for (unsigned long t = 0; t < observations[0].size() - 1; t++) {
        DdNode *beta_temp = Cudd_addApply(manager, Cudd_addTimes, beta[t + 1],
                                          labelling_add[label_index_map.at(observations[0][t + 1])]);
        Cudd_Ref(beta_temp);
        auto alpha_temp = Cudd_addSwapVariables(manager, alpha[t], row_vars, col_vars, n_vars);
        Cudd_Ref(alpha_temp);
        // auto kronecker_product = Cudd_addMatrixMultiply(manager, alpha_temp, beta_temp, row_vars, 0);
        auto kronecker_product = Cudd_addApply(manager, Cudd_addTimes, alpha_temp, beta_temp);
        Cudd_Ref(kronecker_product);
        auto xi_value = Cudd_addApply(manager, Cudd_addTimes, scaled_transition_add, kronecker_product);
        xi[t] = Cudd_addSwapVariables(manager, xi_value, row_vars, col_vars, n_vars);
        Cudd_Ref(xi[t]);
        Cudd_RecursiveDeref(manager, beta_temp);
        Cudd_RecursiveDeref(manager, alpha_temp);
        Cudd_RecursiveDeref(manager, kronecker_product);
    }
    Cudd_RecursiveDeref(manager, scaled_transition_add);
    return xi;
}

void cupaal::MarkovModel_ADD::update_model_parameters_add(DdNode **gamma, DdNode **xi) {
    DdNode *temporary_gamma_sum = gamma[0];
    DdNode *temporary_xi_sum = xi[0];
    for (int t = 1; t < observations[0].size() - 1; t++) {
        temporary_gamma_sum = Cudd_addApply(manager, Cudd_addPlus, temporary_gamma_sum, gamma[t]);
        temporary_xi_sum = Cudd_addApply(manager, Cudd_addPlus, temporary_xi_sum, xi[t]);
    }
    // Update transitions
    Cudd_PrintDebug(manager, temporary_gamma_sum, 4, 2);
    Cudd_PrintDebug(manager, temporary_xi_sum, 4, 2);

    transition_add = Cudd_addApply(manager, Cudd_addDivide, temporary_xi_sum, temporary_gamma_sum);
    Cudd_PrintDebug(manager, transition_add, 4, 2);

    // Update labelling
    for (int l = 0; l < labels.size(); l++) {
        labelling_add[l] = Cudd_Zero(manager);
    }
    for (int t = 0; t < observations[0].size(); t++) {
        labelling_add[label_index_map.at(observations[0][t])] = Cudd_addApply(
            manager, Cudd_addPlus, labelling_add[label_index_map.at(observations[0][t])], gamma[t]);
    }
    for (int l = 0; l < labels.size(); l++) {
        std::cout << "LABELS AS THEY ARE UPDATED!!!!" << std::endl;
        Cudd_PrintDebug(manager, labelling_add[l], 4, 2);
        labelling_add[l] = Cudd_addApply(manager, Cudd_addDivide, labelling_add[l], temporary_gamma_sum);
        Cudd_PrintDebug(manager, labelling_add[l], 4, 2);
    }

    // Update initial distribution
    initial_distribution_add = gamma[0];
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


DdNode **cupaal::forward(DdManager *manager, DdNode **omega, DdNode *P, DdNode *pi, DdNode **row_vars,
                         DdNode **column_vars, const int n_vars, const int n_obs) {
    const auto alpha = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_obs + 1));
    alpha[0] = pi;
    for (int t = 1; t <= n_obs; t++) {
        DdNode *alpha_temp_0 = Cudd_addApply(manager, Cudd_addTimes, omega[t - 1], alpha[t - 1]);
        Cudd_Ref(alpha_temp_0);
        DdNode *alpha_temp_1 = Cudd_addMatrixMultiply(manager, P, alpha_temp_0, row_vars, n_vars);
        Cudd_Ref(alpha_temp_1);
        alpha[t] = Cudd_addSwapVariables(manager, alpha_temp_1, column_vars, row_vars, n_vars);
        Cudd_Ref(alpha[t]);
        Cudd_RecursiveDeref(manager, alpha_temp_0);
        Cudd_RecursiveDeref(manager, alpha_temp_1);
    }
    return alpha;
}

DdNode **cupaal::backward(DdManager *manager, DdNode **omega, DdNode *P, DdNode **row_vars,
                          DdNode **column_vars, const int n_vars, const int n_obs) {
    DdNode *temporary_P = Cudd_addSwapVariables(manager, P, column_vars, row_vars, n_vars);
    Cudd_Ref(temporary_P);
    const auto beta = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_obs + 1));
    beta[n_obs] = Cudd_ReadOne(manager);
    for (int t = n_obs - 1; 0 <= t; t--) {
        DdNode *beta_temp_0 = Cudd_addMatrixMultiply(manager, temporary_P, beta[t + 1], row_vars, n_vars);
        Cudd_Ref(beta_temp_0);
        DdNode *beta_temp_1 = Cudd_addSwapVariables(manager, beta_temp_0, column_vars, row_vars, n_vars);
        Cudd_Ref(beta_temp_1);
        beta[t] = Cudd_addApply(manager, Cudd_addTimes, omega[t], beta_temp_1);
        Cudd_Ref(beta[t]);
        Cudd_RecursiveDeref(manager, beta_temp_0);
        Cudd_RecursiveDeref(manager, beta_temp_1);
    }
    Cudd_RecursiveDeref(manager, temporary_P);
    return beta;
}
