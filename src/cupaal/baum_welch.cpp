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

    for (int s = 0; s < states.size(); s++) {
        initial_distribution_vector[s] = initial_distribution[s];
    }

    for (int s = 0; s < states.size(); s++) {
        std::vector<double> transition_probabilities = generate_stochastic_probabilities(
            states.size(), generator);
        for (int s_prime = 0; s_prime < states.size(); s_prime++) {
            transition_matrix[s * states.size() + s_prime] = transition_probabilities[s_prime];
        }
    }

    for (int s = 0; s < states.size(); s++) {
        std::vector<double> labelling_probabilities = generate_stochastic_probabilities(
            labels.size(), generator);
        for (int l = 0; l < labels.size(); l++) {
            labelling_matrix[s * labels.size() + l] = labelling_probabilities[l];
        }
    }
    if (!observations.empty() && labelling_matrix != nullptr) {
        calculate_omega();
    }

    int dump_n_rows = states.size();
    int dump_n_cols = states.size();
    int n_row_vars = 0;
    int n_col_vars = 0;
    int n_vars = ceil(log2(states.size()));
    auto row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode*), n_vars));
    auto col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode*), n_vars));
    auto comp_row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode*), n_vars));
    auto comp_col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode*), n_vars));

    Sudd_addRead(initial_distribution_vector, states.size(), states.size(), manager, &initial_distribution_add, &row_vars,
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

void cupaal::MarkovModel_ADD::print_model_parameters() const {
    std::cout << "Model Details:" << std::endl;
    std::cout << "Number of States: " << states.size() << std::endl;
}

void cupaal::MarkovModel_ADD::calculate_omega() {
    }

std::vector<probability> cupaal::forward_matrix(const MarkovModel_Matrix &model) {
    const unsigned long number_of_states = model.states.size();
    // Allocate alpha
    std::vector alpha((number_of_states + 1) * model.observations[0].size(), 0.0);

    // Base case: t = 0
    for (int t = 0; t < number_of_states; t++) {
        alpha[t] = model.initial_distribution_vector[t];
    }

    // Case: 0 < t <= n_obs
    for (int t = 0; t < model.observations[0].size(); t++) {
        for (int s = 0; s < number_of_states; s++) {
            double temp = 0.0;
            for (int ss = 0; ss < number_of_states; ss++) {
                temp += model.transition_matrix[ss * number_of_states + s] * model.omega_matrix[
                    t * number_of_states + ss] * alpha[t * number_of_states + ss];
            }
            alpha[(t + 1) * number_of_states + s] = temp;
        }
    }

    return alpha;
}

std::vector<probability> cupaal::backward_matrix(const MarkovModel_Matrix &model) {
    const unsigned long number_of_states = model.states.size();
    // Allocate beta + Base case: t = n_obs
    std::vector beta((number_of_states + 1) * model.observations[0].size(), 1.0);

    // Case: 0 <= t < n_obs
    for (unsigned long t = model.observations[0].size(); t > 0; t--) {
        for (int s = 0; s < number_of_states; s++) {
            double temp = 0.0;
            for (int ss = 0; ss < number_of_states; ss++) {
                temp += beta[t * number_of_states + ss] * model.transition_matrix[s * number_of_states + ss];
            }
            beta[(t - 1) * number_of_states + s] = model.omega_matrix[(t - 1) * number_of_states + s] * temp;
        }
    }

    return beta;
}

cupaal::MarkovModel_Matrix cupaal::baum_welch_matrix(const MarkovModel_Matrix &model) {
    const auto alpha = forward_matrix(model);
    const auto beta = backward_matrix(model);

    if (model.print_calculations) {
        std::cout << "Alpha Matrix:" << std::endl;
        for (int t = 0; t <= model.observations[0].size(); t++) {
            // Rows: time steps
            for (int s = 0; s < model.states.size(); s++) {
                // Columns: states
                std::cout << alpha[t * model.states.size() + s] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Beta Matrix:" << std::endl;
        for (int t = 0; t <= model.observations[0].size(); t++) {
            // Rows: time steps
            for (int s = 0; s < model.states.size(); s++) {
                // Columns: states
                std::cout << beta[t * model.states.size() + s] << " ";
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
