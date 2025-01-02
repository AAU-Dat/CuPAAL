#include <iostream>
#include <storm/utility/initialize.h>
#include <cuddInt.h>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"

#define ROW_VAR_INDEX_OFFSET 0
#define ROW_VAR_INDEX_MULTIPLIER 2
#define COL_VAR_INDEX_OFFSET 1
#define COL_VAR_INDEX_MULTIPLIER 2

using state = int;
using probability = CUDD_VALUE_TYPE;
using label = std::string;

class CupaalMarkovModel_Matrix {
public:
    std::set<state> states;
    std::vector<label> labels;
    std::map<label, int> label_index_map;
    std::vector<std::vector<label> > observations;
    probability *labelling_matrix; // omega
    probability *transition_matrix; // P
    probability *initial_distribution_vector; // pi
    int number_of_observations;
    int number_of_states;
};

int kronecker_product() {
    // const auto kronecker_product = static_cast<DdNode *>(cupaal::safe_malloc(sizeof(DdNode *), 4));
    DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    int n_vars = ceil(log2(2));
    int dump_n_rows = 2;
    int dump_n_cols = 2;
    int n_row_vars = 0;
    int n_col_vars = 0;
    auto row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    double A_mat[4] = {1, 0, 0, 1};
    double B_mat[4] = {1, 2, 3, 4};

    DdNode *A_as_add;
    DdNode *B_as_add;

    cupaal::Sudd_addRead(
        A_mat,
        2,
        2,
        gbm,
        &A_as_add,
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

    cupaal::Sudd_addRead(
        B_mat,
        2,
        2,
        gbm,
        &B_as_add,
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
    //

    auto row_vars2 = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), 4));

    std::cout << row_vars2[0]->index << std::endl;
    auto x = Cudd_addMatrixMultiply(gbm, A_as_add, B_as_add, row_vars2, 2);
    // auto y = Cudd_addTimesPlus(gbm, A_as_add, B_as_add, row_vars, 1);

    cupaal::write_dd_to_dot(gbm, A_as_add, "/home/runge/CuPAAL/A_matrix.dot");
    cupaal::write_dd_to_dot(gbm, B_as_add, "/home/runge/CuPAAL/B_matrix.dot");
    cupaal::write_dd_to_dot(gbm, x, "/home/runge/CuPAAL/X_matix_mult.dot");



    return 1;
}

void experimental_testing() {
    // auto model = cupaal::parseAndBuildPrism("/home/runge/CuPAAL/models/polling.prism");
    // auto sparsemodel = cupaal::parseAndBuildPrismSparse("/home/runge/CuPAAL/models/polling.prism");
    // std::cout << sparsemodel->getNumberOfStates() << std::endl;
    // std::cout << sparsemodel->getType() << std::endl;
    // storm::storage::SparseMatrix<double>::index_type var2 = 0;
    // while (var2 < sparsemodel->getNumberOfStates()) {
    //     for (const auto &label: sparsemodel->getStateLabeling().getLabelsOfState(var2)) {
    //         std::cout << var2 << "<< index, label >> " << label << std::endl;
    //         std::cout << "index actions: " << sparsemodel->getTransitionMatrix().getRowGroupSize(var2) << std::endl;
    //     }
    //     var2 = var2 + 1;
    // }
    // for (const auto &basic_string: sparsemodel->getStateLabeling().getLabels()) {
    //     std::cout << basic_string << std::endl;
    //     std::cout << sparsemodel->getStates(basic_string) << std::endl;
    // }
    // // std::cout << sparsemodel->getTransitionMatrix().getRow(3).begin()->getValue();
    // for (auto element: sparsemodel->getTransitionMatrix().getRow(27)) {
    //     std::cout << element.getColumn() << " << col, val >> ";
    //     std::cout << element.getValue() << std::endl;
    // }
}

void initialize_model_parameters(const CupaalMarkovModel_Matrix &model, int seed = 0) {
    std::vector<double> initial_distribution = cupaal::generate_stochastic_probabilities(model.states.size(), seed);

    for (state s = 0; s < model.states.size(); s++) {
        model.initial_distribution_vector[s] = initial_distribution[s];
    }

    for (state s = 0; s < model.states.size(); s++) {
        std::vector<double> trasition_probabilities = cupaal::generate_stochastic_probabilities(model.states.size(), seed);
        for (state s_prime = 0; s_prime < model.states.size(); s_prime++) {
            model.transition_matrix[s * model.states.size() + s_prime] = trasition_probabilities[s_prime];
        }
    }

    for (int l = 0; l < model.labels.size(); l++) {
        std::vector<double> labelling_probabilities = cupaal::generate_stochastic_probabilities(model.states.size(), seed);
        for (int s = 0; s < model.states.size(); s++) {
            model.labelling_matrix[s * model.labels.size() + l] = labelling_probabilities[s];
        }
    }
}

void forwards_matrices(const CupaalMarkovModel_Matrix &model) {
    // Allocate alpha
    std::vector<double> alpha((model.number_of_states + 1) * model.number_of_states, 0.0);

    // Base case: t = 0
    for (int t = 0; t < model.number_of_states; t++) {
        alpha[t] = model.initial_distribution_vector[t];
    }

    // Case: 0 < t <= n_obs
    for (int t = 0; t < model.number_of_states; ++t) {
        for (int s = 0; s < model.number_of_states; ++s) {
            double temp = 0.0;
            for (int ss = 0; ss < model.number_of_states; ++ss) {
                temp += model.transition_matrix[ss * model.number_of_states + s] * model.labelling_matrix[t * model.number_of_states + ss] * alpha[t * model.number_of_states + ss];
            }
            alpha[(t + 1) * model.number_of_states + s] = temp;
        }
    }

    std::cout << "Alpha Matrix:" << std::endl;
    for (int t = 0; t < model.number_of_states; ++t) { // Rows: time steps
        for (int s = 0; s < model.number_of_states; ++s) { // Columns: states
            std::cout << alpha[t * model.number_of_states + s] << " ";
        }
        std::cout << std::endl;
    }
}

void backwards_matrices(const CupaalMarkovModel_Matrix &model) {
    // Allocate beta
    std::vector<double> beta((model.number_of_states + 1) * model.number_of_states, 0.0);

    for (int t = 0; t < model.number_of_states; t++) {
        beta[model.number_of_observations * model.number_of_states + t] = 1;
    }

    for (int t = 0; t < model.number_of_states; t++) {
        for (int s = 0; s < model.number_of_states; ++s) {
            double temp = 0.0;
            for (int ss = 0; ss < model.number_of_states; ++ss) {
                temp += beta[t*model.number_of_states + ss] * model.transition_matrix[s * model.number_of_states + ss];
            }
            beta[(t-1) * model.number_of_states + s] = model.labelling_matrix[(t-1)*model.number_of_states + s] * temp;
        }
    }
}

void print_model(const CupaalMarkovModel_Matrix &model) {
    std::cout << "Model Details:" << std::endl;

    // Print number of states
    std::cout << "Number of States: " << model.number_of_states << std::endl;

    // Print Initial Distribution Vector
    std::cout << "Initial Distribution Vector:" << std::endl;
    for (int i = 0; i < model.number_of_states; i++) {
        std::cout << model.initial_distribution_vector[i] << " ";
    }
    std::cout << std::endl;

    // Print Transition Matrix
    std::cout << "Transition Matrix:" << std::endl;
    for (int i = 0; i < model.number_of_states; ++i) {
        for (int j = 0; j < model.number_of_states; ++j) {
            std::cout << model.transition_matrix[i * model.number_of_states + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print Labelling Matrix
    std::cout << "Labelling Matrix:" << std::endl;
    for (int l = 0; l < model.labels.size(); ++l) { // Time steps or rows
        for (int s = 0; s < model.number_of_states; ++s) { // States or columns
            std::cout << model.labelling_matrix[s * model.labels.size() + l] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;
}

void baum_welch(const CupaalMarkovModel_Matrix &model, int seed=0) {
    // Initialize pi, P, and omega (based on observations)
    initialize_model_parameters(model, seed);
    print_model(model);
    forwards_matrices(model);
}

int main(int argc, char *argv[]) {
    storm::utility::setUp();
    storm::settings::initializeAll("CuPAAL", "CuPAAL");
    DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

    CupaalMarkovModel_Matrix model;
    model.states = {1, 2};
    model.labels = {"Hungry", "Full"};
    model.observations = {{"Hungry", "Full", "Hungry", "Hungry", "Full", "Full"}};
    model.number_of_states = model.states.size();
    model.number_of_observations = model.observations.size();
    model.transition_matrix = new probability[100];
    model.labelling_matrix = new probability[100];
    model.initial_distribution_vector = new probability[100];

    baum_welch(model, 10);

    // states = {"a", "b"} = {1, 2}
    // labels = {"hungry", "full"}
    // l(s) = if s == "a" return "hungry"
    //        if s == "b" return "full"
    // l = ["hungry", "full"]
    // l_matrix = [1, 0]
    //            [0, 1]
    double l_matrix[8] = {
        1, 0,
        1, 0,
        0, 1,
        1, 0
    }; // rows = length of observation sequence, columns = number of states

    // matrix = [0.3, 0.7]
    //          [0.1, 0.9]
    double transition_matrix[4] = {0.3, 0.7, 0.1, 0.9};

    // initial states = [1, 0]
    double initial_states[2] = {1, 0};

    // observation/trace
    // o = ["hungry", "hungry", "full", "hungry"]

    // transition_matrix_estimate = [0.5, 0.5]
    //                              [1.0, 0.0]

    // emission_matrix_estimate = [1, 0]
    //                            [0, 1]

    // initial_state_estimate = [1, 0]

    // int dump_n_rows = 2;
    // int dump_n_cols = 2;
    // int n_row_vars = 0;
    // int n_col_vars = 0;
    //
    // int n_vars = ceil(log2(2));
    // auto row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    // auto col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    // auto comp_row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    // auto comp_col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    //
    // DdNode *transition_matrix_as_add;
    // DdNode *initial_states_matrix_as_add;
    // auto labelling_matrix_as_add = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), 4));
    //
    // auto add_result = cupaal::Sudd_addRead(
    //     transition_matrix,
    //     2,
    //     2,
    //     gbm,
    //     &transition_matrix_as_add,
    //     &row_vars,
    //     &col_vars,
    //     &comp_row_vars,
    //     &comp_col_vars,
    //     &n_row_vars,
    //     &n_col_vars,
    //     &dump_n_rows,
    //     &dump_n_cols,
    //     ROW_VAR_INDEX_OFFSET,
    //     ROW_VAR_INDEX_MULTIPLIER,
    //     COL_VAR_INDEX_OFFSET,
    //     COL_VAR_INDEX_MULTIPLIER
    // );
    // std::cout << add_result << std::endl;
    //
    //
    // cupaal::Sudd_addRead(
    //     initial_states,
    //     2,
    //     1,
    //     gbm,
    //     &initial_states_matrix_as_add,
    //     &row_vars,
    //     &col_vars,
    //     &comp_row_vars,
    //     &comp_col_vars,
    //     &n_row_vars,
    //     &n_col_vars,
    //     &dump_n_rows,
    //     &dump_n_cols,
    //     ROW_VAR_INDEX_OFFSET,
    //     ROW_VAR_INDEX_MULTIPLIER,
    //     COL_VAR_INDEX_OFFSET,
    //     COL_VAR_INDEX_MULTIPLIER
    // );
    //
    // for (int t = 0; t < 4; t++) {
    //     cupaal::Sudd_addRead(
    //         &l_matrix[t * 2],
    //         2,
    //         1,
    //         gbm,
    //         &labelling_matrix_as_add[t],
    //         &row_vars,
    //         &col_vars,
    //         &comp_row_vars,
    //         &comp_col_vars,
    //         &n_row_vars,
    //         &n_col_vars,
    //         &dump_n_rows,
    //         &dump_n_cols,
    //         ROW_VAR_INDEX_OFFSET,
    //         ROW_VAR_INDEX_MULTIPLIER,
    //         COL_VAR_INDEX_OFFSET,
    //         COL_VAR_INDEX_MULTIPLIER
    //     );
    // }
    //
    // DdNode **alpha_result = cupaal::forward(
    //     gbm,
    //     labelling_matrix_as_add,
    //     transition_matrix_as_add,
    //     initial_states_matrix_as_add,
    //     row_vars,
    //     col_vars,
    //     n_row_vars,
    //     4
    // );
    //
    // DdNode **beta_result = cupaal::backward(
    //     gbm,
    //     labelling_matrix_as_add,
    //     transition_matrix_as_add,
    //     row_vars,
    //     col_vars,
    //     n_vars,
    //     4
    // );
    //
    // cupaal::write_dd_to_dot(gbm, transition_matrix_as_add, "/home/runge/CuPAAL/transition_matrix.dot");
    // cupaal::write_dd_to_dot(gbm, initial_states_matrix_as_add, "/home/runge/CuPAAL/initial_states.dot");
    // cupaal::write_dd_to_dot(gbm, labelling_matrix_as_add[3], "/home/runge/CuPAAL/omega4.dot");
    // cupaal::write_dd_to_dot(gbm, alpha_result[3], "/home/runge/CuPAAL/alpha4.dot");
    // cupaal::write_dd_to_dot(gbm, beta_result[0], "/home/runge/CuPAAL/beta1.dot");
    //
    // auto res = kronecker_product();
    // std::cout << res << std::endl;
    Cudd_Quit(gbm);
    exit(EXIT_SUCCESS);
}

class CupaalMarkovModel {
};

// States (S)
// Labels (L)
// Labelling function / emissions (l)
// Transitions (Rate)
// Initial State(s) (Pi)
//

// How do we do Baum-Welch?
// Using OBSERVATIONS/TRACES
// Forward-Backward
// Update-step
// Epsilon (convergence criterion threshold)
//

// Kronecker product:
// A = [1, 0]
//     [0, 2]
// B = [1, 2]
//     [3, 4]
// kronecker(A, B) = [1, 2, 0, 0]
//                 = [3, 4, 0, 0]
//                 = [0, 0, 2, 4]
//                 = [0, 0, 6, 8]


// extern DdNode * Cudd_addMatrixMultiply(DdManager *dd, DdNode *A, DdNode *B, DdNode **z, int nz);

