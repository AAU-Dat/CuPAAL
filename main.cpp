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

class CupaalMarkovModel {
public:
    std::set<state> states;
    std::vector<label> labels;
    std::vector<std::vector<label> > observations;
    std::map<state, std::map<label, probability> > labeling_function; // l
    std::map<state, std::map<state, probability> > transition_function; // tau
    std::map<state, probability> initial_distribution; // pi
};


class CupaalMarkovModel_Matrix {
public:
    std::set<state> states;
    std::vector<label> labels;
    std::map<label, int> label_index_map;
    std::vector<std::vector<label> > observations;
    probability *labelling_matrix; // omega
    probability *transition_matrix; // P
    probability *initial_distribution_vector; // pi
};


class CupaalMarkovModel_AlgebraicDecisionDiagram {
public:
    std::set<state> states;
    std::vector<label> labels;
    std::map<label, int> label_index_map;
    std::vector<std::vector<label> > observations;
    DdNode *labelling_ADD; // omega
    DdNode *transition_ADD; // P
    DdNode *initial_distribution_ADD; // pi
};

// Seed is by default set to 0 and this makes it completely random. A specific seed can be used to replicate results
// The size given is the number of random variables are needed.
// generate_probability_array normalizes values to two decimal points
std::vector<double> generate_probability_array(int size, int times_called=0, int seed=0) {
    if (size == 0) return {}; // Return an empty array for size 0

    std::uniform_real_distribution<> dis(0.01, 0.99);
    std::vector<double> probabilities(size);

    std::mt19937 gen;
    if (seed == 0) {
        std::random_device rd;
        gen.seed(rd());
        for (size_t i = 0; i < size; ++i) {
            probabilities[i] = dis(gen);
        }
    }
    else {
        //the geneation of a seed uses the inital seed and the times called in combination to give something that feels random even when giving similar seeds.
        std::mt19937 gen_with_seed(seed*(times_called+1));

        for (size_t i = 0; i < size; ++i) {
            probabilities[i] = dis(gen_with_seed);
        }
    }

    // Normalize values to sum to 1.0
    double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    for (double& p : probabilities) {
        p /= sum;
    }

    // Round each probability to two decimal places and adjust the sum to exactly 1.0
    for (double& p : probabilities) {
        p = std::round(p * 100.0) / 100.0; // Round to two decimal places
    }

    // Adjust to ensure the sum is exactly 1.0
    double roundedSum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    double diff = 1.0 - roundedSum;

    // Add or subtract the difference to the largest element to fix the sum
    if (std::abs(diff) > 1e-9) { // Avoid floating-point precision errors
        auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
        *maxIt += diff;
        *maxIt = std::round(*maxIt * 100.0) / 100.0; // Re-round to ensure two decimals
    }
    return probabilities;
}

void initialize_model_parameters(const CupaalMarkovModel_Matrix &model, int seed = 0) {
    int times_called_random_values = 0;
    std::vector<double> initial_distribution = generate_probability_array(model.states.size(), seed);
    times_called_random_values++;
    for (const state s: model.states) {
        model.initial_distribution_vector[s] = initial_distribution[s];
    }

    for (const state s: model.states) {
        std::vector<double> trasition_probabilities = generate_probability_array(model.states.size(), times_called_random_values, seed);
        times_called_random_values++;
        for (const state s_prime: model.states) {
            model.transition_matrix[s * model.states.size() + s_prime] = trasition_probabilities[s_prime];
        }
    }

    for (int s = 0; s < model.states.size(); s++) {
        std::vector<double> labelling_probabilities = generate_probability_array(model.states.size(),times_called_random_values, seed);
        times_called_random_values++;
        for (int l = 0; l < model.labels.size(); l++) {
            model.labelling_matrix[s * model.labels.size() + l] = labelling_probabilities[l];
        }
    }
}



probability *forward(const CupaalMarkovModel_Matrix &model) {
    // alpha matrix is a S x t matrix
    auto *alpha = static_cast<probability *>(cupaal::safe_malloc(sizeof(probability),
                                                                 model.states.size() * model.observations.at(
                                                                     0).size()));

    // alpha[t] where t = 0
    // alpha matrix is a S x t matrix
    // first S elements of alpha array alpha[0..5]
    for (int t = 0; t < model.states.size(); ++t) {
        const auto observation_index = model.label_index_map.at(model.observations.at(0).at(0));
        const auto omega_l = model.labelling_matrix[observation_index * model.states.size() + t];
        alpha[t] = omega_l * model.initial_distribution_vector[t];
    }

    // alpha[t] where 0 < t <= model.observations.size()
    for (int t = 1; t < model.observations.at(0).size(); ++t) {
        const auto alpha_index = t * model.states.size();
        const auto observation_index = model.label_index_map.at(model.observations.at(0).at(t));
        const auto omega_l = model.labelling_matrix[observation_index * model.states.size() + alpha_index];

        // P transposed matrix multiply with alpha[t-1]
        auto previous_transition_state_observation = 1;


        alpha[alpha_index] = omega_l * 1.0;
    }


    return alpha;
}


void baum_welch(const CupaalMarkovModel_Matrix &model, int seed=0) {
    // Initialize pi, P, and omega (based on observations)
    initialize_model_parameters(model, seed);
    const auto alpha = forward(model);


    free(alpha);
}


void manual_experiment2() {
    DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

    // Matrix A = [1, 2]
    //            [3, 4]
    DdNode *varX = Cudd_addNewVar(gbm);
    Cudd_Ref(varX);
    DdNode *varY = Cudd_addNewVar(gbm);
    Cudd_Ref(varY);
    DdNode *node1 = Cudd_addIte(gbm, varY, Cudd_addConst(gbm, 4), Cudd_addConst(gbm, 3));
    Cudd_Ref(node1);
    DdNode *node2 = Cudd_addIte(gbm, varY, Cudd_addConst(gbm, 2), Cudd_addConst(gbm, 1));
    Cudd_Ref(node2);
    const auto matrix_A_as_ADD = Cudd_addIte(gbm, varX, node1, node2);

    // Matrix B = [1, 2]
    //            [0, 1]
    DdNode *varZ = Cudd_addNewVar(gbm);
    Cudd_Ref(varZ);
    DdNode *varU = Cudd_addNewVar(gbm);
    Cudd_Ref(varU);
    DdNode *node3 = Cudd_addIte(gbm, varU, Cudd_addConst(gbm, 1.0), Cudd_addConst(gbm, 0.0));
    Cudd_Ref(node3);
    DdNode *node4 = Cudd_addIte(gbm, varU, Cudd_addConst(gbm, 2.0), Cudd_addConst(gbm, 1.0));
    Cudd_Ref(node4);
    const auto matrix_B_as_ADD = Cudd_addIte(gbm, varZ, node3, node4);

    int numVars = Cudd_ReadSize(gbm);
    std::cout << numVars << std::endl;
    int *permutation = (int *) malloc(numVars * sizeof(int));
    for (int i = 0; i < numVars; i++) {
        permutation[i] = i; // Default mapping: identity
    }

    permutation[2] = 1; // Map index 2 to index 1

    DdNode *renamedAdd = Cudd_addPermute(gbm, matrix_B_as_ADD, permutation);
    Cudd_Ref(renamedAdd);

    // Expected result of A x B = [1, 2]   [1, 2] = [1, 4]
    //                            [3, 4] X [0, 1]   [3, 10]
    auto mult_result = Cudd_addMatrixMultiply(gbm, matrix_A_as_ADD, renamedAdd, &varY, 1);

    cupaal::write_dd_to_dot(gbm, matrix_A_as_ADD, "/home/runge/CuPAAL/manual_A.dot");
    cupaal::write_dd_to_dot(gbm, matrix_B_as_ADD, "/home/runge/CuPAAL/manual_B.dot");
    cupaal::write_dd_to_dot(gbm, renamedAdd, "/home/runge/CuPAAL/manual_B_renamed.dot");
    cupaal::write_dd_to_dot(gbm, mult_result, "/home/runge/CuPAAL/manual_result.dot");

    Cudd_Quit(gbm);
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

int main(int argc, char *argv[]) {
    storm::utility::setUp();
    storm::settings::initializeAll("CuPAAL", "CuPAAL");
    DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

    CupaalMarkovModel_Matrix model;
    model.states = {0, 1, 2, 3, 4};
    model.labels = {"a", "b"};
    auto index = 0;
    for (const auto& label : model.labels) {
        model.label_index_map[label] = index;
        index++;
    }
    model.observations.push_back({"a", "b", "b", "a", "a", "b"});

    model.initial_distribution_vector = static_cast<probability *>(cupaal::safe_malloc(
        sizeof(probability), model.states.size()));

    model.transition_matrix = static_cast<probability *>(cupaal::safe_malloc(
        sizeof(probability), model.states.size() * model.states.size()));
    model.labelling_matrix = static_cast<probability *>(cupaal::safe_malloc(
        sizeof(probability), model.states.size() * model.labels.size()));


    baum_welch(model);

    for (state s: model.states) {
        std::cout << "probability: " << model.initial_distribution_vector[s] << std::endl;
    }

    for (state s: model.states) {
        for (state s_prime: model.states) {
            std::cout << "state " << s << " -> " << s_prime << " ";
            std::cout << model.transition_matrix[s * model.states.size() + s_prime] << ", ";
        }
        std::cout << std::endl;
    }

    for (state s = 0; s < model.states.size(); s++) {
        for (int l = 0; l < model.labels.size(); l++) {
            std::cout << "labelling: " << model.labelling_matrix[s * model.labels.size() + l] << " ";
        }
        std::cout << std::endl;
    }


    //
    // for (auto const& [state, distribution] : model.transition_matrix) {
    //     for (auto const [state_prime, probability] : distribution) {
    //         std::cout << "state -> state_prime" << state << ", " << state_prime << std::endl;
    //         std::cout << "probability" << probability << std::endl;
    //     }
    // }


    Cudd_Quit(gbm);
    exit(EXIT_SUCCESS);
}


// States (S) = [1, 2]
//              [3, 4]
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
