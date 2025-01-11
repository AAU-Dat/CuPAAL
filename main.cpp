#include <iostream>
#include <storm/utility/initialize.h>
#include <cuddInt.h>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"

using state = int;
using probability = CUDD_VALUE_TYPE;
using label = std::string;

// int kronecker_product() {
//     // const auto kronecker_product = static_cast<DdNode *>(cupaal::safe_malloc(sizeof(DdNode *), 4));
//     DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
//     int n_vars = ceil(log2(2));
//     int dump_n_rows = 2;
//     int dump_n_cols = 2;
//     int n_row_vars = 0;
//     int n_col_vars = 0;
//     auto row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
//     auto col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
//     auto comp_row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
//     auto comp_col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
//     double A_mat[4] = {1, 0, 0, 1};
//     double B_mat[4] = {1, 2, 3, 4};
//
//     DdNode *A_as_add;
//     DdNode *B_as_add;
//
//     cupaal::Sudd_addRead(
//         A_mat,
//         2,
//         2,
//         gbm,
//         &A_as_add,
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
//
//     cupaal::Sudd_addRead(
//         B_mat,
//         2,
//         2,
//         gbm,
//         &B_as_add,
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
//     //
//
//     auto row_vars2 = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), 4));
//
//     std::cout << row_vars2[0]->index << std::endl;
//     auto x = Cudd_addMatrixMultiply(gbm, A_as_add, B_as_add, row_vars2, 2);
//     // auto y = Cudd_addTimesPlus(gbm, A_as_add, B_as_add, row_vars, 1);
//
//     cupaal::write_dd_to_dot(gbm, A_as_add, "/home/runge/CuPAAL/A_matrix.dot");
//     cupaal::write_dd_to_dot(gbm, B_as_add, "/home/runge/CuPAAL/B_matrix.dot");
//     cupaal::write_dd_to_dot(gbm, x, "/home/runge/CuPAAL/X_matix_mult.dot");
//
//
//     return 1;
// }

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

void matrix_variant() {
    cupaal::MarkovModel_Matrix model;
    model.states = {0, 1, 2, 3, 4};
    model.labels = {"a", "b"};
    model.observations.push_back({"a", "b", "b", "a", "a", "b", "a"});
    model.initialize_model_parameters_randomly();
    model.print_model_parameters();
    model.print_calculations = true;
    const auto result = baum_welch_matrix(model);
}

void algebraic_decision_diagram_variant() {
    DdManager *dd_manager = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    std::random_device random_device;
    std::mt19937 generator(random_device());
    // generator.seed(42);
    cupaal::MarkovModel_ADD model;
    model.manager = dd_manager;
    model.states = {0, 1, 2, 3};
    model.labels = {"a", "b"};
    model.observations.push_back({"a", "b", "b", "a", "a", "b", "a"});
    model.initialize_model_parameters_randomly(generator);
    model.print_model_parameters();

    Cudd_Quit(dd_manager);
}


int main(int argc, char *argv[]) {
    // storm::utility::setUp();
    // storm::settings::initializeAll("CuPAAL", "CuPAAL");

    algebraic_decision_diagram_variant();

    exit(EXIT_SUCCESS);
}


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

