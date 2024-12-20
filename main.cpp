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

void manual_experiment1() {
    DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

    // Matrix A = [0.3, 0.7]
    //            [0.1, 0.9]
    DdNode *varX = Cudd_addNewVar(gbm);
    Cudd_Ref(varX);
    DdNode *varY = Cudd_addNewVar(gbm);
    Cudd_Ref(varY);
    DdNode *node1 = Cudd_addIte(gbm, varY, Cudd_addConst (gbm, 0.9), Cudd_addConst (gbm, 0.1));
    Cudd_Ref(node1);
    DdNode *node2 = Cudd_addIte(gbm, varY, Cudd_addConst (gbm, 0.7), Cudd_addConst (gbm, 0.3));
    Cudd_Ref(node2);
    const auto matrix_A_as_ADD = Cudd_addIte(gbm, varX, node1, node2);

    // Vector B = [1, 0]
    // DdNode *varB = Cudd_addNewVar(gbm);
    // Cudd_Ref(varB);
    DdNode *vector_B_as_ADD = Cudd_addIte(gbm, varY, Cudd_addConst (gbm, 0.0), Cudd_addConst (gbm, 1.0));
    Cudd_Ref(vector_B_as_ADD);

    auto mult_result = Cudd_addMatrixMultiply(gbm, matrix_A_as_ADD, vector_B_as_ADD, &varY, 1);

    cupaal::write_dd_to_dot(gbm, matrix_A_as_ADD, "/home/lars/CuPAAL/manual_A.dot");
    cupaal::write_dd_to_dot(gbm, vector_B_as_ADD, "/home/lars/CuPAAL/manual_B.dot");
    cupaal::write_dd_to_dot(gbm, mult_result, "/home/lars/CuPAAL/manual_result.dot");

    Cudd_Quit(gbm);
}


void manual_experiment2() {
    DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

    // Matrix A = [1, 2]
    //            [3, 4]
    DdNode *varX = Cudd_addNewVar(gbm);
    Cudd_Ref(varX);
    DdNode *varY = Cudd_addNewVar(gbm);
    Cudd_Ref(varY);
    DdNode *node1 = Cudd_addIte(gbm, varY, Cudd_addConst (gbm, 4), Cudd_addConst (gbm, 3));
    Cudd_Ref(node1);
    DdNode *node2 = Cudd_addIte(gbm, varY, Cudd_addConst (gbm, 2), Cudd_addConst (gbm, 1));
    Cudd_Ref(node2);
    const auto matrix_A_as_ADD = Cudd_addIte(gbm, varX, node1, node2);

    // Matrix B = [1, 2]
    //            [0, 1]
    DdNode *varZ = Cudd_addNewVar(gbm);
    Cudd_Ref(varZ);
    DdNode *varU = Cudd_addNewVar(gbm);
    Cudd_Ref(varU);
    DdNode *node3 = Cudd_addIte(gbm, varU, Cudd_addConst (gbm, 1.0), Cudd_addConst (gbm, 0.0));
    Cudd_Ref(node3);
    DdNode *node4 = Cudd_addIte(gbm, varU, Cudd_addConst (gbm, 2.0), Cudd_addConst (gbm, 1.0));
    Cudd_Ref(node4);
    const auto matrix_B_as_ADD = Cudd_addIte(gbm, varZ, node3, node4);

    int numVars = Cudd_ReadSize(gbm);
    std::cout << numVars << std::endl;
    int *permutation = (int *)malloc(numVars * sizeof(int));
    for (int i = 0; i < numVars; i++) {
        permutation[i] = i;  // Default mapping: identity
    }

    permutation[2] = 1;  // Map index 2 to index 1

    DdNode *renamedAdd = Cudd_addPermute(gbm, matrix_B_as_ADD, permutation);
    Cudd_Ref(renamedAdd);

    // Expected result of A x B = [1, 2]   [1, 2] = [1, 4]
    //                            [3, 4] X [0, 1]   [3, 10]
    auto mult_result = Cudd_addMatrixMultiply(gbm, matrix_A_as_ADD, renamedAdd, &varY, 1);

    cupaal::write_dd_to_dot(gbm, matrix_A_as_ADD, "/home/lars/CuPAAL/manual_A.dot");
    cupaal::write_dd_to_dot(gbm, matrix_B_as_ADD, "/home/lars/CuPAAL/manual_B.dot");
    cupaal::write_dd_to_dot(gbm, renamedAdd, "/home/lars/CuPAAL/manual_B_renamed.dot");
    cupaal::write_dd_to_dot(gbm, mult_result, "/home/lars/CuPAAL/manual_result.dot");

    Cudd_Quit(gbm);
}

int manual_experiment3() {
    FILE *fp = fopen("/home/lars/CuPAAL/matrix.txt", "r");
    if (!fp) {
        perror("File opening failed");
        return 1;
    }

    DdManager *dd = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    DdNode *E;
    DdNode **x, **y, **xn, **yn;
    int nx, ny, m, n;
    int bx = 0, sx = 1, by = 0, sy = 1;

    // Allocate arrays for variables
    x = (DdNode **)malloc(sizeof(DdNode *) * nx);
    y = (DdNode **)malloc(sizeof(DdNode *) * ny);
    xn = (DdNode **)malloc(sizeof(DdNode *) * nx);
    yn = (DdNode **)malloc(sizeof(DdNode *) * ny);

    // Read matrix into an ADD
    int result = Cudd_addRead(fp, dd, &E, &x, &y, &xn, &yn, &nx, &ny, &m, &n, bx, sx, by, sy);
    if (result == 0) {
        printf("Matrix successfully read into an ADD.\n");
    } else {
        printf("Error reading matrix into an ADD.\n");
    }

    cupaal::write_dd_to_dot(dd, E, "matrix_from_file");

    // Clean up
    Cudd_RecursiveDeref(dd, E);
    free(x);
    free(y);
    free(xn);
    free(yn);
    Cudd_Quit(dd);
    fclose(fp);

    return 0;
}

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

    auto row_vars2 = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto col_vars2 = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_row_vars2 = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_col_vars2 = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    double B_mat[4] = {1, 0, 0, 1};
    double A_mat[4] = {1, 2, 3, 4};

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
        &row_vars2,
        &col_vars2,
        &comp_row_vars2,
        &comp_col_vars2,
        &n_row_vars,
        &n_col_vars,
        &dump_n_rows,
        &dump_n_cols,
        2,
        ROW_VAR_INDEX_MULTIPLIER,
        2,
        COL_VAR_INDEX_MULTIPLIER
    );
    // Add<LibraryType, ValueType> swapVariables(std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& metaVariablePairs) const;

    //auto row_vars2 = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), 2));
    //B_as_add = Add<Dd<LibraryType>>::swapVariables(gbm, B_as_add, col_vars, row_vars, n_vars);
    auto transposed_a = Cudd_addSwapVariables(gbm, A_as_add, row_vars, col_vars, 1);

    auto x = Cudd_addMatrixMultiply(gbm, transposed_a, B_as_add, row_vars, 1);
    //auto x = Cudd_addTimesPlus(gbm, A_as_add, B_as_add, col_vars, 0);

    std::cout << row_vars[0]->index;
    std::cout << col_vars[0]->index;


    cupaal::write_dd_to_dot(gbm, A_as_add, "/home/lars/CuPAAL/A_matrix.dot");
    cupaal::write_dd_to_dot(gbm, transposed_a, "/home/lars/CuPAAL/A_matrix_transposed.dot");
    cupaal::write_dd_to_dot(gbm, B_as_add, "/home/lars/CuPAAL/B_matrix.dot");
    cupaal::write_dd_to_dot(gbm, x, "/home/lars/CuPAAL/X_matrix_mult.dot");



    return 1;
}

void experimental_testing() {
    // auto model = cupaal::parseAndBuildPrism("/home/lars/CuPAAL/models/polling.prism");
    // auto sparsemodel = cupaal::parseAndBuildPrismSparse("/home/lars/CuPAAL/models/polling.prism");
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

DdNode random_initialization_pi(DdManager *manager, int seed, const int n_vars, const int n_obs) {

    srand(seed); // Seed the random number generator
    DdNode *pi = Cudd_ReadZero(manager); // Start with the zero ADD
    Cudd_Ref(pi);

    return pi;
}


int main(int argc, char *argv[]) {
    storm::utility::setUp();
    storm::settings::initializeAll("CuPAAL", "CuPAAL");

    manual_experiment2();

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


