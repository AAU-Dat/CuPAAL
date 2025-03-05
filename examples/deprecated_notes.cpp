//
// Created by runge on 2/24/25.
//
#include <iostream>
#include "../src/cupaal/baum_welch.h"
#include "../src/cupaal/cudd_extensions.h"
#include "../src/cupaal/helpers.h"

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
    model.initialize_model_parameters_randomly(42);
    model.print_model_parameters();
    model.print_calculations = true;
    const auto result = baum_welch_matrix(model);
}

void manual_matrix_variant() {
    cupaal::MarkovModel_Matrix model;
    double labeling[] = {0.3, 0.4, 0.3, 0.4, 0.3, 0.3};
    double mat[] = {0.6, 0.4, 0.3, 0.7};
    double init[] = {0.8, 0.2};
    const std::map<std::string, int> index_map = {{"r", 0}, {"w", 1}, {"b", 2}};

    model.states = {0, 1};
    model.labels = {"r", "w", "b"};
    model.label_index_map = index_map;
    model.observations.push_back({"r", "w", "b", "b"});
    model.labelling_matrix = labeling;
    model.transition_matrix = mat;
    model.initial_distribution_vector = init;
    model.print_calculations = true;
    model.print_model_parameters();
    const auto result = baum_welch_matrix(model);
}

void algebraic_decision_diagram_variant() {
    DdManager *dd_manager = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    Cudd_SetEpsilon(dd_manager, 0);
    std::random_device random_device;
    std::mt19937 generator(random_device());
    generator.seed(42);
    cupaal::MarkovModel_ADD model;
    model.manager = dd_manager;
    model.states = {0, 1, 2, 3};
    model.labels = {"a", "b"};
    model.observations.push_back({"a", "b", "b", "a", "a", "b", "a"});
    model.initialize_model_parameters_randomly(generator);
    model.print_model_parameters();

    Cudd_Quit(dd_manager);
}


void manual_multiplication() {
    //double transition_function[] = {0.6, 0.3, 0.4, 0.7};
    DdManager *dd_manager = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    Cudd_SetEpsilon(dd_manager, 0);
    DdNode *x = Cudd_addNewVar(dd_manager); /*First variable in the crossover DD*/
    Cudd_Ref(x);
    DdNode *y = Cudd_addNewVar(dd_manager); /*Second variable in the crossover DD*/
    Cudd_Ref(y);
    DdNode *node1 = Cudd_addIte(dd_manager, y, Cudd_addConst(dd_manager, 0.4), Cudd_addConst(dd_manager, 0.6));
    /*Else branch=0, Then branch=2*/
    Cudd_Ref(node1);
    DdNode *node2 = Cudd_addIte(dd_manager, y, Cudd_addConst(dd_manager, 0.7), Cudd_addConst(dd_manager, 0.3));
    /*Else branch=1, Then branch=3*/
    Cudd_Ref(node2);

    // alpha[0] = 0.24 0.08
    // DdNode *alpha_x = Cudd_addNewVar(dd_manager); /*First variable in the crossover DD*/
    // Cudd_Ref(alpha_x);
    // DdNode *alpha_y = Cudd_addNewVar(dd_manager); /*Second variable in the crossover DD*/
    // Cudd_Ref(alpha_y);
    DdNode *alpha_node_1 = Cudd_addIte(dd_manager, x, Cudd_addConst(dd_manager, 0.08), Cudd_addConst(dd_manager, 0.28));
    Cudd_Ref(alpha_node_1);
    DdNode *alpha_zero = Cudd_addIte(dd_manager, y, alpha_node_1, alpha_node_1);

    DdNode *transition_function = Cudd_addIte(dd_manager, x, node2, node1);

    // int input[] = {0, 1};
    // auto evaluated = Cudd_Eval(dd_manager, retval, input);
    // Cudd_PrintDebug(dd_manager, evaluated, 1, 4);

    Cudd_RecursiveDeref(dd_manager, x);
    Cudd_RecursiveDeref(dd_manager, y);
    Cudd_RecursiveDeref(dd_manager, node1);
    Cudd_RecursiveDeref(dd_manager, node2);

    Cudd_PrintDebug(dd_manager, transition_function, 1, 4);
    Cudd_PrintDebug(dd_manager, alpha_zero, 1, 4);

    auto multiplication_one = Cudd_addMatrixMultiply(dd_manager, transition_function, alpha_zero, &x, 1);
    cuddRef(multiplication_one);
    auto alpha_one = Cudd_addSwapVariables(dd_manager, multiplication_one, &y, &x, 1);
    cuddRef(alpha_one);
    Cudd_PrintDebug(dd_manager, multiplication_one, 2, 4);
    Cudd_PrintDebug(dd_manager, alpha_one, 2, 4);

    auto multiplication_two = Cudd_addMatrixMultiply(dd_manager, transition_function, alpha_one, &x, 1);
    cuddRef(multiplication_two);
    auto alpha_two = Cudd_addSwapVariables(dd_manager, multiplication_two, &y, &x, 1);
    cuddRef(alpha_two);

    Cudd_PrintDebug(dd_manager, multiplication_two, 2, 4);
    Cudd_PrintDebug(dd_manager, alpha_two, 2, 4);
}
