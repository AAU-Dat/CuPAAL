# Dependencies
This project relies on the CUDD library for the implementation of ADDs, and the Storm library for parsing Prism models

There are two immediate options for developing this library. Either you

1. Install Storm on your machine, which includes CUDD (**Recommended**, Linux)
2. Use the supplied devcontainer built on the storm image (Windows)

DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

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
        0.7, 0.3,
        1, 0
    }; // rows = length of observation sequence, columns = number of states

    // matrix = [0.3, 0.7]
    //          [0.1, 0.9]
    double transition_matrix[4] = {0.3, 0.7, 0.1, 0.9};

    // initial states = [1, 0]
    double initial_states[2] = {0.9, 0.1};

    // observation/trace
    // o = ["hungry", "hungry", "full", "hungry"]

    // transition_matrix_estimate = [0.5, 0.5]
    //                              [1.0, 0.0]

    // emission_matrix_estimate = [1, 0]
    //                            [0, 1]

    // initial_state_estimate = [1, 0]

    int dump_n_rows = 2;
    int dump_n_cols = 2;
    int n_row_vars = 0;
    int n_col_vars = 0;

    int n_vars = ceil(log2(2));
    auto row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_row_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));
    auto comp_col_vars = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), n_vars));

    DdNode *transition_matrix_as_add;
    DdNode *initial_states_matrix_as_add;
    auto labelling_matrix_as_add = static_cast<DdNode **>(cupaal::safe_malloc(sizeof(DdNode *), 4));

    auto add_result = cupaal::Sudd_addRead(
        transition_matrix,
        2,
        2,
        gbm,
        &transition_matrix_as_add,
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
    std::cout << add_result << std::endl;


    cupaal::Sudd_addRead(
        initial_states,
        2,
        1,
        gbm,
        &initial_states_matrix_as_add,
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

    for (int t = 0; t < 4; t++) {
        cupaal::Sudd_addRead(
            &l_matrix[t * 2],
            2,
            1,
            gbm,
            &labelling_matrix_as_add[t],
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

    DdNode **alpha_result = cupaal::forward(
        gbm,
        labelling_matrix_as_add,
        transition_matrix_as_add,
        initial_states_matrix_as_add,
        row_vars,
        col_vars,
        n_row_vars,
        4
    );

    DdNode **beta_result = cupaal::backward(
        gbm,
        labelling_matrix_as_add,
        transition_matrix_as_add,
        row_vars,
        col_vars,
        n_vars,
        4
    );

    cupaal::write_dd_to_dot(gbm, transition_matrix_as_add, "/home/runge/CuPAAL/transition_matrix.dot");
    cupaal::write_dd_to_dot(gbm, initial_states_matrix_as_add, "/home/runge/CuPAAL/initial_states.dot");
    cupaal::write_dd_to_dot(gbm, labelling_matrix_as_add[2], "/home/runge/CuPAAL/omega2.dot");
    cupaal::write_dd_to_dot(gbm, alpha_result[0], "/home/runge/CuPAAL/alpha0.dot");
    cupaal::write_dd_to_dot(gbm, beta_result[0], "/home/runge/CuPAAL/beta1.dot");

    auto res = kronecker_product();
    std::cout << res << std::endl;
    Cudd_Quit(gbm);