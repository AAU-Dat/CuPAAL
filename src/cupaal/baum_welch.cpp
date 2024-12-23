#include "baum_welch.h"
#include "helpers.h"
#include <cuddInt.h>

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
