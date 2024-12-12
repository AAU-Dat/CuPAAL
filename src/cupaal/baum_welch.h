#ifndef BAUM_H
#define BAUM_H
#include <cuddObj.hh>

// This file should contain all the functions related to the Baum-Welch algorithm: forward-backward, update-parameter-estimates, etc.
namespace cupaal {
    extern DdNode **forward(DdManager *manager, DdNode **omega, DdNode *P, DdNode *pi, DdNode **row_vars,
                     DdNode **column_vars, int n_vars, int n_obs);

    extern DdNode **backward(DdManager *manager, DdNode **omega, DdNode *P, DdNode **row_vars,
                       DdNode **column_vars, int n_vars, int n_obs);
}

#endif //BAUM_H
