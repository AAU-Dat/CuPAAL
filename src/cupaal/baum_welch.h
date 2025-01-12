#ifndef BAUM_H
#define BAUM_H
#include <cuddObj.hh>
#include <map>
#include <set>

#include "helpers.h"

using state = int;
using probability = CUDD_VALUE_TYPE;
using label = std::string;

// This file should contain all the functions related to the Baum-Welch algorithm: forward-backward, update-parameter-estimates, etc.
namespace cupaal {
    class MarkovModel_Matrix {
    public:
        bool print_calculations = false;
        std::set<state> states;
        std::vector<label> labels;
        std::map<label, state> label_index_map;
        std::vector<std::vector<label> > observations;
        probability *labelling_matrix; // labelling matrix
        probability *transition_matrix; // P
        probability *initial_distribution_vector; // pi

        void initialize_model_parameters_randomly(int seed = 0);

        void print_model_parameters() const;

        void calculate_omega();
    };

    extern std::vector<probability> forward_matrix(const MarkovModel_Matrix &model);

    extern std::vector<probability> backward_matrix(const MarkovModel_Matrix &model);

    extern std::vector<probability> gamma_matrix(const MarkovModel_Matrix &model, const std::vector<probability> &alpha, const std::vector<probability> &beta);

    extern std::vector<probability> xi_matrix(const MarkovModel_Matrix &model, const std::vector<probability> &alpha, const std::vector<probability> &beta);
    extern void update_matrix(const MarkovModel_Matrix &model, const std::vector<probability> &alpha, const std::vector<probability> &beta);

    extern MarkovModel_Matrix baum_welch_matrix(const MarkovModel_Matrix &model);

    extern DdNode **forward(DdManager *manager, DdNode **omega, DdNode *P, DdNode *pi, DdNode **row_vars,
                            DdNode **column_vars, int n_vars, int n_obs);

    extern DdNode **backward(DdManager *manager, DdNode **omega, DdNode *P, DdNode **row_vars,
                             DdNode **column_vars, int n_vars, int n_obs);
}

#endif //BAUM_H
