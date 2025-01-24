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
        probability *omega_matrix; // omega

        void initialize_model_parameters_randomly(int seed = 0);

        void print_model_parameters() const;

        void calculate_omega();
    };

    class MarkovModel_ADD {
    public:
        DdManager *manager;
        bool print_calculations = false;
        std::set<state> states;
        std::vector<label> labels;
        std::map<label, state> label_index_map;
        std::vector<std::vector<label> > observations;
        DdNode *transition_add; // P
        DdNode *initial_distribution_add; // pi
        DdNode **labelling_add; // labelling add

        void initialize_probabilities(probability *transition_function, probability *labelling_function,
                                      probability *initial_distribution);

        void initialize_helpers();

        void initialize_model_parameters_randomly(std::mt19937 generator);

        void print_model_parameters() const;

        [[nodiscard]] DdNode **forward_add() const;

        [[nodiscard]] DdNode **backward_add() const;

        [[nodiscard]] DdNode **gamma_add(DdNode **alpha, DdNode **beta) const;

        [[nodiscard]] DdNode **xi_add(DdNode **alpha, DdNode **beta) const;

        void update_model_parameters_add(DdNode **gamma, DdNode **xi);

    private:
        int dump_n_rows = 0;
        int dump_n_cols = 0;
        int n_row_vars = 0;
        int n_col_vars = 0;
        int n_vars = 0;
        DdNode **row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
        DdNode **col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
        DdNode **comp_row_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
        DdNode **comp_col_vars = static_cast<DdNode **>(safe_malloc(sizeof(DdNode *), n_vars));
    };

    extern std::vector<probability> forward_matrix(const MarkovModel_Matrix &model);

    extern std::vector<probability> backward_matrix(const MarkovModel_Matrix &model);

    extern std::vector<probability> gamma_matrix(const MarkovModel_Matrix &model, const std::vector<probability> &alpha,
                                                 const std::vector<probability> &beta);

    extern std::vector<probability> xi_matrix(const MarkovModel_Matrix &model, const std::vector<probability> &alpha,
                                              const std::vector<probability> &beta);

    extern MarkovModel_Matrix baum_welch_matrix(const MarkovModel_Matrix &model);

    extern DdNode **forward(DdManager *manager, DdNode **omega, DdNode *P, DdNode *pi, DdNode **row_vars,
                            DdNode **column_vars, int n_vars, int n_obs);

    extern DdNode **backward(DdManager *manager, DdNode **omega, DdNode *P, DdNode **row_vars,
                             DdNode **column_vars, int n_vars, int n_obs);
}

#endif //BAUM_H
