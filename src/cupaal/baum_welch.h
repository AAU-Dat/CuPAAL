#ifndef BAUM_H
#define BAUM_H
#include <cuddObj.hh>
#include <map>

#include "helpers.h"

using state = int;
using probability = CUDD_VALUE_TYPE;
using label = std::string;

// This file should contain all the functions related to the Baum-Welch algorithm: forward-backward, update-parameter-estimates, etc.
namespace cupaal {
    struct report {
        unsigned int iterations;
        unsigned long microseconds;
        CUDD_VALUE_TYPE log_likelihood;
    };

    class MarkovModel {
    public:
        DdManager *manager;
        std::vector<std::string> states;
        std::vector<std::string> labels;
        std::vector<std::vector<std::string> > observations;

        std::vector<double> emissions;
        std::vector<double> transitions;
        std::vector<double> initial_distribution;

        DdNode **omega;
        DdNode *tau;
        DdNode *pi;
        DdNode *row_cube;

        [[nodiscard]] DdNode **calculate_alpha() const;

        [[nodiscard]] DdNode **calculate_beta() const;

        [[nodiscard]] DdNode **calculate_gamma(DdNode **alpha, DdNode **beta) const;

        [[nodiscard]] DdNode **calculate_xi(DdNode **alpha, DdNode **beta) const;

        void update_model_parameters(DdNode **gamma, DdNode **xi);

        void baum_welch(unsigned int max_iterations = 100, double epsilon = 1e-6);

        void initialize_from_file(const std::string &filename);

        void add_observation(const std::vector<std::string> &observation);

        void export_to_file(const std::string &filename);

        void clean_up_cudd() const;

    private:
        int number_of_states = 0;
        int number_of_labels = 0;
        std::map<std::string, int> label_index_map;
        std::vector<std::vector<int> > mapped_observations;

        int dump_n_rows = 0;
        int dump_n_cols = 0;
        int n_row_vars = 0;
        int n_col_vars = 0;
        int n_vars = 0;
        DdNode **row_vars = nullptr;
        DdNode **col_vars = nullptr;
        DdNode **comp_row_vars = nullptr;
        DdNode **comp_col_vars = nullptr;

        double calculate_log_likelihood(DdNode **alpha) const;
    };
}

#endif //BAUM_H
