#ifndef BAUM_H
#define BAUM_H
#include <cuddObj.hh>
#include <map>

#include "helpers.h"


// This file should contain all the functions related to the Baum-Welch algorithm: forward-backward, update-parameter-estimates, etc.
namespace cupaal {
    struct iterationReport {
        unsigned int iteration_number;
        std::chrono::microseconds running_time_microseconds;
        double log_likelihood;
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

        [[nodiscard]] DdNode **calculate_alpha(const std::vector<int> &observation) const;

        [[nodiscard]] DdNode **calculate_beta(const std::vector<int> &observation) const;

        [[nodiscard]] DdNode **calculate_gamma(DdNode **alpha, DdNode **beta) const;

        [[nodiscard]] DdNode **calculate_xi(DdNode **alpha, DdNode **beta, const std::vector<int> &observation) const;

        void update_model_parameters(DdNode **gamma, DdNode **xi);

        void update_model_parameters_multiple_observations(const std::vector<DdNode **> &gammas,
                                                           const std::vector<DdNode **> &xis,
                                                           const std::map<std::vector<int>, int> &observation_map);

        void baum_welch(unsigned int max_iterations = 100, double epsilon = 1e-6,
                        std::chrono::seconds time = std::chrono::seconds(3600));

        void baum_welch_multiple_observations(unsigned int max_iterations = 100, double epsilon = 1e-6,
                                              std::chrono::seconds time = std::chrono::seconds(3600));

        void export_to_file(const std::string &filename);

        void save_experiment_to_csv(const std::string &filename);

        void clean_up_cudd() const;

        MarkovModel(std::string model_path, std::string sequences_path){
            this->manager = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
            Cudd_SetEpsilon(this->manager, 0);
            initialize_from_file(model_path);
            add_observation_from_file(sequences_path);
        }

        MarkovModel(std::vector<std::string> states, std::vector<std::string> labels,std::vector<double> initial_distribution, std::vector<double> transitions, std::vector<double> emissions, std::vector<std::vector<std::string>> observations){
            this->manager = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
            Cudd_SetEpsilon(this->manager, 0);
            this->states = std::move(states);
            this->labels = std::move(labels);
            this->initial_distribution = std::move(initial_distribution);
            this->transitions = std::move(transitions);
            this->emissions = std::move(emissions);
            
            initialize_adds();
            for (const auto& observation : observations) {
                add_observation(observation);
            }
        }

        
    private:
        std::vector<iterationReport> iteration_reports;
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
        void initialize_adds();
        void initialize_from_file(const std::string &filename);
        void add_observation_from_file(const std::string &filename);
        void add_observation(const std::vector<std::string> &observation);

    };
}

#endif //BAUM_H
