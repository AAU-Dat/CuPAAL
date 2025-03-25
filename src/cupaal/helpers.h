//
// Created by runge on 11/14/24.
//

#ifndef HELPERS_H
#define HELPERS_H
#include <cuddObj.hh>
#include <storm/api/storm.h>

namespace cupaal {
    class InputParser {
    public:
        InputParser(const int &argc, char **argv) {
            for (int i = 1; i < argc; ++i)
                this->tokens.emplace_back(argv[i]);
        }

        [[nodiscard]] std::string getCmdOption(const std::string &option) const {
            if (auto itr = std::ranges::find(this->tokens, option);
                itr != this->tokens.end() && ++itr != this->tokens.end()) {
                return *itr;
            }
            static const std::string empty_string;
            return empty_string;
        }

        [[nodiscard]] bool hasCmdOption(const std::string &option) const {
            return std::ranges::find(this->tokens, option) != this->tokens.end();
        }

        [[nodiscard]] int getIntFromCmdOption(const std::string &option) const {
            try {
                return std::stoi(getCmdOption(option));
            } catch (const std::invalid_argument &e) {
                std::cerr << "wrong input for " << option << ": " << getCmdOption(option) << std::endl;
                exit(EXIT_FAILURE);
            } catch (const std::out_of_range &e) {
                std::cerr << "input out of rage for " << option << ": " << getCmdOption(option) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        [[nodiscard]] double getDoubleFromCmdOption(const std::string &option) const {
            try {
                return std::stod(getCmdOption(option));
            } catch (const std::invalid_argument &e) {
                std::cerr << "wrong input for " << option << ": " << getCmdOption(option) << std::endl;
                exit(EXIT_FAILURE);
            } catch (const std::out_of_range &e) {
                std::cerr << "input out of rage for " << option << ": " << getCmdOption(option) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

    private:
        std::vector<std::string> tokens;
    };

    extern void *safe_malloc(size_t type_size, size_t amount);

    extern std::vector<double> generate_stochastic_probabilities(unsigned long size, int seed = 0);

    extern std::vector<double> generate_stochastic_probabilities(unsigned long size, std::mt19937 generator);

    extern void write_dd_to_dot(DdManager *manager, DdNode *dd, const char *filename);

    extern std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD> > parseAndBuildPrism(
        std::string const &filename);

    extern std::shared_ptr<storm::models::sparse::Model<double> > parseAndBuildPrismSparse(std::string const &filename);
}

#endif //HELPERS_H
