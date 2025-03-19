//
// Created by runge on 11/14/24.
//

#include "helpers.h"

void *cupaal::safe_malloc(const size_t type_size, size_t amount) {
    void *ptr = malloc(type_size * amount);
    if (ptr == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/**
 * @brief Create an array of values between 0 and 1, not included, which sums to 1.
 * @param size the size of the array to be generated
 * @param seed Controls the randomization. 0 means random. Any other value is a seed
 * @sideeffect None
 * @return a vector of probabilities summing to 1, or an empty array if size is less than 0
 */
std::vector<double> cupaal::generate_stochastic_probabilities(const unsigned long size, const int seed) {
    std::vector<double> probabilities(size);
    std::uniform_real_distribution<> distribution(0.01, 0.99);
    std::random_device random_device;
    std::mt19937 generator(random_device());

    if (seed != 0) {
        generator.seed(seed);
    }

    for (int i = 0; i < size; ++i) {
        probabilities[i] = distribution(generator);
    }

    // normalize
    double sum = 0.0;
    for (const auto probability: probabilities) {
        sum += probability;
    }
    for (double &p: probabilities) {
        p /= sum;
    }

    return probabilities;
}

std::vector<double> cupaal::generate_stochastic_probabilities(const unsigned long size, std::mt19937 generator) {
    std::vector<double> probabilities(size);
    std::uniform_real_distribution<> distribution(0.01, 0.99);

    for (int i = 0; i < size; ++i) {
        probabilities[i] = distribution(generator);
    }

    // normalize
    double sum = 0.0;
    for (const auto probability: probabilities) {
        sum += probability;
    }
    for (double &p: probabilities) {
        p /= sum;
    }

    return probabilities;
}


/**
 * @brief Write a decision diagram to a file.
 * @sideeffect None
 * @see Storm's exportToDot
*/
void cupaal::write_dd_to_dot(DdManager *manager, DdNode *dd, const char *filename) {
    FILE *outfile = fopen(filename, "w");
    if (outfile == nullptr) return;
    Cudd_DumpDot(manager, 1, &dd, nullptr, nullptr, outfile); // dump the function to .dot file
    fclose(outfile);
}

/**
 * @brief Parse a prism file into a symbolic representation using the Storm parser
 * @sideeffect None
 * @see storm::api::parseProgram, storm::api::buildSymbolicModel
*/
std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD> > cupaal::parseAndBuildPrism(
    std::string const &filename) {
    const auto program = storm::api::parseProgram(filename, true);
    constexpr std::vector<std::shared_ptr<storm::logic::Formula const> > formulas;
    auto symbolicmodel = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas);
    return symbolicmodel;
}


/**
 * @brief Parse a prism file into a sparse representation using the Storm parser
 * @sideeffect None
 * @see storm::api::parseProgram, storm::api::buildSymbolicModel
*/
std::shared_ptr<storm::models::sparse::Model<double> > cupaal::parseAndBuildPrismSparse(
    std::string const &filename) {
    const auto program = storm::api::parseProgram(filename, true);
    constexpr std::vector<std::shared_ptr<storm::logic::Formula const> > formulas;
    auto sparsemodel = storm::api::buildSparseModel<double>(program, formulas);
    return sparsemodel;
}
