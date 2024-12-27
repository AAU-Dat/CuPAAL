//
// Created by runge on 11/14/24.
//

#ifndef HELPERS_H
#define HELPERS_H
#include <cuddObj.hh>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>

namespace cupaal {
    extern void *safe_malloc(size_t type_size, size_t amount);

    extern std::vector<double> generate_stochastic_probabilities(unsigned long size, int seed = 0);

    extern void write_dd_to_dot(DdManager *manager, DdNode *dd, const char *filename);

    extern std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD> > parseAndBuildPrism(
        std::string const &filename);

    extern std::shared_ptr<storm::models::sparse::Model<double> > parseAndBuildPrismSparse(std::string const &filename);
}

#endif //HELPERS_H
