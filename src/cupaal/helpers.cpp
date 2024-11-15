//
// Created by runge on 11/14/24.
//

#include "helpers.h"

/**
 * @brief Write a decision diagram to a file.
 * @sideeffect None
 * @see Storm's exportToDot
*/
void cupaal::write_dd_to_dot(DdManager *manager, DdNode *dd, const char *filename) {
    FILE *outfile = fopen(filename, "w");
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
