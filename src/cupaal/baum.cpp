//
// Created by Runge on 13/10/2024.

#include "baum.h"
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>

/**
 * Parse a prism file and build a symbolic model representation
 * @param filename Name of the Prism file to parse
 * @return A (storm) model representation of the Prism file
 */
std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> parser::parseAndBuildPrism(std::string const & filename) {
    const auto program = storm::api::parseProgram(filename);

    constexpr std::vector<std::shared_ptr<storm::logic::Formula const>> formulas;

    auto symbolicmodel = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas);

    return symbolicmodel;
}
