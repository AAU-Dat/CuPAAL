//
// Created by Runge on 13/10/2024.
//

#include "baum.h"
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>

using namespace parser;

void parseAndBuildPrism(std::string const & filename) {
    const auto program = storm::api::parseProgram(filename);

    constexpr std::vector<std::shared_ptr<storm::logic::Formula const>> formulas;

    auto symbolicmodel = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas);
}
