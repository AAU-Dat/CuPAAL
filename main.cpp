#include <iostream>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>

int main() {
    std::cout << "Hello, World!" << std::endl;


    const auto program = storm::api::parseProgram("/workspaces/CuPAAL/polling.3.v1.prism", true);

    constexpr std::vector<std::shared_ptr<storm::logic::Formula const>> formulas;

    auto model = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas);
    auto init = model->getInitialStates();
    init.exportToDot("penis.dot");

    return 0;
}
