//
// Created by Runge on 13/10/2024.
//

#include "baum.h"
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <cudd.h>
#include <cuddInt.h>

#include "Cudd_extensions.h"

DdNode* cupaal::cuddTestTwo(DdManager* dd)
{
    DdNode* x1 = Cudd_addNewVar(dd);
    Cudd_Ref(x1);
    DdNode* x2 = Cudd_addNewVar(dd);
    Cudd_Ref(x2);
    DdNode* node1 =
        Cudd_addIte(dd, x2,
                    Cudd_addConst(dd, 2),
                    Cudd_addConst(dd, 0)); //Then branch=2, Else branch=0
    Cudd_Ref(node1);

    DdNode* node2 =
        Cudd_addIte(dd, x2,
                    Cudd_addConst(dd, 3),
                    Cudd_addConst(dd, 2)); // Then branch=3, Else branch=2
    Cudd_Ref(node2);

    DdNode* add = Cudd_addIte(dd, x1, node1, node2); // Then branch=node1, Else branch=node2
    return add;
}


std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> parseAndBuildPrism(std::string const& filename)
{
    const auto program = storm::api::parseProgram(filename);
    constexpr std::vector<std::shared_ptr<storm::logic::Formula const>> formulas;
    auto symbolicmodel = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas);
    auto init = symbolicmodel->getInitialStates();
    // auto reachable = symbolicmodel->getReachableStates();
    // auto transitions = symbolicmodel->getTransitionMatrix();

    init.exportToDot("asdas");
    // Cudd_addLog(symbolicmodel->getManager().getInternalDdManager().getCuddManager().getManager(), init.getInternalBdd().getCuddDdNode());
    // Cudd_addMonadicApply(symbolicmodel->getManager().getInternalDdManager().getCuddManager().getManager(), Cudd_addLog, init.getInternalBdd().getCuddBdd().getNode());
    return symbolicmodel;
}
