#include <cuddInt.h>
#include <iostream>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include "src/cupaal/baum.h"
#include "src/cupaal/Cudd_extensions.h"
#include <storm/utility/initialize.h>
// #include <cudd.h>

int main()
{
    std::cout << cupaal::AddRead() << std::endl;

    storm::utility::setUp();
    storm::settings::initializeAll("storm-starter-project", "main");

    const auto program = storm::api::parseProgram("/home/runge/CuPAAL/polling.prism", true);

    constexpr std::vector<std::shared_ptr<storm::logic::Formula const>> formulas;

    const auto model = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas)->as
        <storm::models::symbolic::Ctmc<storm::dd::DdType::CUDD, double>>();

    const auto model2 = storm::api::buildSparseModel<double>(program, formulas);

    const auto testawer = model2->getTransitionMatrix();
    std::cout << testawer << std::endl;

    // Below is similar to ExitRate apparently?
    // auto testing = model->getTransitionMatrix().sumAbstract(model->getColumnVariables());

    auto transitions = model->getTransitionMatrix();

    transitions.exportToDot("/home/runge/CuPAAL/init.dot");

    auto result = Cudd_addMonadicApply(model->getManager().getInternalDdManager().getCuddManager().getManager(),
                                       Cudd_addLog, transitions.getInternalAdd().getCuddDdNode());

    // storm::dd::Add<storm::dd::DdType::CUDD> res = result;
    FILE* fp = fopen("/home/runge/CuPAAL/init-log.dot", "w");
    Cudd_DumpDot(model->getManager().getInternalDdManager().getCuddManager().getManager(), 1, &result, nullptr, nullptr,
                 fp);

    fclose(fp);

    const auto gbm = Cudd_Init(0,0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS,0);

    auto resultCudd = cupaal::cuddTestTwo(gbm);
    std::cout <<gbm->constants.slots << std::endl;

    fprintf(fp, "%x", gbm->vars[0]->index);


    FILE *fp2 = fopen("/home/runge/CuPAAL/cuddtest2.dot", "w");
    const char* nodeNames[] = {"x1", "x2", "x3", "x4"};
    char const* str2[] = {"a", "b"};
    Cudd_DumpDot(gbm, 2, &resultCudd, str2, nullptr, fp2);
    fclose(fp2);


    // result.exportToDot("/home/runge/CuPAAL/init-log.dot");

    // auto model = parser::parseAndBuildPrism("some file path");
    //
    // DdManager* gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    //
    // DdNode *bdd = Cudd_bddNewVar(gbm); /*Create a new BDD variable*/
    // Cudd_Ref(bdd); /*Increases the reference count of a node*/
    // bdd = Cudd_BddToAdd(gbm, bdd);
    //
    // Cudd_PrintDebug(gbm, bdd, 2, 4);
    //
    // cuddAddMonadicApplyRecur(gbm, Cudd_addExp, bdd);
    //
    // Cudd_PrintDebug(gbm, bdd, 2, 4);
    //
    // std::cout << "Hello, World!" << std::endl;
    // Cudd_Quit(gbm);
    return 0;
}
