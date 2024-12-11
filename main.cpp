#include <iostream>
#include <storm/utility/initialize.h>
#include <cuddInt.h>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"

int main(int argc, char *argv[]) {
    storm::utility::setUp();
    storm::settings::initializeAll("CuPAAL", "CuPAAL");

    // DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

    auto model = cupaal::parseAndBuildPrism("/home/runge/CuPAAL/models/polling.prism");
    auto sparsemodel = cupaal::parseAndBuildPrismSparse("/home/runge/CuPAAL/models/polling.prism");

    std::cout << sparsemodel->getNumberOfStates() << std::endl;
    std::cout << sparsemodel->getType() << std::endl;

    storm::storage::SparseMatrix<double>::index_type var2 = 0;

    while (var2 < sparsemodel->getNumberOfStates()) {
        for (const auto& label : sparsemodel->getStateLabeling().getLabelsOfState(var2)) {
            std::cout << var2 << "<< index, label >> " << label << std::endl;
            std::cout << "index actions: " << sparsemodel->getTransitionMatrix().getRowGroupSize(var2) << std::endl;
        }
        var2 = var2 + 1;
    }

    for (const auto& basic_string : sparsemodel->getStateLabeling().getLabels()) {
        std::cout << basic_string << std::endl;
        std::cout << sparsemodel->getStates(basic_string) << std::endl;
    }




    // std::cout << sparsemodel->getTransitionMatrix().getRow(3).begin()->getValue();

    for (auto element : sparsemodel->getTransitionMatrix().getRow(27)) {
        std::cout << element.getColumn() << " << col, val >> ";
        std::cout << element.getValue() << std::endl;
    }


    // Cudd_Quit(gbm);
    exit(EXIT_SUCCESS);
}
