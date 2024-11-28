#include <iostream>
// #include <storm/utility/initialize.h>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"

int main(int argc, char *argv[]) {
    // storm::utility::setUp();
    storm::settings::initializeAll("CuPAAL", "CuPAAL");

    // DdManager *gbm = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);

    // auto model = cupaal::parseAndBuildPrism("/home/runge/CuPAAL/models/polling.prism");
    auto model = cupaal::parseAndBuildPrismSparseModel("/workspaces/CuPAAL/polling.3.v1.prism");

    std::cout << "penis";;

    // Cudd_Quit(gbm);
    exit(EXIT_SUCCESS);
}
