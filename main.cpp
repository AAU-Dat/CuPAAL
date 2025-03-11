#include <iostream>
#include <filesystem>
#include <storm/utility/initialize.h>
#include <cuddInt.h>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"


int main(int argc, char *argv[]) {
    // storm::utility::setUp();
    // storm::settings::initializeAll("CuPAAL", "CuPAAL");
    const std::filesystem::path dir(getenv("HOME"));
    const std::filesystem::path full_path = dir / "thesis" / "CuPAAL";
    DdManager *dd_manager = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    // Cudd_SetEpsilon(dd_manager, 0);

    cupaal::MarkovModel model;
    model.manager = dd_manager;
    model.initialize_from_file("../example-model.txt");
    cupaal::write_dd_to_dot(model.manager, model.tau, (full_path / "tau.dot").c_str());

    model.add_observation({{"r", "b", "g"}});
    model.baum_welch(100);
    model.export_to_file("../example-model-output.txt");

    model.clean_up_cudd();

    std::cout << "Remaining references (expecting 0): " << Cudd_CheckZeroRef(dd_manager) << std::endl;
    Cudd_Quit(dd_manager);

    exit(EXIT_SUCCESS);
}
