#define DD_DEBUG
#include <iostream>
#include <cuddInt.h>
#include <filesystem>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"

struct options {
    std::string modelPath;
    std::string sequencesPath;
    int iterations = 100;
    double epsilon = 1e-6;
    int time = 3600;
    std::string outputPath;
    std::string resultPath;
} options;

void parse_input_options(const int argc, char **argv) {
    const cupaal::InputParser input(argc, argv);

    if (!input.hasCmdOption("-m") || !input.hasCmdOption("-s")) {
        std::cerr << "error: please supply both a model and data file" << std::endl;
        exit(EXIT_FAILURE);
    }

    options.modelPath = input.getCmdOption("-m");
    options.sequencesPath = input.getCmdOption("-s");

    if (input.hasCmdOption("-i")) {
        options.iterations = std::stoi(input.getCmdOption("-i"));
    }
    if (input.hasCmdOption("-e")) {
        options.epsilon = std::stod(input.getCmdOption("-e"));
    }
    if (input.hasCmdOption("-t")) {
        options.time = std::stoi(input.getCmdOption("-t"));
    }
    if (input.hasCmdOption("-o")) {
        options.outputPath = input.getCmdOption("-o");
    }
    if (input.hasCmdOption("-r")) {
        options.resultPath = input.getCmdOption("-r");
    }
}

int main(const int argc, char *argv[]) {
    parse_input_options(argc, argv);
    std::cout << "Reading model from file: " << options.modelPath << std::endl;
    std::cout << "Reading sequences from file: " << options.sequencesPath << std::endl;
    DdManager *dd_manager = Cudd_Init(0, 0,CUDD_UNIQUE_SLOTS,CUDD_CACHE_SLOTS, 0);
    // Cudd_SetEpsilon(dd_manager, 0);

    cupaal::MarkovModel model;
    model.manager = dd_manager;
    model.initialize_from_file(options.modelPath);

    Cudd_PrintDebug(dd_manager, model.tau, 2, 4);
    // cupaal::write_dd_to_dot(model.manager, model.tau, "../tau.dot");

    model.add_observation_from_file(options.sequencesPath);
    model.baum_welch(options.iterations, options.epsilon);

    if (!options.outputPath.empty()) {
        std::cout << "Saving model to: " << options.outputPath << std::endl;
        model.export_to_file(options.outputPath);
    }

    if (!options.resultPath.empty()) {
        std::cout << "Saving iteration details to: " << options.resultPath << std::endl;
        // TODO: implement logic to save the data. Possible options: csv or json
    }

    model.clean_up_cudd();
    std::cout << "Remaining references (expecting 0): " << Cudd_CheckZeroRef(dd_manager) << std::endl;
    Cudd_Quit(dd_manager);
    exit(EXIT_SUCCESS);
}
