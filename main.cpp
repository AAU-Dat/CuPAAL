#define DD_DEBUG
#include <chrono>
#include <cuddInt.h>
#include <iostream>

#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"
#include "src/cupaal/helpers.h"

struct options {
    std::string modelPath;
    std::string sequencesPath;
    int iterations = 100;
    double epsilon = 1e-2;
    std::chrono::seconds time = std::chrono::seconds(14400);
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
        options.iterations = input.getIntFromCmdOption("-i");
    }
    if (input.hasCmdOption("-e")) {
        options.epsilon = input.getDoubleFromCmdOption("-e");
    }
    if (input.hasCmdOption("-t")) {
        const int seconds = input.getIntFromCmdOption("-t");
        options.time = std::chrono::seconds(seconds);
    }
    if (input.hasCmdOption("-o")) {
        options.outputPath = input.getCmdOption("-o");
    }
    if (input.hasCmdOption("-r")) {
        options.resultPath = input.getCmdOption("-r");
    }
}

int main(const int argc, char *argv[]) {
    const auto program_start = std::chrono::steady_clock::now();
    parse_input_options(argc, argv);
    std::cout << "Reading model from file: " << options.modelPath << std::endl;
    std::cout << "Reading sequences from file: " << options.sequencesPath << std::endl;

    cupaal::MarkovModel model(options.modelPath, options.sequencesPath);

    if (model.observations.size() > 1) {
        model.baum_welch_multiple_observations(options.iterations, options.epsilon, options.time);
    } else {
        model.baum_welch(options.iterations, options.epsilon, options.time);
    }


    if (!options.outputPath.empty()) {
        std::cout << "Saving model to: " << options.outputPath << std::endl;
        model.export_to_file(options.outputPath);
    }

    if (!options.resultPath.empty()) {
        std::cout << "Saving iteration details to: " << options.resultPath << std::endl;
        model.save_experiment_to_csv(options.resultPath);
    }

    model.clean_up_cudd();
    std::cout << "Remaining references (expecting 0): " << Cudd_CheckZeroRef(model.manager) << std::endl;
    const auto program_end = std::chrono::steady_clock::now();
    const auto elapsed_time = program_end - program_start;
    std::cout << "Total time spent(s): " << std::chrono::duration_cast<std::chrono::seconds>(elapsed_time) << std::endl;
    Cudd_Quit(model.manager);
    exit(EXIT_SUCCESS);
}
