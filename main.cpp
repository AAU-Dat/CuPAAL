#include <iostream>
#include <storm/api/storm.h>
#include "src/cupaal/baum.h"

int main() {
    parser::parseAndBuildPrism("some file path");
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
