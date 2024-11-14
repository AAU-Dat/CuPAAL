#include <iostream>
#include "src/cupaal/baum_welch.h"
#include "src/cupaal/cudd_extensions.h"

int main() {
    std::cout << "Hello, World!" << cupaal::Forward() <<"\n";

    cupaal::Cudd_addExp(nullptr, nullptr);

    return 0;
}
