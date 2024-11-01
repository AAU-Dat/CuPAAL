//
// Created by Runge on 13/10/2024.
//

#ifndef BAUM_H
#define BAUM_H
#include <memory>
#include <string>
#include <storm/models/symbolic/Model.h>

namespace cupaal {
    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> parseAndBuildPrism(std::string const & filename);

    DdNode * cuddTestTwo(DdManager * dd);

}

#endif //BAUM_H
