#ifndef CUDD_EXTENSIONS_H
#define CUDD_EXTENSIONS_H
#include <cuddObj.hh>

namespace cupaal {
    extern DdNode * Cudd_addExp(DdManager *dd, const DdNode *f);
}

#endif //CUDD_EXTENSIONS_H
