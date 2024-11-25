#ifndef CUDD_EXTENSIONS_H
#define CUDD_EXTENSIONS_H
#include <cuddObj.hh>

namespace cupaal {
    extern DdNode *Cudd_addExp(DdManager *dd, DdNode *f);

    extern DdNode *Cudd_addLogMatrixMultiply(DdManager *dd, DdNode *A, DdNode *B, DdNode **z, int nz);

    extern CUDD_VALUE_TYPE log_add(CUDD_VALUE_TYPE x, CUDD_VALUE_TYPE y);

    extern DdNode *Cudd_addLogPlus(DdManager *dd, DdNode **f, DdNode **g);

    extern DdNode *addLogMMRecur(DdManager *dd, DdNode *A, DdNode *B, int topP, int *vars);
}

#endif //CUDD_EXTENSIONS_H
