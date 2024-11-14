#include "cudd_extensions.h"

#include <cuddInt.h>

DdNode *cupaal::Cudd_addExp(DdManager *dd, const DdNode *f) {
    if (cuddIsConstant(f)) {
        const CUDD_VALUE_TYPE value = exp(cuddV(f));
        DdNode *res = cuddUniqueConst(dd, value);
        return (res);
    }
    return nullptr;
} /* end of Cudd_addExp */
