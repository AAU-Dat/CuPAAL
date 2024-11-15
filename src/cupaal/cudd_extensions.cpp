#include "cudd_extensions.h"

#include <cuddInt.h>

/**
 * @brief Natural exponent of f an %ADD.
 * @return NULL if not a terminal case; exp(f) otherwise.
 * @sideeffect None
 * @see Cudd_addMonadicApply
*/
DdNode *cupaal::Cudd_addExp(DdManager *dd, DdNode *f) {
    if (cuddIsConstant(f)) {
        const CUDD_VALUE_TYPE value = exp(cuddV(f));
        DdNode *res = cuddUniqueConst(dd, value);
        return (res);
    }
    return nullptr;
} /* end of Cudd_addExp */

