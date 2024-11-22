#include "cudd_extensions.h"

#include <util.h>
#include <cuddInt.h>

/**
 * @brief Natural exponent of f an %ADD.
 * @return nullptr if not a terminal case; exp(f) otherwise.
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

CUDD_VALUE_TYPE log_add(CUDD_VALUE_TYPE x, CUDD_VALUE_TYPE y) {
    if (y > x) {
        CUDD_VALUE_TYPE temp = x;
        x = y;
        y = temp;
    }
    if (x == -std::numeric_limits<double>::infinity()) {
        return x;
    }
    return x + log1p(exp(y - x));
}

/**
 *  @brief floating point addition for numbers in the log semiring.
 *  @return nullptr if not a terminal case; log(exp(f) * exp(g)) otherwise.
 *  @sideeffect None
 *  @see Cudd_addApply
*/
DdNode *
Cudd_addLogPlus(
  DdManager * dd,
  DdNode ** f,
  DdNode ** g)
{
    DdNode *res;
    DdNode *F, *G;
    CUDD_VALUE_TYPE value;

    F = *f; G = *g;
    if (cuddIsConstant(F) && cuddV(F) == -std::numeric_limits<double>::infinity()) return(F);
    if (cuddIsConstant(G) && cuddV(G) == -std::numeric_limits<double>::infinity()) return(G);
    if (cuddIsConstant(F) && cuddIsConstant(G)) {
        value = log_add(cuddV(F), cuddV(G));
        res = cuddUniqueConst(dd,value);
        return(res);
    }
    if (F > G) { /* swap f and g */
        *f = G;
        *g = F;
    }
    return(nullptr);

} /* end of Cudd_addPlus */

DdNode *cupaal::Cudd_addLogMatrixMultiply(DdManager *dd, DdNode *A, DdNode *B, DdNode **z, int nz) {
    int i, nvars, *vars;
    DdNode *res;
    nvars = dd->size;
    vars = ALLOC(int, nvars);
    if (vars == nullptr) {
        dd->errorCode = CUDD_MEMORY_OUT;
        return (nullptr);
    }
    for (i = 0; i < nvars; i++) {
        vars[i] = 0;
    }
    for (i = 0; i < nz; i++) {
        vars[z[i]->index] = 1;
    }

    do {
        dd->reordered = 0;
        res = addLogMMRecur(dd, A, B, -1, vars);
    } while (dd->reordered == 1);
    FREE(vars);
    if (dd->errorCode == CUDD_TIMEOUT_EXPIRED && dd->timeoutHandler) {
        dd->timeoutHandler(dd, dd->tohArg);
    }
    return (res);
}


DdNode *
addLogMMRecur(
    DdManager *dd,
    DdNode *A,
    DdNode *B,
    int topP,
    int *vars) {
    DdNode *zero,
            *At, /* positive cofactor of first operand */
            *Ae, /* negative cofactor of first operand */
            *Bt, /* positive cofactor of second operand */
            *Be, /* negative cofactor of second operand */
            *t, /* positive cofactor of result */
            *e, /* negative cofactor of result */
            *scaled, /* scaled result */
            *add_scale, /* ADD representing the scaling factor */
            *res;
    int i; /* loop index */
    double scale; /* scaling factor */
    int index; /* index of the top variable */
    CUDD_VALUE_TYPE value;
    int topA, topB, topV;
    DD_CTFP cacheOp;

    statLine(dd);
    zero = cuddUniqueConst(dd, -std::numeric_limits<double>::infinity());
    cuddRef(zero);

    if (A == zero || B == zero) {
        return (zero);
    }
    if (cuddIsConstant(A) && cuddIsConstant(B)) {
        /* Compute the scaling factor. It is 2^k, where k is the
         * number of summation variables below the current variable.
         * Indeed, these constants represent blocks of 2^k identical
         * constant values in both A and B.
         */

        value = cuddV(A) + cuddV(B);
        for (i = 0; i < dd->size; i++) {
            if (vars[i]) {
                if (dd->perm[i] > topP) {
                    value += (CUDD_VALUE_TYPE) log(2);
                }
            }
        }
        res = cuddUniqueConst(dd, value);
        return (res);
    }
    /* Standardize to increase cache efficiency. Clearly, A*B != B*A
    ** in matrix multiplication. However, which matrix is which is
    ** determined by the variables appearing in the ADDs and not by
    ** which one is passed as first argument.
    */
    if (A > B) {
        DdNode *tmp = A;
        A = B;
        B = tmp;
    }

    topA = cuddI(dd, A->index);
    topB = cuddI(dd, B->index);
    topV = ddMin(topA, topB);

    cacheOp = (DD_CTFP) addLogMMRecur;
    res = cuddCacheLookup2(dd, cacheOp, A, B);
    if (res != nullptr) {
        /* If the result is 0, there is no need to normalize.
        ** Otherwise we count the number of z variables between
        ** the current depth and the top of the ADDs. These are
        ** the missing variables that determine the size of the
        ** constant blocks.
        */
        if (res == zero) return (res);
        scale = 0.0;
        for (i = 0; i < dd->size; i++) {
            if (vars[i]) {
                if (dd->perm[i] > topP && dd->perm[i] < topV) {
                    scale += log(2);
                }
            }
        }
        if (scale > 0.0) {
            cuddRef(res);
            add_scale = cuddUniqueConst(dd, (CUDD_VALUE_TYPE) scale);
            if (add_scale == nullptr) {
                Cudd_RecursiveDeref(dd, res);
                return (nullptr);
            }
            cuddRef(add_scale);
            scaled = cuddAddApplyRecur(dd, Cudd_addPlus, res, add_scale);
            if (scaled == nullptr) {
                Cudd_RecursiveDeref(dd, add_scale);
                Cudd_RecursiveDeref(dd, res);
                return (nullptr);
            }
            cuddRef(scaled);
            Cudd_RecursiveDeref(dd, add_scale);
            Cudd_RecursiveDeref(dd, res);
            res = scaled;
            cuddDeref(res);
        }
        return (res);
    }

    checkWhetherToGiveUp(dd);

    /* compute the cofactors */
    if (topV == topA) {
        At = cuddT(A);
        Ae = cuddE(A);
    } else {
        At = Ae = A;
    }
    if (topV == topB) {
        Bt = cuddT(B);
        Be = cuddE(B);
    } else {
        Bt = Be = B;
    }

    t = addLogMMRecur(dd, At, Bt, (int) topV, vars);
    if (t == nullptr) return (nullptr);
    cuddRef(t);
    e = addLogMMRecur(dd, Ae, Be, (int) topV, vars);
    if (e == nullptr) {
        Cudd_RecursiveDeref(dd, t);
        return (nullptr);
    }
    cuddRef(e);

    index = dd->invperm[topV];
    if (vars[index] == 0) {
        /* We have split on either the rows of A or the columns
        ** of B. We just need to connect the two subresults,
        ** which correspond to two submatrices of the result.
        */
        res = (t == e) ? t : cuddUniqueInter(dd, index, t, e);
        if (res == nullptr) {
            Cudd_RecursiveDeref(dd, t);
            Cudd_RecursiveDeref(dd, e);
            return (nullptr);
        }
        cuddRef(res);
        cuddDeref(t);
        cuddDeref(e);
    } else {
        /* we have simultaneously split on the columns of A and
        ** the rows of B. The two subresults must be added.
        */
        res = cuddAddApplyRecur(dd, Cudd_addLogPlus, t, e);
        if (res == nullptr) {
            Cudd_RecursiveDeref(dd, t);
            Cudd_RecursiveDeref(dd, e);
            return (nullptr);
        }
        cuddRef(res);
        Cudd_RecursiveDeref(dd, t);
        Cudd_RecursiveDeref(dd, e);
    }

    cuddCacheInsert2(dd, cacheOp, A, B, res);

    /* We have computed (and stored in the computed table) a minimal
    ** result; that is, a result that assumes no summation variables
    ** between the current depth of the recursion and its top
    ** variable. We now take into account the z variables by properly
    ** scaling the result.
    */
    if (res != zero) {
        scale = 0.0;
        for (i = 0; i < dd->size; i++) {
            if (vars[i]) {
                if (dd->perm[i] > topP && dd->perm[i] < topV) {
                    scale += log(2);
                }
            }
        }
        if (scale > 0.0) {
            add_scale = cuddUniqueConst(dd, (CUDD_VALUE_TYPE) scale);
            if (add_scale == nullptr) {
                Cudd_RecursiveDeref(dd, res);
                return (nullptr);
            }
            cuddRef(add_scale);
            scaled = cuddAddApplyRecur(dd, Cudd_addPlus, res, add_scale);
            if (scaled == nullptr) {
                Cudd_RecursiveDeref(dd, res);
                Cudd_RecursiveDeref(dd, add_scale);
                return (nullptr);
            }
            cuddRef(scaled);
            Cudd_RecursiveDeref(dd, add_scale);
            Cudd_RecursiveDeref(dd, res);
            res = scaled;
        }
    }
    cuddDeref(res);
    return (res);
} /* end of addLogMMRecur */
