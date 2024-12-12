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

CUDD_VALUE_TYPE cupaal::log_add(CUDD_VALUE_TYPE x, CUDD_VALUE_TYPE y) {
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
cupaal::Cudd_addLogPlus(
    DdManager *dd,
    DdNode **f,
    DdNode **g) {
    DdNode *res;
    DdNode *F, *G;
    CUDD_VALUE_TYPE value;

    F = *f;
    G = *g;
    if (cuddIsConstant(F) && cuddV(F) == -std::numeric_limits<double>::infinity()) return (F);
    if (cuddIsConstant(G) && cuddV(G) == -std::numeric_limits<double>::infinity()) return (G);
    if (cuddIsConstant(F) && cuddIsConstant(G)) {
        value = log_add(cuddV(F), cuddV(G));
        res = cuddUniqueConst(dd, value);
        return (res);
    }
    if (F > G) {
        /* swap f and g */
        *f = G;
        *g = F;
    }
    return (nullptr);
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
cupaal::addLogMMRecur(
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

int cupaal::Sudd_addRead(
    CUDD_VALUE_TYPE *array,
    ssize_t array_n_rows,
    ssize_t array_n_cols,
    DdManager *dd,
    DdNode **E /**< characteristic function of the graph */,
    DdNode ***x /**< array of row variables */,
    DdNode ***y /**< array of column variables */,
    DdNode ***xn /**< array of complemented row variables */,
    DdNode ***yn_ /**< array of complemented column variables */,
    int *nx /**< number or row variables */,
    int *ny /**< number or column variables */,
    int *m /**< number of rows */,
    int *n /**< number of columns */,
    int bx /**< first index of row variables */,
    int sx /**< step of row variables */,
    int by /**< first index of column variables */,
    int sy /**< step of column variables */
) {
    DdNode *one, *zero;
    DdNode *w, *neW;
    DdNode *minterm1;
    int u, v, err, i, nv;
    int lnx, lny;
    CUDD_VALUE_TYPE val;
    DdNode **lx, **ly, **lxn, **lyn;

    one = DD_ONE(dd);
    zero = DD_ZERO(dd);

    u = array_n_rows;
    v = array_n_cols;

    *m = u;
    /* Compute the number of x variables. */
    lx = *x;
    lxn = *xn;
    u--; /* row and column numbers start from 0 */
    for (lnx = 0; u > 0; lnx++) {
        u >>= 1;
    }
    /* Here we rely on the fact that REALLOC of a null pointer is
    ** translates to an ALLOC.
    */
    if (lnx > *nx) {
        *x = lx = REALLOC(DdNode *, *x, lnx);
        if (lx == NULL) {
            dd->errorCode = CUDD_MEMORY_OUT;
            return (0);
        }
        *xn = lxn = REALLOC(DdNode *, *xn, lnx);
        if (lxn == NULL) {
            dd->errorCode = CUDD_MEMORY_OUT;
            return (0);
        }
    }

    *n = v;
    /* Compute the number of y variables. */
    ly = *y;
    lyn = *yn_;
    v--; /* row and column numbers start from 0 */
    for (lny = 0; v > 0; lny++) {
        v >>= 1;
    }
    /* Here we rely on the fact that REALLOC of a null pointer is
    ** translates to an ALLOC.
    */
    if (lny > *ny) {
        *y = ly = REALLOC(DdNode *, *y, lny);
        if (ly == NULL) {
            dd->errorCode = CUDD_MEMORY_OUT;
            return (0);
        }
        *yn_ = lyn = REALLOC(DdNode *, *yn_, lny);
        if (lyn == NULL) {
            dd->errorCode = CUDD_MEMORY_OUT;
            return (0);
        }
    }

    /* Create all new variables. */
    for (i = *nx, nv = bx + (*nx) * sx; i < lnx; i++, nv += sx) {
        do {
            dd->reordered = 0;
            lx[i] = cuddUniqueInter(dd, nv, one, zero);
        } while (dd->reordered == 1);
        if (lx[i] == NULL) {
            if (dd->errorCode == CUDD_TIMEOUT_EXPIRED && dd->timeoutHandler) {
                dd->timeoutHandler(dd, dd->tohArg);
            }
            return (0);
        }
        cuddRef(lx[i]);
        do {
            dd->reordered = 0;
            lxn[i] = cuddUniqueInter(dd, nv, zero, one);
        } while (dd->reordered == 1);
        if (lxn[i] == NULL) {
            if (dd->errorCode == CUDD_TIMEOUT_EXPIRED && dd->timeoutHandler) {
                dd->timeoutHandler(dd, dd->tohArg);
            }
            return (0);
        }
        cuddRef(lxn[i]);
    }
    for (i = *ny, nv = by + (*ny) * sy; i < lny; i++, nv += sy) {
        do {
            dd->reordered = 0;
            ly[i] = cuddUniqueInter(dd, nv, one, zero);
        } while (dd->reordered == 1);
        if (ly[i] == NULL) {
            if (dd->errorCode == CUDD_TIMEOUT_EXPIRED && dd->timeoutHandler) {
                dd->timeoutHandler(dd, dd->tohArg);
            }
            return (0);
        }
        cuddRef(ly[i]);
        do {
            dd->reordered = 0;
            lyn[i] = cuddUniqueInter(dd, nv, zero, one);
        } while (dd->reordered == 1);
        if (lyn[i] == NULL) {
            if (dd->errorCode == CUDD_TIMEOUT_EXPIRED && dd->timeoutHandler) {
                dd->timeoutHandler(dd, dd->tohArg);
            }
            return (0);
        }
        cuddRef(lyn[i]);
    }
    *nx = lnx;
    *ny = lny;

    *E = dd->background; /* this call will never cause reordering */
    cuddRef(*E);

    for (int row = 0; row < array_n_rows; row++) {
        for (int col = 0; col < array_n_cols; col++) {
            val = array[row * array_n_cols + col];
            if (val == 0)
                continue;
            u = row;
            v = col;

            minterm1 = one;
            cuddRef(minterm1);

            /* Build minterm1 corresponding to this arc */
            for (i = lnx - 1; i >= 0; i--) {
                if (u & 1) {
                    w = Cudd_addApply(dd, Cudd_addTimes, minterm1, lx[i]);
                } else {
                    w = Cudd_addApply(dd, Cudd_addTimes, minterm1, lxn[i]);
                }
                if (w == NULL) {
                    Cudd_RecursiveDeref(dd, minterm1);
                    return (0);
                }
                cuddRef(w);
                Cudd_RecursiveDeref(dd, minterm1);
                minterm1 = w;
                u >>= 1;
            }
            for (i = lny - 1; i >= 0; i--) {
                if (v & 1) {
                    w = Cudd_addApply(dd, Cudd_addTimes, minterm1, ly[i]);
                } else {
                    w = Cudd_addApply(dd, Cudd_addTimes, minterm1, lyn[i]);
                }
                if (w == NULL) {
                    Cudd_RecursiveDeref(dd, minterm1);
                    return (0);
                }
                cuddRef(w);
                Cudd_RecursiveDeref(dd, minterm1);
                minterm1 = w;
                v >>= 1;
            }
            /* Create new constant node if necessary.
            ** This call will never cause reordering.
            */
            neW = cuddUniqueConst(dd, val);
            if (neW == NULL) {
                Cudd_RecursiveDeref(dd, minterm1);
                return (0);
            }
            cuddRef(neW);

            w = Cudd_addIte(dd, minterm1, neW, *E);
            if (w == NULL) {
                Cudd_RecursiveDeref(dd, minterm1);
                Cudd_RecursiveDeref(dd, neW);
                return (0);
            }
            cuddRef(w);
            Cudd_RecursiveDeref(dd, minterm1);
            Cudd_RecursiveDeref(dd, neW);
            Cudd_RecursiveDeref(dd, *E);
            *E = w;
        }
    }
    return (1);
}
