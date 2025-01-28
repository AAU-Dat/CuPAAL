#ifndef CUDD_EXTENSIONS_H
#define CUDD_EXTENSIONS_H
#include <cuddObj.hh>

namespace cupaal {
    extern DdNode *Cudd_addExp(DdManager *dd, DdNode *f);

    extern DdNode *addLog(DdManager *dd, DdNode *f);

    extern DdNode *Cudd_addLogMatrixMultiply(DdManager *dd, DdNode *A, DdNode *B, DdNode **z, int nz);

    extern CUDD_VALUE_TYPE log_add(CUDD_VALUE_TYPE x, CUDD_VALUE_TYPE y);

    extern DdNode *Cudd_addLogPlus(DdManager *dd, DdNode **f, DdNode **g);

    extern DdNode *addLogMMRecur(DdManager *dd, DdNode *A, DdNode *B, int topP, int *vars);

    extern int Sudd_addRead(
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
    );
}

#endif //CUDD_EXTENSIONS_H
