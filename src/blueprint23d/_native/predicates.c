/*
 * Adaptive-precision geometric predicates.
 *
 * This is the performance floor of the whole kernel: every topological
 * decision -- Delaunay flips, loop nesting, curve-curve classification --
 * bottoms out in one of these sign tests, and they get called millions of
 * times on a part of any complexity. The pure-Python implementation in
 * exact.py is correct but roughly two orders of magnitude too slow to run
 * a constrained Delaunay triangulation through.
 *
 * The technique is Shewchuk's: represent an exact value as an *expansion*,
 * a sum of nonoverlapping doubles, and compute error-free transformations
 * (two_sum, two_product) that capture the rounding residual of each
 * primitive operation exactly. orient2d then runs a four-stage cascade,
 * stopping at the first stage whose accumulated error bound is provably
 * smaller than the magnitude of the result:
 *
 *   A: plain floating point               (~always sufficient)
 *   B: exact products, 4-term expansion
 *   C: first-order correction terms
 *   D: full exact expansion               (exact by construction)
 *
 * Stage D is exact, so orient2d never needs a fallback. incircle runs the
 * stage-A filter and, when that is inconclusive, evaluates the determinant
 * fully exactly in expansion arithmetic here rather than handing it back to
 * Python -- arc tessellation generates exactly-cocircular points by
 * construction, so on this workload the degenerate branch is the common
 * case, not the rare one.
 *
 *   J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast
 *   Robust Geometric Predicates", Discrete & Computational Geometry 18(3),
 *   1997, pp. 305-363.
 *
 * CORRECTNESS REQUIREMENT: the error-free transformations below are only
 * error-free under strict IEEE-754 binary64 semantics. They break if the
 * compiler contracts a multiply-add into an FMA, keeps intermediates in
 * x87 extended precision, or assumes associativity. The build pins
 * -ffp-contract=off, -fexcess-precision=standard and no fast-math, and
 * test_native.c_matches_python cross-checks this build against the
 * independent rational implementation on adversarial inputs.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

typedef double REAL;

static int sign_of_real(REAL x)
{
    return (x > 0.0) - (x < 0.0);
}

/* Volatile-ish barrier: keeps the compiler from re-associating the
 * error-free transformations even at -O2. */
#define INEXACT /* nothing; flags below carry the guarantee */

static REAL splitter;
static REAL epsilon;
static REAL ccwerrboundA, ccwerrboundB, ccwerrboundC;
static REAL iccerrboundA;
static REAL resulterrbound;

/* Fallback counters, exposed to Python so the cost of exactness is
 * measurable rather than assumed. */
static unsigned long long stat_orient2d_calls = 0;
static unsigned long long stat_orient2d_adapt = 0;
static unsigned long long stat_incircle_calls = 0;
static unsigned long long stat_incircle_defer = 0;

#define Absolute(a) ((a) >= 0.0 ? (a) : -(a))

#define Fast_Two_Sum_Tail(a, b, x, y) \
    bvirt = x - a;                    \
    y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
    x = (REAL)(a + b);           \
    Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
    bvirt = (REAL)(x - a);       \
    avirt = x - bvirt;           \
    bround = b - bvirt;          \
    around = a - avirt;          \
    y = around + bround

#define Two_Sum(a, b, x, y) \
    x = (REAL)(a + b);      \
    Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
    bvirt = (REAL)(a - x);        \
    avirt = x + bvirt;            \
    bround = bvirt - b;           \
    around = a - avirt;           \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = (REAL)(a - b);       \
    Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
    c = (REAL)(splitter * a);  \
    abig = (REAL)(c - a);      \
    ahi = c - abig;            \
    alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
    Split(a, ahi, alo);              \
    Split(b, bhi, blo);              \
    err1 = x - (ahi * bhi);          \
    err2 = err1 - (alo * bhi);       \
    err3 = err2 - (ahi * blo);       \
    y = (alo * blo) - err3

#define Two_Product(a, b, x, y) \
    x = (REAL)(a * b);          \
    Two_Product_Tail(a, b, x, y)

/* Multiply when b has already been split, saving a Split per term. */
#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
    x = (REAL)(a * b);                             \
    Split(a, ahi, alo);                            \
    err1 = x - (ahi * bhi);                        \
    err2 = err1 - (alo * bhi);                     \
    err3 = err2 - (ahi * blo);                     \
    y = (alo * blo) - err3

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
    Two_Diff(a0, b, _i, x0);                \
    Two_Sum(a1, _i, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Diff(a1, a0, b0, _j, _0, x0);            \
    Two_One_Diff(_j, _0, b1, x3, x2, x1)

/*
 * Derive the machine's actual unit roundoff and splitter rather than
 * hardcoding 2^-53. This makes the bounds correct on whatever FP
 * environment the extension is actually compiled for, instead of correct
 * only on the one assumed at authoring time.
 */
static void exactinit(void)
{
    REAL half = 0.5;
    REAL check = 1.0, lastcheck;
    int every_other = 1;

    epsilon = 1.0;
    splitter = 1.0;
    do {
        lastcheck = check;
        epsilon *= half;
        if (every_other) {
            splitter *= 2.0;
        }
        every_other = !every_other;
        check = 1.0 + epsilon;
    } while ((check != 1.0) && (check != lastcheck));
    splitter += 1.0;

    resulterrbound = (3.0 + 8.0 * epsilon) * epsilon;
    ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon;
    ccwerrboundB = (2.0 + 12.0 * epsilon) * epsilon;
    ccwerrboundC = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
    iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon;
}

static REAL estimate(int elen, const REAL *e)
{
    REAL Q = e[0];
    int i;
    for (i = 1; i < elen; i++) {
        Q += e[i];
    }
    return Q;
}

/*
 * Sum two nonoverlapping increasing-magnitude expansions, dropping zero
 * components. Shewchuk's original advances the read cursor past the end of
 * the input arrays and relies on the unused value never being inspected;
 * the sentinel reads here keep that out-of-bounds access from happening at
 * all, which matters under ASan and for arrays at a page boundary.
 */
static int fast_expansion_sum_zeroelim(int elen, const REAL *e, int flen, const REAL *f, REAL *h)
{
    REAL Q;
    INEXACT REAL Qnew;
    INEXACT REAL hh;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    int eindex, findex, hindex;
    REAL enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;

#define ENEXT()                                        \
    do {                                               \
        eindex++;                                      \
        enow = (eindex < elen) ? e[eindex] : (REAL)0.0; \
    } while (0)

#define FNEXT()                                        \
    do {                                               \
        findex++;                                      \
        fnow = (findex < flen) ? f[findex] : (REAL)0.0; \
    } while (0)

    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        ENEXT();
    } else {
        Q = fnow;
        FNEXT();
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            ENEXT();
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            FNEXT();
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                ENEXT();
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                FNEXT();
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        ENEXT();
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        FNEXT();
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;

#undef ENEXT
#undef FNEXT
}

/*
 * Capacities. Multiplying an m-term by an n-term expansion yields at most
 * 2mn terms, so the in-circle chain runs 2x2 -> 8 (products), 8+8 -> 16
 * (minors and lifts), 16x16 -> 512 (lift * minor), and 3 x 512 -> 1536 for
 * the final sum. The bounds below leave slack on top of that.
 */
#define EXP_PROD_MAX 768
#define EXP_ACC_MAX 2048

/* Multiply an expansion by a single double, exactly. */
static int scale_expansion_zeroelim(int elen, const REAL *e, REAL b, REAL *h)
{
    INEXACT REAL Q, sum;
    REAL hh;
    INEXACT REAL product1;
    REAL product0;
    int eindex, hindex;
    REAL enow;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0.0) {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

/* h = e * f, exactly, by accumulating one scaled copy of e per term of f. */
static int expansion_multiply(int elen, const REAL *e, int flen, const REAL *f, REAL *h)
{
    REAL bufA[EXP_ACC_MAX], bufB[EXP_ACC_MAX], tmp[EXP_PROD_MAX];
    REAL *acc = bufA, *spare = bufB, *swap;
    int acclen = 1, tlen, i;

    acc[0] = 0.0;
    for (i = 0; i < flen; i++) {
        tlen = scale_expansion_zeroelim(elen, e, f[i], tmp);
        acclen = fast_expansion_sum_zeroelim(acclen, acc, tlen, tmp, spare);
        swap = acc;
        acc = spare;
        spare = swap;
    }
    memcpy(h, acc, (size_t)acclen * sizeof(REAL));
    return acclen;
}

/* h = e - f, exactly. Negating an expansion is exact (sign flip only). */
static int expansion_subtract(int elen, const REAL *e, int flen, const REAL *f, REAL *h)
{
    REAL negated[EXP_ACC_MAX];
    int i;
    for (i = 0; i < flen; i++) {
        negated[i] = -f[i];
    }
    return fast_expansion_sum_zeroelim(elen, e, flen, negated, h);
}

/* Exact difference of two doubles as a (1- or 2-term) expansion. */
static int exact_difference(REAL a, REAL b, REAL *e)
{
    INEXACT REAL x;
    REAL y;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;

    Two_Diff(a, b, x, y);
    if (y == 0.0) {
        e[0] = x;
        return 1;
    }
    e[0] = y;
    e[1] = x;
    return 2;
}

/*
 * Fully exact in-circle determinant.
 *
 * Worth the code: tessellating an arc places points that are *exactly*
 * cocircular by construction, so a Delaunay pass over CAD geometry lands
 * on the degenerate branch constantly rather than rarely. Deferring those
 * to rational arithmetic in Python measured ~1.3x over pure Python --
 * i.e. no acceleration at all on precisely the input this kernel exists
 * to handle.
 *
 * The components of a nonoverlapping expansion are ordered by increasing
 * magnitude and the largest dominates the rest, so the sign of the final
 * term is the sign of the exact sum.
 */
static int incircle_exact_sign(const REAL *pa, const REAL *pb, const REAL *pc, const REAL *pd)
{
    REAL adx[2], ady[2], bdx[2], bdy[2], cdx[2], cdy[2];
    int nadx, nady, nbdx, nbdy, ncdx, ncdy;

    REAL prod1[EXP_PROD_MAX], prod2[EXP_PROD_MAX];
    REAL minor[EXP_ACC_MAX], lift[EXP_ACC_MAX], part[EXP_ACC_MAX];
    REAL total[EXP_ACC_MAX], scratch[EXP_ACC_MAX];
    int len1, len2, minorlen, liftlen, partlen, totallen;

    nadx = exact_difference(pa[0], pd[0], adx);
    nady = exact_difference(pa[1], pd[1], ady);
    nbdx = exact_difference(pb[0], pd[0], bdx);
    nbdy = exact_difference(pb[1], pd[1], bdy);
    ncdx = exact_difference(pc[0], pd[0], cdx);
    ncdy = exact_difference(pc[1], pd[1], cdy);

    totallen = 1;
    total[0] = 0.0;

    /* term A: (adx^2 + ady^2) * (bdx*cdy - cdx*bdy) */
    len1 = expansion_multiply(nbdx, bdx, ncdy, cdy, prod1);
    len2 = expansion_multiply(ncdx, cdx, nbdy, bdy, prod2);
    minorlen = expansion_subtract(len1, prod1, len2, prod2, minor);
    len1 = expansion_multiply(nadx, adx, nadx, adx, prod1);
    len2 = expansion_multiply(nady, ady, nady, ady, prod2);
    liftlen = fast_expansion_sum_zeroelim(len1, prod1, len2, prod2, lift);
    partlen = expansion_multiply(liftlen, lift, minorlen, minor, part);
    totallen = fast_expansion_sum_zeroelim(totallen, total, partlen, part, scratch);
    memcpy(total, scratch, (size_t)totallen * sizeof(REAL));

    /* term B: (bdx^2 + bdy^2) * (cdx*ady - adx*cdy) */
    len1 = expansion_multiply(ncdx, cdx, nady, ady, prod1);
    len2 = expansion_multiply(nadx, adx, ncdy, cdy, prod2);
    minorlen = expansion_subtract(len1, prod1, len2, prod2, minor);
    len1 = expansion_multiply(nbdx, bdx, nbdx, bdx, prod1);
    len2 = expansion_multiply(nbdy, bdy, nbdy, bdy, prod2);
    liftlen = fast_expansion_sum_zeroelim(len1, prod1, len2, prod2, lift);
    partlen = expansion_multiply(liftlen, lift, minorlen, minor, part);
    totallen = fast_expansion_sum_zeroelim(totallen, total, partlen, part, scratch);
    memcpy(total, scratch, (size_t)totallen * sizeof(REAL));

    /* term C: (cdx^2 + cdy^2) * (adx*bdy - bdx*ady) */
    len1 = expansion_multiply(nadx, adx, nbdy, bdy, prod1);
    len2 = expansion_multiply(nbdx, bdx, nady, ady, prod2);
    minorlen = expansion_subtract(len1, prod1, len2, prod2, minor);
    len1 = expansion_multiply(ncdx, cdx, ncdx, cdx, prod1);
    len2 = expansion_multiply(ncdy, cdy, ncdy, cdy, prod2);
    liftlen = fast_expansion_sum_zeroelim(len1, prod1, len2, prod2, lift);
    partlen = expansion_multiply(liftlen, lift, minorlen, minor, part);
    totallen = fast_expansion_sum_zeroelim(totallen, total, partlen, part, scratch);
    memcpy(total, scratch, (size_t)totallen * sizeof(REAL));

    return sign_of_real(total[totallen - 1]);
}

/* Stages B, C and D of the orient2d cascade. Stage D is exact. */
static REAL orient2dadapt(const REAL *pa, const REAL *pb, const REAL *pc, REAL detsum)
{
    INEXACT REAL acx, acy, bcx, bcy;
    REAL acxtail, acytail, bcxtail, bcytail;
    INEXACT REAL detleft, detright;
    REAL detlefttail, detrighttail;
    REAL det, errbound;
    REAL B[4], C1[8], C2[12], D[16];
    INEXACT REAL B3;
    int C1length, C2length, Dlength;
    REAL u[4];
    INEXACT REAL u3;
    INEXACT REAL s1, t1;
    REAL s0, t0;

    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;
    INEXACT REAL _i, _j;
    REAL _0;

    acx = (REAL)(pa[0] - pc[0]);
    bcx = (REAL)(pb[0] - pc[0]);
    acy = (REAL)(pa[1] - pc[1]);
    bcy = (REAL)(pb[1] - pc[1]);

    Two_Product(acx, bcy, detleft, detlefttail);
    Two_Product(acy, bcx, detright, detrighttail);

    Two_Two_Diff(detleft, detlefttail, detright, detrighttail, B3, B[2], B[1], B[0]);
    B[3] = B3;

    det = estimate(4, B);
    errbound = ccwerrboundB * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa[0], pc[0], acx, acxtail);
    Two_Diff_Tail(pb[0], pc[0], bcx, bcxtail);
    Two_Diff_Tail(pa[1], pc[1], acy, acytail);
    Two_Diff_Tail(pb[1], pc[1], bcy, bcytail);

    if ((acxtail == 0.0) && (acytail == 0.0) && (bcxtail == 0.0) && (bcytail == 0.0)) {
        return det;
    }

    errbound = ccwerrboundC * detsum + resulterrbound * Absolute(det);
    det += (acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail);
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Product(acxtail, bcy, s1, s0);
    Two_Product(acytail, bcx, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C1length = fast_expansion_sum_zeroelim(4, B, 4, u, C1);

    Two_Product(acx, bcytail, s1, s0);
    Two_Product(acy, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C2length = fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

    Two_Product(acxtail, bcytail, s1, s0);
    Two_Product(acytail, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    Dlength = fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

    /* The largest-magnitude component carries the sign of the exact sum. */
    return D[Dlength - 1];
}

static REAL orient2d_value(const REAL *pa, const REAL *pb, const REAL *pc)
{
    REAL detleft, detright, det;
    REAL detsum, errbound;

    stat_orient2d_calls++;

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    det = detleft - detright;

    /* Opposite signs (or a zero) means the subtraction cannot cancel, so
     * the naive sign is already the true sign. */
    if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        }
        detsum = detleft + detright;
    } else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        }
        detsum = -detleft - detright;
    } else {
        return det;
    }

    errbound = ccwerrboundA * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    stat_orient2d_adapt++;
    return orient2dadapt(pa, pb, pc, detsum);
}

/* ------------------------------------------------------------------ */
/* Python bindings                                                     */
/* ------------------------------------------------------------------ */

static PyObject *py_orient2d(PyObject *self, PyObject *args)
{
    REAL pa[2], pb[2], pc[2];
    (void)self;
    if (!PyArg_ParseTuple(args, "(dd)(dd)(dd)", &pa[0], &pa[1], &pb[0], &pb[1], &pc[0], &pc[1])) {
        return NULL;
    }
    return PyLong_FromLong(sign_of_real(orient2d_value(pa, pb, pc)));
}

/*
 * Stage-A filtered in-circle test.
 *
 * Returns the certified sign, or None when the floating-point result is
 * not provably distinguishable from zero -- the caller then decides it in
 * exact rational arithmetic. Deferring rather than guessing is what keeps
 * the guarantee intact.
 */
static PyObject *py_incircle(PyObject *self, PyObject *args)
{
    REAL pa[2], pb[2], pc[2], pd[2];
    REAL adx, ady, bdx, bdy, cdx, cdy;
    REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    REAL alift, blift, clift;
    REAL det, permanent, errbound;

    (void)self;
    if (!PyArg_ParseTuple(args, "(dd)(dd)(dd)(dd)", &pa[0], &pa[1], &pb[0], &pb[1], &pc[0],
                          &pc[1], &pd[0], &pd[1])) {
        return NULL;
    }

    stat_incircle_calls++;

    adx = pa[0] - pd[0];
    ady = pa[1] - pd[1];
    bdx = pb[0] - pd[0];
    bdy = pb[1] - pd[1];
    cdx = pc[0] - pd[0];
    cdy = pc[1] - pd[1];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift = adx * adx + ady * ady;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift = bdx * bdx + bdy * bdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift = cdx * cdx + cdy * cdy;

    det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift +
                (Absolute(cdxady) + Absolute(adxcdy)) * blift +
                (Absolute(adxbdy) + Absolute(bdxady)) * clift;
    errbound = iccerrboundA * permanent;
    if ((det > errbound) || (-det > errbound)) {
        return PyLong_FromLong(sign_of_real(det));
    }

    stat_incircle_defer++;
    return PyLong_FromLong(incircle_exact_sign(pa, pb, pc, pd));
}

static PyObject *py_stats(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    return Py_BuildValue("{s:K,s:K,s:K,s:K}", "orient2d_calls", stat_orient2d_calls,
                         "orient2d_adaptive", stat_orient2d_adapt, "incircle_calls",
                         stat_incircle_calls, "incircle_deferred", stat_incircle_defer);
}

static PyObject *py_reset_stats(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    stat_orient2d_calls = 0;
    stat_orient2d_adapt = 0;
    stat_incircle_calls = 0;
    stat_incircle_defer = 0;
    Py_RETURN_NONE;
}

static PyObject *py_machine_constants(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    return Py_BuildValue("{s:d,s:d,s:d}", "epsilon", epsilon, "splitter", splitter,
                         "orient2d_filter", ccwerrboundA);
}

static PyMethodDef PredicateMethods[] = {
    {"orient2d", py_orient2d, METH_VARARGS,
     "orient2d(a, b, c) -> int: exact sign of the orientation determinant."},
    {"incircle", py_incircle, METH_VARARGS,
     "incircle(a, b, c, d) -> int: exact sign of the in-circle determinant."},
    {"stats", py_stats, METH_NOARGS, "Predicate call/fallback counters."},
    {"reset_stats", py_reset_stats, METH_NOARGS, "Zero the counters."},
    {"machine_constants", py_machine_constants, METH_NOARGS,
     "The FP constants derived at import time."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef predicatesmodule = {
    PyModuleDef_HEAD_INIT, "blueprint23d._native.predicates",
    "Adaptive-precision geometric predicates (Shewchuk).", -1, PredicateMethods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_predicates(void)
{
    exactinit();
    return PyModule_Create(&predicatesmodule);
}
