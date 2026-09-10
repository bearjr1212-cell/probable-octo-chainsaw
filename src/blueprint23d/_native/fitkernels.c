/*
 * Least-squares primitive fitting, and greedy segmentation of a point
 * chain into lines and arcs.
 *
 * This is what turns a scanned drawing back into CAD geometry. A contour
 * traced from pixels is a jagged polyline; a drilled hole in that polyline
 * is a lumpy blob with no radius. Fitting a circle to the points recovers
 * the radius the draughtsman actually specified, to far better precision
 * than the pixel grid, because a fit over N points averages down the
 * independent per-point error by roughly sqrt(N).
 *
 * Three pieces of numerics matter here, and the naive choice is wrong in
 * each case:
 *
 * 1. ORDINARY least squares on a line minimises *vertical* offsets, which
 *    is meaningless for a drawing: rotate the sheet and the answer
 *    changes, and a vertical edge has infinite slope. Total least squares
 *    minimises perpendicular distance instead, which is rotation
 *    invariant. For 2D it reduces to the principal eigenvector of the
 *    scatter matrix, available in closed form.
 *
 * 2. The obvious algebraic circle fit (minimising the residual of
 *    x^2+y^2+Dx+Ey+F) is heavily biased when the points span only a short
 *    arc -- exactly the case for a fillet -- because it weights points by
 *    their distance from the centre. Taubin's method normalises the
 *    constraint by the gradient of the algebraic form, which removes most
 *    of that bias and, unlike Kasa or Pratt, stays accurate on arcs of
 *    well under a quarter turn.
 *
 * 3. Taubin is still algebraic: it does not minimise the geometric
 *    (orthogonal) distance that actually defines a best-fit circle. So it
 *    is used as the starting point for Gauss-Newton iterations on the true
 *    residual sqrt((x-cx)^2 + (y-cy)^2) - r. Starting Gauss-Newton from a
 *    good algebraic estimate is what keeps it from diverging on short arcs.
 *
 * Segmentation is greedy: from each start point, extend a line as far as
 * the tolerance allows, extend an arc as far as the tolerance allows, and
 * keep whichever covers more points, preferring the line on a tie since a
 * straight edge is the more likely intent. Running it in C matters because
 * each extension re-evaluates the fit residual over the whole span, so the
 * work is quadratic in the span length with a small constant.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdlib.h>

typedef struct {
    double cx, cy, r;
    double rms, max_dev;
    double span;    /* angular extent covered by the points, radians */
    double sagitta; /* bulge of the arc away from its chord */
    int ok;
} CircleFit;

typedef struct {
    double px, py;   /* a point on the line */
    double dx, dy;   /* unit direction */
    double rms, max_dev;
    int ok;
} LineFit;

/* ---------------------------------------------------------------- */

/*
 * Total-least-squares line fit.
 *
 * The best line through a point set passes through the centroid along the
 * principal eigenvector of the scatter matrix. For 2x2 that eigenvector is
 * closed form, so there is no iteration and no degenerate slope case.
 */
static LineFit fit_line_tls(const double *xs, const double *ys, int n)
{
    LineFit fit;
    fit.ok = 0;
    fit.px = fit.py = fit.dx = fit.dy = fit.rms = fit.max_dev = 0.0;
    if (n < 2) {
        return fit;
    }

    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) {
        mx += xs[i];
        my += ys[i];
    }
    mx /= n;
    my /= n;

    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = xs[i] - mx;
        double dy = ys[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    /* Principal eigenvector of [[sxx, sxy], [sxy, syy]]. */
    double trace = sxx + syy;
    double diff = sxx - syy;
    double disc = sqrt(diff * diff + 4.0 * sxy * sxy);
    double lambda = 0.5 * (trace + disc);

    double vx, vy;
    if (fabs(sxy) > 1e-300) {
        vx = lambda - syy;
        vy = sxy;
    } else {
        /* Already axis aligned; pick the axis with the larger spread. */
        if (sxx >= syy) {
            vx = 1.0; vy = 0.0;
        } else {
            vx = 0.0; vy = 1.0;
        }
    }
    double norm = sqrt(vx * vx + vy * vy);
    if (norm <= 0.0) {
        return fit;
    }
    vx /= norm;
    vy /= norm;

    fit.px = mx;
    fit.py = my;
    fit.dx = vx;
    fit.dy = vy;

    double sum_sq = 0.0, worst = 0.0;
    for (int i = 0; i < n; i++) {
        /* Perpendicular distance to the fitted line. */
        double ex = xs[i] - mx;
        double ey = ys[i] - my;
        double perp = fabs(ex * (-vy) + ey * vx);
        sum_sq += perp * perp;
        if (perp > worst) {
            worst = perp;
        }
    }
    fit.rms = sqrt(sum_sq / n);
    fit.max_dev = worst;
    fit.ok = 1;
    return fit;
}

/*
 * Taubin algebraic circle fit.
 *
 * Solves the characteristic polynomial by Newton from zero, which for this
 * problem converges in a handful of steps and lands on the root of
 * interest. Coordinates are centred first so the moments stay
 * well-scaled -- a circle far from the origin otherwise loses precision in
 * the z = x^2 + y^2 terms.
 */
static CircleFit fit_circle_taubin(const double *xs, const double *ys, int n)
{
    CircleFit fit;
    fit.ok = 0;
    fit.cx = fit.cy = fit.r = fit.rms = fit.max_dev = 0.0;
    fit.span = fit.sagitta = 0.0;
    if (n < 3) {
        return fit;
    }

    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) {
        mx += xs[i];
        my += ys[i];
    }
    mx /= n;
    my /= n;

    double Mxx = 0, Myy = 0, Mxy = 0, Mxz = 0, Myz = 0, Mzz = 0, Mz = 0;
    for (int i = 0; i < n; i++) {
        double u = xs[i] - mx;
        double v = ys[i] - my;
        double z = u * u + v * v;
        Mxx += u * u;
        Myy += v * v;
        Mxy += u * v;
        Mxz += u * z;
        Myz += v * z;
        Mzz += z * z;
        Mz += z;
    }
    Mxx /= n; Myy /= n; Mxy /= n; Mxz /= n; Myz /= n; Mzz /= n; Mz /= n;

    double cov_xy = Mxx * Myy - Mxy * Mxy;
    double var_z = Mzz - Mz * Mz;
    double A3 = 4.0 * Mz;
    double A2 = -3.0 * Mz * Mz - Mzz;
    double A1 = var_z * Mz + 4.0 * cov_xy * Mz - Mxz * Mxz - Myz * Myz;
    double A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - var_z * cov_xy;
    double A22 = A2 + A2;
    double A33 = A3 + A3 + A3;

    double x = 0.0, y = A0;
    for (int iter = 0; iter < 100; iter++) {
        double dy = A1 + x * (A22 + A33 * x);
        if (dy == 0.0) {
            break;
        }
        double xnew = x - y / dy;
        if (xnew == x || !isfinite(xnew)) {
            break;
        }
        double ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3));
        if (fabs(ynew) >= fabs(y)) {
            break;
        }
        x = xnew;
        y = ynew;
    }

    double det = x * x - x * Mz + cov_xy;
    if (fabs(det) < 1e-300) {
        return fit; /* points are collinear: no circle through them */
    }
    double xc = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.0;
    double yc = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.0;

    fit.cx = xc + mx;
    fit.cy = yc + my;
    fit.r = sqrt(xc * xc + yc * yc + Mz);
    fit.ok = isfinite(fit.cx) && isfinite(fit.cy) && isfinite(fit.r) && fit.r > 0.0;
    return fit;
}

/*
 * Angular extent of the points about the fitted centre, and the resulting
 * sagitta.
 *
 * The sagitta is the conditioning number that matters for a circle fit. It
 * is how far the arc bulges away from its own chord, so it measures how
 * much evidence of curvature the data actually contains. When the sagitta
 * drops to the level of the point noise, the radius is unidentifiable --
 * which is exactly why a noisy straight edge can "fit" a circle of radius
 * 3000 and why that fit must be rejected rather than believed.
 *
 * The extent is found by sorting the angles and taking the complement of
 * the largest gap, which handles the wrap through zero without special
 * cases.
 */
static void circle_span(const double *xs, const double *ys, int n, CircleFit *fit)
{
    fit->span = 0.0;
    fit->sagitta = 0.0;
    if (n < 2 || !fit->ok) {
        return;
    }

    double *angles = (double *)malloc((size_t)n * sizeof(double));
    if (!angles) {
        return;
    }
    for (int i = 0; i < n; i++) {
        double a = atan2(ys[i] - fit->cy, xs[i] - fit->cx);
        angles[i] = (a < 0.0) ? a + 2.0 * M_PI : a;
    }
    /* Insertion sort: spans here are short and already nearly ordered. */
    for (int i = 1; i < n; i++) {
        double key = angles[i];
        int j = i - 1;
        while (j >= 0 && angles[j] > key) {
            angles[j + 1] = angles[j];
            j--;
        }
        angles[j + 1] = key;
    }
    double largest_gap = angles[0] + 2.0 * M_PI - angles[n - 1];
    for (int i = 1; i < n; i++) {
        double gap = angles[i] - angles[i - 1];
        if (gap > largest_gap) {
            largest_gap = gap;
        }
    }
    free(angles);

    double span = 2.0 * M_PI - largest_gap;
    if (span < 0.0) {
        span = 0.0;
    }
    fit->span = span;
    double half = span * 0.5;
    if (half >= M_PI) {
        fit->sagitta = 2.0 * fit->r; /* more than half a turn: full diameter */
    } else {
        fit->sagitta = fit->r * (1.0 - cos(half));
    }
}

static void circle_residuals(const double *xs, const double *ys, int n, CircleFit *fit)
{
    double sum_sq = 0.0, worst = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = xs[i] - fit->cx;
        double dy = ys[i] - fit->cy;
        double res = fabs(sqrt(dx * dx + dy * dy) - fit->r);
        sum_sq += res * res;
        if (res > worst) {
            worst = res;
        }
    }
    fit->rms = sqrt(sum_sq / n);
    fit->max_dev = worst;
    circle_span(xs, ys, n, fit);
}

/*
 * Gauss-Newton refinement of the true geometric residual.
 *
 * Taubin minimises an algebraic quantity; the actual definition of a best
 * fit circle minimises orthogonal distance. Three unknowns, so the normal
 * equations are a 3x3 solve per iteration, done here by Cramer's rule.
 */
static CircleFit refine_circle_geometric(const double *xs, const double *ys, int n, CircleFit start,
                                         int max_iter)
{
    CircleFit fit = start;
    if (!fit.ok || n < 3) {
        return fit;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        double jtj[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double jtr[3] = {0, 0, 0};

        for (int i = 0; i < n; i++) {
            double dx = xs[i] - fit.cx;
            double dy = ys[i] - fit.cy;
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < 1e-300) {
                continue;
            }
            double residual = dist - fit.r;
            /* d(residual)/d(cx, cy, r) */
            double j0 = -dx / dist;
            double j1 = -dy / dist;
            double j2 = -1.0;

            jtj[0][0] += j0 * j0; jtj[0][1] += j0 * j1; jtj[0][2] += j0 * j2;
            jtj[1][1] += j1 * j1; jtj[1][2] += j1 * j2;
            jtj[2][2] += j2 * j2;
            jtr[0] += j0 * residual;
            jtr[1] += j1 * residual;
            jtr[2] += j2 * residual;
        }
        jtj[1][0] = jtj[0][1];
        jtj[2][0] = jtj[0][2];
        jtj[2][1] = jtj[1][2];

        /* Levenberg-style damping keeps the step sane when the span is a
         * short arc and the centre is poorly constrained along one axis. */
        for (int k = 0; k < 3; k++) {
            jtj[k][k] *= 1.0 + 1e-9;
        }

        double det = jtj[0][0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
                   - jtj[0][1] * (jtj[1][0] * jtj[2][2] - jtj[1][2] * jtj[2][0])
                   + jtj[0][2] * (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]);
        if (fabs(det) < 1e-300) {
            break;
        }

        double b[3] = {-jtr[0], -jtr[1], -jtr[2]};
        double step[3];
        for (int col = 0; col < 3; col++) {
            double m[3][3];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    m[r][c] = (c == col) ? b[r] : jtj[r][c];
                }
            }
            double d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                     - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                     + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            step[col] = d / det;
        }

        if (!isfinite(step[0]) || !isfinite(step[1]) || !isfinite(step[2])) {
            break;
        }

        fit.cx += step[0];
        fit.cy += step[1];
        fit.r += step[2];
        if (fit.r <= 0.0 || !isfinite(fit.r)) {
            return start;
        }
        if (fabs(step[0]) + fabs(step[1]) + fabs(step[2]) < 1e-14) {
            break;
        }
    }

    circle_residuals(xs, ys, n, &fit);
    return fit;
}

/* ---------------------------------------------------------------- */
/* Python bindings                                                   */
/* ---------------------------------------------------------------- */

static int read_points(PyObject *obj, double **xs, double **ys, int *n)
{
    PyObject *seq = PySequence_Fast(obj, "expected a sequence of (x, y) points");
    if (!seq) {
        return 0;
    }
    Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    if (count < 1) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "no points given");
        return 0;
    }
    double *ax = (double *)malloc((size_t)count * sizeof(double));
    double *ay = (double *)malloc((size_t)count * sizeof(double));
    if (!ax || !ay) {
        free(ax);
        free(ay);
        Py_DECREF(seq);
        PyErr_NoMemory();
        return 0;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyArg_ParseTuple(item, "dd", &ax[i], &ay[i])) {
            free(ax);
            free(ay);
            Py_DECREF(seq);
            return 0;
        }
    }
    Py_DECREF(seq);
    *xs = ax;
    *ys = ay;
    *n = (int)count;
    return 1;
}

static PyObject *py_fit_line(PyObject *self, PyObject *args)
{
    PyObject *obj;
    double *xs, *ys;
    int n;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (!read_points(obj, &xs, &ys, &n)) {
        return NULL;
    }
    LineFit fit = fit_line_tls(xs, ys, n);
    free(xs);
    free(ys);
    if (!fit.ok) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("{s:(dd),s:(dd),s:d,s:d,s:i}", "point", fit.px, fit.py, "direction", fit.dx,
                         fit.dy, "rms", fit.rms, "max_deviation", fit.max_dev, "points", n);
}

static PyObject *py_fit_circle(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int refine = 1;
    double *xs, *ys;
    int n;
    (void)self;
    if (!PyArg_ParseTuple(args, "O|i", &obj, &refine)) {
        return NULL;
    }
    if (!read_points(obj, &xs, &ys, &n)) {
        return NULL;
    }
    CircleFit fit = fit_circle_taubin(xs, ys, n);
    if (fit.ok) {
        circle_residuals(xs, ys, n, &fit);
        if (refine) {
            fit = refine_circle_geometric(xs, ys, n, fit, 50);
        }
    }
    free(xs);
    free(ys);
    if (!fit.ok) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("{s:(dd),s:d,s:d,s:d,s:d,s:d,s:i}", "center", fit.cx, fit.cy, "radius",
                         fit.r, "rms", fit.rms, "max_deviation", fit.max_dev, "span", fit.span,
                         "sagitta", fit.sagitta, "points", n);
}

/*
 * segment(points, tolerance, min_points) -> [(start, end, kind, params...)]
 *
 * Greedy: from each start, grow a line span and an arc span as far as each
 * stays within tolerance, then emit whichever reached further. A tie goes
 * to the line, because a drawing is more likely to contain a straight edge
 * than an arc that happens to look straight.
 */
static PyObject *py_segment(PyObject *self, PyObject *args)
{
    PyObject *obj;
    double tolerance;
    int min_points = 5;
    double *xs, *ys;
    int n;
    (void)self;

    if (!PyArg_ParseTuple(args, "Od|i", &obj, &tolerance, &min_points)) {
        return NULL;
    }
    if (tolerance <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "tolerance must be positive");
        return NULL;
    }
    if (!read_points(obj, &xs, &ys, &n)) {
        return NULL;
    }
    if (min_points < 3) {
        min_points = 3;
    }

    PyObject *result = PyList_New(0);
    if (!result) {
        free(xs);
        free(ys);
        return NULL;
    }

    int start = 0;
    while (start < n - 1) {
        int line_end = start + 1;
        LineFit best_line = fit_line_tls(xs + start, ys + start, 2);
        for (int end = start + 2; end < n; end++) {
            LineFit candidate = fit_line_tls(xs + start, ys + start, end - start + 1);
            if (!candidate.ok || candidate.max_dev > tolerance) {
                break;
            }
            best_line = candidate;
            line_end = end;
        }

        int arc_end = start;
        CircleFit best_arc;
        best_arc.ok = 0;
        best_arc.cx = best_arc.cy = best_arc.r = best_arc.rms = best_arc.max_dev = 0.0;
        best_arc.span = best_arc.sagitta = 0.0;

        /*
         * Grow the arc past early failures.
         *
         * A circle fit over a handful of points spanning a few degrees is
         * ill-conditioned by nature: the centre is barely constrained, so
         * the residual is large and the radius meaningless. Stopping at the
         * first failure therefore abandons every arc before it becomes
         * identifiable -- a full circle came out as three dozen line
         * segments. Conditioning improves monotonically as the span grows,
         * so keep extending through a grace period, and only stop early
         * once a good span has been found and then genuinely ends (a real
         * corner).
         */
        int failures = 0;
        for (int end = start + min_points - 1; end < n; end++) {
            int count = end - start + 1;
            CircleFit candidate = fit_circle_taubin(xs + start, ys + start, count);
            int acceptable = 0;
            if (candidate.ok) {
                circle_residuals(xs + start, ys + start, count, &candidate);
                candidate = refine_circle_geometric(xs + start, ys + start, count, candidate, 12);
                /* Curvature must be *resolvable*, not merely fittable: an
                 * arc bulging away from its chord by less than the noise
                 * carries no usable evidence of a radius, and accepting it
                 * would report a 3000mm arc where the drawing has a
                 * straight edge. */
                /* RMS rather than max deviation: a sub-pixel contour has
                 * occasional single-point outliers, and judging a span by
                 * its worst point lets one bad sample truncate an
                 * otherwise excellent arc. The max is still reported so a
                 * caller can see it. */
                acceptable = candidate.ok && candidate.rms <= tolerance &&
                             candidate.max_dev <= 3.0 * tolerance &&
                             candidate.sagitta >= 2.0 * tolerance;
            }

            if (acceptable) {
                best_arc = candidate;
                arc_end = end;
                failures = 0;
            } else {
                failures++;
                if (arc_end > start) {
                    if (failures > 12) {
                        break; /* a good arc ended: this is a corner */
                    }
                } else if (failures > 64) {
                    break; /* never became identifiable: not an arc */
                }
            }
        }

        PyObject *entry;
        /*
         * Prefer the arc on a tie rather than the line.
         *
         * On a large circle a chord stays within tolerance for a long run,
         * so a raw "most points wins" race hands curved geometry to
         * straight segments -- a rendered circle came back as three dozen
         * lines. The arc has already had to clear the sagitta guard to get
         * here, which means its curvature was measurable; when it reaches
         * as far as the line, it is the better model of the same data.
         */
        if (best_arc.ok && arc_end >= line_end && (arc_end - start) >= min_points - 1) {
            entry = Py_BuildValue("{s:i,s:i,s:s,s:(dd),s:d,s:d,s:d,s:d,s:d}", "start", start, "end",
                                  arc_end, "kind", "arc", "center", best_arc.cx, best_arc.cy,
                                  "radius", best_arc.r, "rms", best_arc.rms, "max_deviation",
                                  best_arc.max_dev, "span", best_arc.span, "sagitta",
                                  best_arc.sagitta);
            start = arc_end;
        } else {
            entry = Py_BuildValue("{s:i,s:i,s:s,s:(dd),s:(dd),s:d,s:d}", "start", start, "end",
                                  line_end, "kind", "line", "point", best_line.px, best_line.py,
                                  "direction", best_line.dx, best_line.dy, "rms", best_line.rms,
                                  "max_deviation", best_line.max_dev);
            start = line_end;
        }

        if (!entry || PyList_Append(result, entry) != 0) {
            Py_XDECREF(entry);
            Py_CLEAR(result);
            break;
        }
        Py_DECREF(entry);
    }

    free(xs);
    free(ys);
    return result;
}

static PyMethodDef FitMethods[] = {
    {"fit_line", py_fit_line, METH_VARARGS,
     "fit_line(points) -> dict: total-least-squares line (perpendicular distance)."},
    {"fit_circle", py_fit_circle, METH_VARARGS,
     "fit_circle(points, refine=1) -> dict: Taubin fit, optionally refined by "
     "Gauss-Newton on the true orthogonal residual."},
    {"segment", py_segment, METH_VARARGS,
     "segment(points, tolerance, min_points=5) -> list of spans classified as "
     "'line' or 'arc' with their fitted parameters."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef fitmodule = {PyModuleDef_HEAD_INIT, "blueprint23d._native.fitkernels",
                                       "Least-squares primitive fitting and chain segmentation.", -1,
                                       FitMethods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_fitkernels(void)
{
    return PyModule_Create(&fitmodule);
}
