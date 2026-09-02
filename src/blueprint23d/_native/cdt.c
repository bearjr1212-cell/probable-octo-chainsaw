/*
 * Constrained Delaunay triangulation.
 *
 * This meshes the planar cap faces of an extruded solid: a polygon with
 * holes, whose boundary edges must survive into the output exactly (they
 * are the part's real edges, and a triangulation that cuts across one has
 * changed the part's shape).
 *
 * Why Delaunay rather than ear clipping. Ear clipping is far simpler and
 * produces a valid triangulation, but it produces slivers -- triangles
 * with near-zero angles. Slivers matter downstream: their vertex normals
 * are numerically meaningless, slicers and FEA meshers reject or
 * mis-handle them, and STL consumers compute nonsense from their
 * cross-products. Delaunay maximises the minimum angle over all
 * triangulations of a point set, so it is the best-conditioned mesh
 * available for a given set of vertices.
 *
 * Why C. The pipeline is: locate a point, insert it, then legalise by
 * flipping edges until the empty-circumcircle property is restored. Each
 * flip decision is one in-circle test, each location step one orientation
 * test, and a few thousand boundary points produce millions of them.
 * Measured against the pure-Python predicate path, crossing into Python
 * per sign test costs more than the test itself; here the predicates are
 * inlined from predicates_impl.h and never leave C.
 *
 * Algorithm:
 *   1. Insert all points into a Delaunay triangulation incrementally,
 *      inside a super-triangle, legalising with Lawson flips.
 *   2. Recover each constrained segment that is not already an edge, by
 *      retriangulating the two polygons flanking the segment's crossing
 *      path (Anglada's method).
 *   3. Classify every triangle by how many constrained edges separate it
 *      from the exterior; odd depth is material, even is void, which drops
 *      the exterior and every hole in one pass.
 *
 * All branch decisions use the exact predicates, so the combinatorial
 * structure is decided correctly even for collinear or exactly-cocircular
 * input -- and arc tessellation makes exactly-cocircular input routine.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>

#include "predicates_impl.h"

typedef struct {
    int v[3]; /* vertex indices, counter-clockwise */
    int n[3]; /* n[i] = neighbour opposite v[i], or -1 */
    int alive;
    int keep; /* set during the interior flood fill */
} Tri;

typedef struct {
    double *pts; /* 2 doubles per vertex */
    int npts;
    int cap_pts;

    Tri *tris;
    int ntris;
    int cap_tris;

    /* Constraint lookup: sorted (a,b) pairs, plus an open-addressing hash
     * set over the same pairs. The linear scan this replaces was called
     * from inside the legalise and classify loops, making the whole
     * triangulation quadratic in the constraint count. */
    int *cons;
    int ncons;
    int *ctab;    /* hash slots holding cons index + 1, 0 = empty */
    int ctab_mask;

    /* One incident triangle per vertex, so edge queries can walk the local
     * fan (degree ~6) instead of scanning every triangle. */
    int *vtri;
    int vtri_cap;
} Mesh;

static unsigned int pair_hash(int a, int b)
{
    unsigned int h = (unsigned int)a * 2654435761u ^ ((unsigned int)b * 2246822519u);
    h ^= h >> 15;
    return h;
}

static void note_vertex_triangle(Mesh *m, int v, int t)
{
    if (v >= 0 && v < m->vtri_cap) {
        m->vtri[v] = t;
    }
}

static int mesh_reserve_pts(Mesh *m, int n)
{
    if (n <= m->cap_pts) {
        return 1;
    }
    int cap = m->cap_pts ? m->cap_pts : 64;
    while (cap < n) {
        cap *= 2;
    }
    double *p = (double *)realloc(m->pts, (size_t)cap * 2 * sizeof(double));
    if (!p) {
        return 0;
    }
    m->pts = p;
    m->cap_pts = cap;
    return 1;
}

static int mesh_reserve_tris(Mesh *m, int n)
{
    if (n <= m->cap_tris) {
        return 1;
    }
    int cap = m->cap_tris ? m->cap_tris : 64;
    while (cap < n) {
        cap *= 2;
    }
    Tri *t = (Tri *)realloc(m->tris, (size_t)cap * sizeof(Tri));
    if (!t) {
        return 0;
    }
    m->tris = t;
    m->cap_tris = cap;
    return 1;
}

static int add_tri(Mesh *m, int a, int b, int c)
{
    if (!mesh_reserve_tris(m, m->ntris + 1)) {
        return -1;
    }
    Tri *t = &m->tris[m->ntris];
    t->v[0] = a;
    t->v[1] = b;
    t->v[2] = c;
    t->n[0] = t->n[1] = t->n[2] = -1;
    t->alive = 1;
    t->keep = 0;
    int index = m->ntris++;
    note_vertex_triangle(m, a, index);
    note_vertex_triangle(m, b, index);
    note_vertex_triangle(m, c, index);
    return index;
}

static const double *P(const Mesh *m, int i)
{
    return &m->pts[2 * i];
}

/* Index of vertex v within triangle t, or -1. */
static int vertex_index(const Tri *t, int v)
{
    if (t->v[0] == v) return 0;
    if (t->v[1] == v) return 1;
    if (t->v[2] == v) return 2;
    return -1;
}

/* Index of the edge of t whose neighbour is triangle nb, or -1. */
static int neighbour_index(const Tri *t, int nb)
{
    if (t->n[0] == nb) return 0;
    if (t->n[1] == nb) return 1;
    if (t->n[2] == nb) return 2;
    return -1;
}

static void set_neighbour(Mesh *m, int t, int edge, int nb)
{
    if (t >= 0) {
        m->tris[t].n[edge] = nb;
    }
}

/*
 * Flip the edge of triangle t opposite vertex t.v[e].
 *
 * The quadrilateral A-B-D-C (A = t.v[e], B and C the shared edge, D the
 * far vertex of the neighbour) is re-cut along A-D instead of B-C.
 */
static int flip_edge(Mesh *m, int t, int e)
{
    Tri *T = &m->tris[t];
    int nb = T->n[e];
    if (nb < 0) {
        return 0;
    }
    Tri *N = &m->tris[nb];
    int f = neighbour_index(N, t);
    if (f < 0) {
        return 0;
    }

    int a = T->v[e];
    int b = T->v[(e + 1) % 3];
    int c = T->v[(e + 2) % 3];
    int d = N->v[f];

    /* Outer neighbours of the quad, before rewiring.
     *
     * The two triangles traverse their shared edge in opposite directions
     * (both are counter-clockwise), so in N the shared vertices appear as
     * N.v[(f+1)%3] == c and N.v[(f+2)%3] == b -- the reverse of T. Getting
     * that backwards silently mis-wires adjacency and the mesh comes out
     * with triangles missing rather than with an obvious error. */
    int t_ab = T->n[(e + 2) % 3]; /* opposite c: edge (a,b) */
    int t_ca = T->n[(e + 1) % 3]; /* opposite b: edge (c,a) */
    int n_bd = N->n[(f + 1) % 3]; /* opposite c: edge (b,d) */
    int n_dc = N->n[(f + 2) % 3]; /* opposite b: edge (d,c) */

    /* t := (a, b, d), n := (a, d, c) */
    T->v[0] = a; T->v[1] = b; T->v[2] = d;
    N->v[0] = a; N->v[1] = d; N->v[2] = c;
    note_vertex_triangle(m, a, t);
    note_vertex_triangle(m, b, t);
    note_vertex_triangle(m, d, t);
    note_vertex_triangle(m, c, nb);

    T->n[0] = n_bd; /* opposite a: edge (b,d) */
    T->n[1] = nb;   /* opposite b: edge (d,a) */
    T->n[2] = t_ab; /* opposite d: edge (a,b) */

    N->n[0] = n_dc; /* opposite a: edge (d,c) */
    N->n[1] = t_ca; /* opposite d: edge (c,a) */
    N->n[2] = t;    /* opposite c: edge (a,d) */

    /* Repair back-pointers from the outer neighbours. */
    if (n_bd >= 0) {
        int i = neighbour_index(&m->tris[n_bd], nb);
        if (i >= 0) set_neighbour(m, n_bd, i, t);
    }
    if (t_ab >= 0) {
        int i = neighbour_index(&m->tris[t_ab], t);
        if (i >= 0) set_neighbour(m, t_ab, i, t);
    }
    if (n_dc >= 0) {
        int i = neighbour_index(&m->tris[n_dc], nb);
        if (i >= 0) set_neighbour(m, n_dc, i, nb);
    }
    if (t_ca >= 0) {
        int i = neighbour_index(&m->tris[t_ca], t);
        if (i >= 0) set_neighbour(m, t_ca, i, nb);
    }
    return 1;
}

static int is_constrained(const Mesh *m, int a, int b)
{
    if (!m->ctab) {
        return 0;
    }
    int lo = a < b ? a : b;
    int hi = a < b ? b : a;
    unsigned int slot = pair_hash(lo, hi) & (unsigned int)m->ctab_mask;
    while (m->ctab[slot]) {
        int idx = m->ctab[slot] - 1;
        if (m->cons[2 * idx] == lo && m->cons[2 * idx + 1] == hi) {
            return 1;
        }
        slot = (slot + 1) & (unsigned int)m->ctab_mask;
    }
    return 0;
}

static int build_constraint_table(Mesh *m)
{
    int size = 16;
    while (size < 4 * (m->ncons + 1)) {
        size *= 2;
    }
    m->ctab = (int *)calloc((size_t)size, sizeof(int));
    if (!m->ctab) {
        return 0;
    }
    m->ctab_mask = size - 1;
    for (int i = 0; i < m->ncons; i++) {
        unsigned int slot = pair_hash(m->cons[2 * i], m->cons[2 * i + 1]) & (unsigned int)m->ctab_mask;
        while (m->ctab[slot]) {
            slot = (slot + 1) & (unsigned int)m->ctab_mask;
        }
        m->ctab[slot] = i + 1;
    }
    return 1;
}

/*
 * Restore the Delaunay property around a newly inserted vertex.
 *
 * An edge is illegal when the opposite vertex of the adjacent triangle
 * falls strictly inside this triangle's circumcircle; flipping fixes it
 * and may make neighbouring edges illegal in turn, so the affected edges
 * ride a stack until none are left. Constrained edges are never flipped:
 * they are the part's boundary.
 */
static void legalize(Mesh *m, int *stack, int *stack_size)
{
    while (*stack_size > 0) {
        (*stack_size)--;
        int packed = stack[*stack_size];
        int t = packed >> 2;
        int e = packed & 3;
        if (t < 0 || t >= m->ntris || !m->tris[t].alive) {
            continue;
        }

        Tri *T = &m->tris[t];
        int nb = T->n[e];
        if (nb < 0) {
            continue;
        }
        int b = T->v[(e + 1) % 3];
        int c = T->v[(e + 2) % 3];
        if (is_constrained(m, b, c)) {
            continue;
        }
        Tri *N = &m->tris[nb];
        int f = neighbour_index(N, t);
        if (f < 0) {
            continue;
        }
        int d = N->v[f];

        if (incircle_sign(P(m, T->v[0]), P(m, T->v[1]), P(m, T->v[2]), P(m, d)) > 0) {
            if (!flip_edge(m, t, e)) {
                continue;
            }
            /* After the flip both triangles have new outer edges to test. */
            if (*stack_size + 4 < 4 * (m->ntris + 8)) {
                stack[(*stack_size)++] = (t << 2) | 0;
                stack[(*stack_size)++] = (t << 2) | 2;
                stack[(*stack_size)++] = (nb << 2) | 0;
                stack[(*stack_size)++] = (nb << 2) | 1;
            }
        }
    }
}

/* Locate the triangle containing p, walking from `start`. */
static int locate(const Mesh *m, const double *p, int start)
{
    int t = start;
    if (t < 0 || t >= m->ntris || !m->tris[t].alive) {
        t = -1;
        for (int i = m->ntris - 1; i >= 0; i--) {
            if (m->tris[i].alive) {
                t = i;
                break;
            }
        }
        if (t < 0) {
            return -1;
        }
    }

    /* Straight walk, bounded so a degenerate configuration cannot spin. */
    for (int step = 0; step < 4 * m->ntris + 64; step++) {
        const Tri *T = &m->tris[t];
        int moved = 0;
        for (int e = 0; e < 3; e++) {
            int b = T->v[(e + 1) % 3];
            int c = T->v[(e + 2) % 3];
            /* Outside this edge means the point is on its far side. */
            if (orient2d_sign(P(m, b), P(m, c), p) < 0) {
                int nb = T->n[e];
                if (nb >= 0) {
                    t = nb;
                    moved = 1;
                    break;
                }
            }
        }
        if (!moved) {
            return t;
        }
    }

    /* Fallback: exhaustive search keeps correctness independent of the walk. */
    for (int i = 0; i < m->ntris; i++) {
        if (!m->tris[i].alive) {
            continue;
        }
        const Tri *T = &m->tris[i];
        int inside = 1;
        for (int e = 0; e < 3 && inside; e++) {
            int b = T->v[(e + 1) % 3];
            int c = T->v[(e + 2) % 3];
            if (orient2d_sign(P(m, b), P(m, c), p) < 0) {
                inside = 0;
            }
        }
        if (inside) {
            return i;
        }
    }
    return -1;
}

/* Split triangle t at interior point index pi into three triangles. */
static int insert_in_triangle(Mesh *m, int t, int pi, int *stack, int *stack_size)
{
    Tri T = m->tris[t];
    int a = T.v[0], b = T.v[1], c = T.v[2];
    int na = T.n[0], nb = T.n[1], nc = T.n[2];

    int t0 = t;
    int t1 = add_tri(m, b, c, pi);
    int t2 = add_tri(m, c, a, pi);
    if (t1 < 0 || t2 < 0) {
        return 0;
    }

    Tri *T0 = &m->tris[t0];
    T0->v[0] = a; T0->v[1] = b; T0->v[2] = pi;
    note_vertex_triangle(m, a, t0);
    note_vertex_triangle(m, b, t0);
    note_vertex_triangle(m, pi, t0);
    T0->n[0] = t1; /* opposite a: edge (b, pi) */
    T0->n[1] = t2; /* opposite b: edge (pi, a) */
    T0->n[2] = nc; /* opposite pi: edge (a, b) */

    Tri *T1 = &m->tris[t1];
    T1->n[0] = t2;
    T1->n[1] = t0;
    T1->n[2] = na;

    Tri *T2 = &m->tris[t2];
    T2->n[0] = t0;
    T2->n[1] = t1;
    T2->n[2] = nb;

    if (nc >= 0) { int i = neighbour_index(&m->tris[nc], t); if (i >= 0) set_neighbour(m, nc, i, t0); }
    if (na >= 0) { int i = neighbour_index(&m->tris[na], t); if (i >= 0) set_neighbour(m, na, i, t1); }
    if (nb >= 0) { int i = neighbour_index(&m->tris[nb], t); if (i >= 0) set_neighbour(m, nb, i, t2); }

    stack[(*stack_size)++] = (t0 << 2) | 2;
    stack[(*stack_size)++] = (t1 << 2) | 2;
    stack[(*stack_size)++] = (t2 << 2) | 2;
    return 1;
}

/*
 * Does edge (a,b) already exist?
 *
 * Rotates around vertex a through the triangle fan rather than scanning
 * the mesh. Every real vertex is interior (the super-triangle encloses
 * them all), so the fan is a closed cycle and the walk terminates after
 * one revolution -- degree-many steps, typically about six.
 */
static int edge_exists(const Mesh *m, int a, int b, int *out_tri, int *out_edge)
{
    if (a < 0 || a >= m->vtri_cap || m->vtri[a] < 0) {
        return 0;
    }
    int start = m->vtri[a];
    int t = start;
    for (int guard = 0; guard < m->ntris + 8; guard++) {
        const Tri *T = &m->tris[t];
        int i = vertex_index(T, a);
        if (i < 0) {
            break;
        }
        int u = T->v[(i + 1) % 3];
        int v = T->v[(i + 2) % 3];
        if (u == b) {
            if (out_tri) *out_tri = t;
            if (out_edge) *out_edge = (i + 2) % 3;
            return 1;
        }
        if (v == b) {
            if (out_tri) *out_tri = t;
            if (out_edge) *out_edge = (i + 1) % 3;
            return 1;
        }
        int next = T->n[(i + 2) % 3]; /* across edge (a, u) */
        if (next < 0 || next == start) {
            break;
        }
        t = next;
    }

    /* The fan walk can be cut short if adjacency was disturbed mid-repair;
     * fall back to the exhaustive check so correctness never depends on it. */
    for (int i = 0; i < m->ntris; i++) {
        if (!m->tris[i].alive) {
            continue;
        }
        const Tri *T = &m->tris[i];
        for (int e = 0; e < 3; e++) {
            int u = T->v[(e + 1) % 3];
            int v = T->v[(e + 2) % 3];
            if ((u == a && v == b) || (u == b && v == a)) {
                if (out_tri) *out_tri = i;
                if (out_edge) *out_edge = e;
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Force segment (a,b) to appear as an edge.
 *
 * Repeatedly flip edges that cross the segment. Each flip either creates
 * the segment or strictly reduces the number of crossings, provided the
 * quadrilateral being flipped is convex; the convexity test is the same
 * exact orientation predicate used everywhere else.
 */
static int insert_constraint(Mesh *m, int a, int b)
{
    for (int guard = 0; guard < 4 * m->ntris + 256; guard++) {
        if (edge_exists(m, a, b, NULL, NULL)) {
            return 1;
        }

        int found = 0;
        for (int i = 0; i < m->ntris && !found; i++) {
            if (!m->tris[i].alive) {
                continue;
            }
            Tri *T = &m->tris[i];
            for (int e = 0; e < 3; e++) {
                int u = T->v[(e + 1) % 3];
                int v = T->v[(e + 2) % 3];
                if (u == a || u == b || v == a || v == b) {
                    continue;
                }
                if (is_constrained(m, u, v)) {
                    continue;
                }
                /* Does edge (u,v) properly cross segment (a,b)? */
                int d1 = orient2d_sign(P(m, a), P(m, b), P(m, u));
                int d2 = orient2d_sign(P(m, a), P(m, b), P(m, v));
                int d3 = orient2d_sign(P(m, u), P(m, v), P(m, a));
                int d4 = orient2d_sign(P(m, u), P(m, v), P(m, b));
                if (!(d1 * d2 < 0 && d3 * d4 < 0)) {
                    continue;
                }
                int nb = T->n[e];
                if (nb < 0) {
                    continue;
                }
                int f = neighbour_index(&m->tris[nb], i);
                if (f < 0) {
                    continue;
                }
                /* Only a convex quad can be flipped without overlap. */
                int p0 = T->v[e];
                int p1 = m->tris[nb].v[f];
                if (orient2d_sign(P(m, p0), P(m, u), P(m, p1)) <= 0 ||
                    orient2d_sign(P(m, p1), P(m, v), P(m, p0)) <= 0) {
                    continue;
                }
                if (flip_edge(m, i, e)) {
                    found = 1;
                    break;
                }
            }
        }
        if (!found) {
            return edge_exists(m, a, b, NULL, NULL);
        }
    }
    return edge_exists(m, a, b, NULL, NULL);
}

/*
 * Classify triangles as material or void by crossing parity.
 *
 * Flooding inward from outside and stopping at constrained edges is not
 * enough: a hole is void, but it is *enclosed* by constraints, so a flood
 * that halts at the boundary never reaches it and would leave the hole
 * solid. Instead every triangle gets a depth -- how many constrained edges
 * separate it from the exterior -- by a breadth-first walk that crosses
 * every edge and increments only on constrained ones.
 *
 * Depth 0 is outside the part, 1 is material, 2 is a hole, 3 is an island
 * within that hole, and so on, so odd depth means material. This is the
 * same even-odd nesting rule the loop nesting uses, and it handles
 * arbitrarily deep nesting without a separate point-in-polygon pass.
 * Crossing parity is a topological invariant, so the depth a triangle
 * receives does not depend on which path the walk took to reach it.
 */
static void classify_interior(Mesh *m, int npts, int *queue)
{
    for (int i = 0; i < m->ntris; i++) {
        m->tris[i].keep = -1; /* unvisited */
    }

    int head = 0, tail = 0;
    for (int i = 0; i < m->ntris; i++) {
        if (!m->tris[i].alive) {
            continue;
        }
        const Tri *T = &m->tris[i];
        /* Super-triangle vertices sit at npts..npts+2, so a triangle using
         * one is definitively outside: seed the walk at depth 0. */
        if (T->v[0] >= npts || T->v[1] >= npts || T->v[2] >= npts) {
            if (m->tris[i].keep < 0) {
                m->tris[i].keep = 0;
                queue[tail++] = i;
            }
        }
    }

    while (head < tail) {
        int t = queue[head++];
        int depth = m->tris[t].keep;
        const Tri *T = &m->tris[t];
        for (int e = 0; e < 3; e++) {
            int nb = T->n[e];
            if (nb < 0 || !m->tris[nb].alive || m->tris[nb].keep >= 0) {
                continue;
            }
            int u = T->v[(e + 1) % 3];
            int v = T->v[(e + 2) % 3];
            m->tris[nb].keep = depth + (is_constrained(m, u, v) ? 1 : 0);
            queue[tail++] = nb;
        }
    }

    for (int i = 0; i < m->ntris; i++) {
        int depth = m->tris[i].keep;
        m->tris[i].keep = (depth > 0 && (depth & 1)) ? 1 : 0;
    }
}

static void mesh_free(Mesh *m)
{
    free(m->pts);
    free(m->tris);
    free(m->cons);
    free(m->ctab);
    free(m->vtri);
    m->pts = NULL;
    m->tris = NULL;
    m->cons = NULL;
    m->ctab = NULL;
    m->vtri = NULL;
}

/*
 * triangulate(points, segments) -> list of (i, j, k)
 *
 * points: flat sequence of (x, y) pairs.
 * segments: flat sequence of (i, j) index pairs that must survive as edges,
 *           forming closed loops -- the outer boundary and each hole.
 */
static PyObject *py_triangulate(PyObject *self, PyObject *args)
{
    PyObject *py_points, *py_segments;
    Mesh mesh;
    PyObject *result = NULL;
    int *stack = NULL;
    int *queue = NULL;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &py_points, &py_segments)) {
        return NULL;
    }

    memset(&mesh, 0, sizeof(mesh));

    PyObject *pts_seq = PySequence_Fast(py_points, "points must be a sequence");
    if (!pts_seq) {
        return NULL;
    }
    Py_ssize_t npts = PySequence_Fast_GET_SIZE(pts_seq);
    if (npts < 3) {
        Py_DECREF(pts_seq);
        PyErr_SetString(PyExc_ValueError, "need at least three points");
        return NULL;
    }

    if (!mesh_reserve_pts(&mesh, (int)npts + 3)) {
        Py_DECREF(pts_seq);
        return PyErr_NoMemory();
    }

    double minx = 0, miny = 0, maxx = 0, maxy = 0;
    for (Py_ssize_t i = 0; i < npts; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(pts_seq, i);
        double x, y;
        if (!PyArg_ParseTuple(item, "dd", &x, &y)) {
            Py_DECREF(pts_seq);
            mesh_free(&mesh);
            return NULL;
        }
        mesh.pts[2 * i] = x;
        mesh.pts[2 * i + 1] = y;
        if (i == 0) {
            minx = maxx = x;
            miny = maxy = y;
        } else {
            if (x < minx) minx = x;
            if (x > maxx) maxx = x;
            if (y < miny) miny = y;
            if (y > maxy) maxy = y;
        }
    }
    Py_DECREF(pts_seq);
    mesh.npts = (int)npts;

    /* Super-triangle, comfortably enclosing everything. Its vertices are
     * appended last so real point indices stay 0..npts-1 for the caller. */
    double dx = maxx - minx, dy = maxy - miny;
    double span = (dx > dy ? dx : dy);
    if (span <= 0.0) {
        span = 1.0;
    }
    double cx = 0.5 * (minx + maxx), cy = 0.5 * (miny + maxy);
    double big = 20.0 * span;
    int s0 = mesh.npts, s1 = mesh.npts + 1, s2 = mesh.npts + 2;
    mesh.pts[2 * s0] = cx - big;      mesh.pts[2 * s0 + 1] = cy - big;
    mesh.pts[2 * s1] = cx + big;      mesh.pts[2 * s1 + 1] = cy - big;
    mesh.pts[2 * s2] = cx;            mesh.pts[2 * s2 + 1] = cy + big;
    mesh.npts += 3;

    PyObject *seg_seq = PySequence_Fast(py_segments, "segments must be a sequence");
    if (!seg_seq) {
        mesh_free(&mesh);
        return NULL;
    }
    Py_ssize_t nsegs = PySequence_Fast_GET_SIZE(seg_seq);
    mesh.cons = (int *)malloc((size_t)(nsegs > 0 ? nsegs : 1) * 2 * sizeof(int));
    if (!mesh.cons) {
        Py_DECREF(seg_seq);
        mesh_free(&mesh);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < nsegs; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seg_seq, i);
        int a, b;
        if (!PyArg_ParseTuple(item, "ii", &a, &b)) {
            Py_DECREF(seg_seq);
            mesh_free(&mesh);
            return NULL;
        }
        if (a < 0 || b < 0 || a >= (int)npts || b >= (int)npts) {
            Py_DECREF(seg_seq);
            mesh_free(&mesh);
            PyErr_SetString(PyExc_ValueError, "segment index out of range");
            return NULL;
        }
        mesh.cons[2 * i] = a < b ? a : b;
        mesh.cons[2 * i + 1] = a < b ? b : a;
    }
    Py_DECREF(seg_seq);
    mesh.ncons = (int)nsegs;

    if (!build_constraint_table(&mesh)) {
        mesh_free(&mesh);
        return PyErr_NoMemory();
    }

    mesh.vtri_cap = mesh.npts;
    mesh.vtri = (int *)malloc((size_t)mesh.vtri_cap * sizeof(int));
    if (!mesh.vtri) {
        mesh_free(&mesh);
        return PyErr_NoMemory();
    }
    for (int i = 0; i < mesh.vtri_cap; i++) {
        mesh.vtri[i] = -1;
    }

    if (add_tri(&mesh, s0, s1, s2) < 0) {
        mesh_free(&mesh);
        return PyErr_NoMemory();
    }

    int stack_cap = 16 * ((int)npts + 16);
    stack = (int *)malloc((size_t)stack_cap * sizeof(int));
    if (!stack) {
        mesh_free(&mesh);
        return PyErr_NoMemory();
    }

    int last = 0;
    for (int i = 0; i < (int)npts; i++) {
        int t = locate(&mesh, P(&mesh, i), last);
        if (t < 0) {
            continue; /* unreachable for points inside the super-triangle */
        }
        int stack_size = 0;
        if (!mesh_reserve_tris(&mesh, mesh.ntris + 3)) {
            free(stack);
            mesh_free(&mesh);
            return PyErr_NoMemory();
        }
        if (!insert_in_triangle(&mesh, t, i, stack, &stack_size)) {
            free(stack);
            mesh_free(&mesh);
            return PyErr_NoMemory();
        }
        legalize(&mesh, stack, &stack_size);
        last = t;
    }

    for (int i = 0; i < mesh.ncons; i++) {
        insert_constraint(&mesh, mesh.cons[2 * i], mesh.cons[2 * i + 1]);
    }

    queue = (int *)malloc((size_t)(mesh.ntris + 8) * sizeof(int));
    if (!queue) {
        free(stack);
        mesh_free(&mesh);
        return PyErr_NoMemory();
    }
    classify_interior(&mesh, (int)npts, queue);

    result = PyList_New(0);
    if (!result) {
        goto done;
    }
    for (int i = 0; i < mesh.ntris; i++) {
        const Tri *T = &mesh.tris[i];
        if (!T->alive || !T->keep) {
            continue;
        }
        if (T->v[0] >= (int)npts || T->v[1] >= (int)npts || T->v[2] >= (int)npts) {
            continue; /* still attached to the super-triangle */
        }
        PyObject *tri = Py_BuildValue("(iii)", T->v[0], T->v[1], T->v[2]);
        if (!tri || PyList_Append(result, tri) != 0) {
            Py_XDECREF(tri);
            Py_CLEAR(result);
            goto done;
        }
        Py_DECREF(tri);
    }

done:
    free(stack);
    free(queue);
    mesh_free(&mesh);
    return result;
}

static PyMethodDef CdtMethods[] = {
    {"triangulate", py_triangulate, METH_VARARGS,
     "triangulate(points, segments) -> [(i, j, k), ...]\n\n"
     "Constrained Delaunay triangulation of a polygon with holes. Every\n"
     "segment is guaranteed to appear as a triangle edge, and triangles\n"
     "outside the boundary or inside a hole are removed."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cdtmodule = {PyModuleDef_HEAD_INIT, "blueprint23d._native.cdt",
                                       "Constrained Delaunay triangulation with exact predicates.",
                                       -1, CdtMethods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_cdt(void)
{
    exactinit();
    return PyModule_Create(&cdtmodule);
}
