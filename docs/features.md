# Feature reference

Generated from the registry; do not edit by hand.

## Face (node) features

| Name | Shape | dtype | Params | Version | Meaning |
| --- | --- | --- | --- | --- | --- |
| `surface_type` | () | int64 | - | 1 | Index of OCP's GeomAbs_SurfaceType (0 plane, 1 cylinder, 2 cone, 3 sphere, 4 torus, 5 Bezier, 6 B-spline, 7 revolution, 8 extrusion, 9 offset, 10 other). |
| `area` | () | float32 | - | 1 | Face area in model units squared. |
| `centroid` | (3) | float32 | - | 1 | Centre of mass of the face. |
| `uv_grid` | (7, 10, 10) * | float32 | size=10 | 1 | Channels-first UV sample grid: x, y, z, nx, ny, nz (outward unit normal), mask (1 = inside the trimmed face). Axis 1 is u, axis 2 is v. |

## Edge features

| Name | Shape | dtype | Params | Version | Meaning |
| --- | --- | --- | --- | --- | --- |
| `curve_type` | () | int64 | - | 1 | Index of OCP's GeomAbs_CurveType (0 line, 1 circle, 2 ellipse, 3 hyperbola, 4 parabola, 5 Bezier, 6 B-spline, 7 offset, 8 other). |
| `length` | () | float32 | - | 1 | Edge length in model units. |
| `u_grid` | (6, 10) * | float32 | size=10 | 1 | Channels-first samples along the edge: x, y, z, tx, ty, tz (unit tangent). Points run along the edge's own B-rep orientation (reversed edges are walked from the last parameter to the first and the tangent follows). |
| `dihedral_angle` | () | float32 | - | 1 | Angle in radians (0..pi) between the two adjacent faces' outward normals at the edge midpoint. |
| `convexity` | () | int64 | tol=0.01 | 1 | 0 concave, 1 convex, 2 smooth (dihedral angle below ``tol`` radians). |

`*` shape shown for default parameters.
