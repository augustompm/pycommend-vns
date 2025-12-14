"""
MO-GVNS Canonical Implementation

References:
    Duarte, A., Marti, R., Alvarez, A., & Angel-Bello, F. (2015).
    Multi-objective variable neighborhood search: an application to
    combinatorial optimization problems. J Glob Optim, 63, 515-536.

    Hansen, P., Mladenovic, N., Todosijevic, R., & Hanafi, S. (2017).
    Variable neighborhood search: basics and variants. EURO J Comput Optim,
    5, 423-454.

Modules:
    dominance: Algorithm 1 (MO-Improvement)
    archive: Algorithm 2 (MO-NeighborhoodChange)
    shake: Algorithm 3 (MO-Shake)
    vnd: Algorithms 5-6 (VND-i and MO-VND)
    mo_gvns: Algorithm 7 (MO-GVNS)
"""

from .dominance import (
    dominates_baseline, dominates_numpy, dominates_numba,
    is_dominated_baseline, is_dominated_numpy, is_dominated_numba,
    mo_improvement_baseline, mo_improvement_numpy, mo_improvement_numba,
)

from .archive import (
    update_archive_baseline, update_archive_numpy, update_archive_numba,
    mo_neighborhood_change_baseline, mo_neighborhood_change_numpy,
    mo_neighborhood_change_numba,
)

from .shake import (
    shake_solution_baseline, shake_solution_numpy, shake_solution_numba,
    mo_shake_baseline, mo_shake_numpy, mo_shake_numba,
)

from .vnd import (
    vnd_i_baseline, vnd_i_numpy, vnd_i_numba,
    mo_vnd_baseline, mo_vnd_numpy, mo_vnd_numba,
)

from .mo_gvns import (
    mo_gvns, mo_gvns_baseline, mo_gvns_numpy, mo_gvns_numba,
    MOGVNSResult, create_initial_archive,
)

__all__ = [
    'dominates_baseline', 'dominates_numpy', 'dominates_numba',
    'is_dominated_baseline', 'is_dominated_numpy', 'is_dominated_numba',
    'mo_improvement_baseline', 'mo_improvement_numpy', 'mo_improvement_numba',
    'update_archive_baseline', 'update_archive_numpy', 'update_archive_numba',
    'mo_neighborhood_change_baseline', 'mo_neighborhood_change_numpy',
    'mo_neighborhood_change_numba',
    'shake_solution_baseline', 'shake_solution_numpy', 'shake_solution_numba',
    'mo_shake_baseline', 'mo_shake_numpy', 'mo_shake_numba',
    'vnd_i_baseline', 'vnd_i_numpy', 'vnd_i_numba',
    'mo_vnd_baseline', 'mo_vnd_numpy', 'mo_vnd_numba',
    'mo_gvns', 'mo_gvns_baseline', 'mo_gvns_numpy', 'mo_gvns_numba',
    'MOGVNSResult', 'create_initial_archive',
]
