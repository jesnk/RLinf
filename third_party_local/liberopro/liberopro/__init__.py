"""liberopro.liberopro - position-perturbation overlay over standard libero.

This package is a thin compatibility shim that lets RLinf's existing
LIBERO-PRO code-path (`LIBERO_TYPE=pro`, `LIBERO_PERTURBATION=swap`) work
without installing the upstream LIBERO-PRO repo as `liberopro`.

What this provides:
- `liberopro.liberopro` is an alias module of `libero.libero` (so submodules
  `benchmark`, `envs`, etc. resolve through standard libero).
- `liberopro.liberopro.get_libero_path("bddl_files")` returns the LIBERO-PRO
  data overlay (LIBEROPRO_DATA_ROOT, default `/home/jskang/sigma/liberopro_data`).
- All other keys delegate to standard libero.

The LIBERO-PRO swap (position) perturbation is wired up by
RLinf's `rlinf/envs/libero/libero_env.py` (which already supports
LIBERO_TYPE=pro / LIBERO_PERTURBATION). For init_state loading from the
matching swap-bddl variant, see `LiberoEnv._maybe_load_swap_init_state`.
"""

import os

# Default overlay root (override via LIBEROPRO_DATA_ROOT env var).
_DEFAULT_DATA_ROOT = "/home/jskang/sigma/liberopro_data"

# Asset paths come from standard libero. We need to expose them so that
# RLinf's libero_env.py (which sets LIBERO_ASSET_ROOT/LIBERO_BDDL_PATH/etc.
# from os.path.dirname(real_core.__file__)) finds the right files.
import libero.libero as _std_libero_core
__file__ = _std_libero_core.__file__
__path__ = _std_libero_core.__path__


def _data_root() -> str:
    return os.environ.get("LIBEROPRO_DATA_ROOT", _DEFAULT_DATA_ROOT)


def get_libero_path(query_key: str):
    """Return path for given key.

    For 'bddl_files' returns the LIBERO-PRO overlay's bddl_files dir
    (containing `<suite>_swap` subdirectories). For 'init_files' /
    'init_states' returns the overlay's init_files dir (containing matching
    `<suite>_swap` subdirectories with regenerated `.pruned_init`s).
    Other keys delegate to standard libero's get_libero_path.
    """
    if query_key == "bddl_files":
        return os.path.join(_data_root(), "bddl_files")
    if query_key in ("init_files", "init_states"):
        return os.path.join(_data_root(), "init_files")
    return _std_libero_core.get_libero_path(query_key)
