#!/usr/bin/env bash
# Build a minimal isaacgym stub for machines without an Isaac Gym install.
#
# The TWIST sim2sim stack (deploy_real/server_high_level_motion_lib.py) only
# needs isaacgym.torch_utils, which is pure torch/numpy. This script extracts
# that single file from the Isaac Gym SDK tarball (the SDK is not
# redistributable, hence built locally instead of committed), patches the
# np.float alias removed in numpy >= 1.24, and writes import-time dummies for
# the compiled isaacgym submodules so legged_gym training code can be
# *imported* for offline tests — simulation/training still needs a real
# Isaac Gym install.
#
# Usage:
#   bash tools/setup_isaacgym_stub.sh [TARBALL] [OUT_DIR]
#     TARBALL  Isaac Gym Preview 4 tarball
#              (default: ~/Downloads/IsaacGym_Preview_4_Package.tar.gz)
#     OUT_DIR  where to build the stub (default: <repo>/isaacgym_stub, gitignored)
#
# Then:
#   # motion server / sim2sim (isaacgym.torch_utils only):
#   export PYTHONPATH=OUT_DIR:$PYTHONPATH
#   # import-light tests of legged_gym code (adds pydelatin/pyfqmr dummies;
#   # keep this off the path in envs that have the real packages):
#   export PYTHONPATH=OUT_DIR:OUT_DIR/shims:$PYTHONPATH
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARBALL="${1:-$HOME/Downloads/IsaacGym_Preview_4_Package.tar.gz}"
OUT_DIR="${2:-$REPO_ROOT/isaacgym_stub}"

if [ ! -f "$TARBALL" ]; then
    echo "error: Isaac Gym tarball not found at $TARBALL" >&2
    echo "download Isaac Gym Preview 4 from NVIDIA and pass its path as the first argument" >&2
    exit 1
fi

mkdir -p "$OUT_DIR/isaacgym" "$OUT_DIR/shims"

tar -xzf "$TARBALL" isaacgym/python/isaacgym/torch_utils.py -O > "$OUT_DIR/isaacgym/torch_utils.py"
# np.float was removed in numpy >= 1.24 (\b keeps np.float32/np.float64 intact)
sed -i 's/np\.float\b/float/g' "$OUT_DIR/isaacgym/torch_utils.py"

touch "$OUT_DIR/isaacgym/__init__.py"

# import-time dummies (PEP 562: any attribute resolves to an inert placeholder)
DUMMY_BODY='# import-time stub: any attribute resolves to an inert placeholder (PEP 562).
# Enough to import code that depends on this module; nothing here can run.
def __getattr__(name):
    class _Stub:
        def __init__(self, *args, **kwargs):
            pass
    return _Stub'

for m in gymtorch gymapi gymutil terrain_utils; do
    printf '%s\n' "$DUMMY_BODY" > "$OUT_DIR/isaacgym/$m.py"
done
# third-party terrain deps of legged_gym, on a separate path so they never
# shadow real installs unless explicitly added
for m in pydelatin pyfqmr; do
    printf '%s\n' "$DUMMY_BODY" > "$OUT_DIR/shims/$m.py"
done

echo "isaacgym stub built at $OUT_DIR"
echo "  motion server / sim2sim:          export PYTHONPATH=$OUT_DIR:\$PYTHONPATH"
echo "  import-light training-code tests: export PYTHONPATH=$OUT_DIR:$OUT_DIR/shims:\$PYTHONPATH"
