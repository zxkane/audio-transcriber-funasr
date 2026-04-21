#!/usr/bin/env python3
"""Patch FunASR's spectral clustering to use sparse eigenvalue decomposition.

Problem: FunASR's SpectralCluster.get_spec_embs() calls scipy.linalg.eigh()
which is O(N^3) — unusable for long recordings (N > 5000 segments).

Fix: For large matrices with known oracle_num, use scipy.sparse.linalg.eigsh()
which is O(N^2 * k) where k = number of speakers.

Also vectorizes the p_pruning loop for additional speedup.

Run this AFTER installing funasr, BEFORE transcribing long audio:
  python3 patch_clustering.py        # interactive (prompts before patching)
  python3 patch_clustering.py --yes  # non-interactive (e.g. from setup_env.sh)

The patch is idempotent — safe to run multiple times.
"""

import argparse
import site
import sys
from pathlib import Path


def find_cluster_backend() -> Path:
    """Locate FunASR's cluster_backend.py."""
    try:
        import funasr
        base = Path(funasr.__file__).parent
        target = base / "models" / "campplus" / "cluster_backend.py"
        if target.exists():
            return target
    except ImportError as e:
        print(f"  Warning: funasr installed but could not be imported ({e}); searching site-packages...")

    # Fallback: search site-packages
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        target = Path(sp) / "funasr" / "models" / "campplus" / "cluster_backend.py"
        if target.exists():
            return target

    return None


ORIGINAL_EIGSH = "lambdas, eig_vecs = scipy.linalg.eigh(L)"
PATCHED_EIGSH = """n = L.shape[0]
        num_eigs_needed = self.max_num_spks + 1 if k_oracle is None else k_oracle

        # Use sparse eigsh for large matrices — O(N^2*k) vs O(N^3)
        if n > 1024 and num_eigs_needed < n // 2:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            L_sparse = csr_matrix(L)
            lambdas, eig_vecs = eigsh(L_sparse, k=num_eigs_needed, which='SM')
            idx = np.argsort(lambdas)
            lambdas = lambdas[idx]
            eig_vecs = eig_vecs[:, idx]
        else:
            lambdas, eig_vecs = scipy.linalg.eigh(L)"""

ORIGINAL_PRUNING = """        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0"""
PATCHED_PRUNING = """        # Vectorized: zero out the smallest n_elems values per row
        sorted_indices = np.argsort(A, axis=1)
        low_indices = sorted_indices[:, :n_elems]
        rows = np.arange(A.shape[0])[:, None]
        A[rows, low_indices] = 0"""


def patch_file(path: Path) -> bool:
    content = path.read_text(encoding="utf-8")
    changed = False

    if ORIGINAL_EIGSH in content and "eigsh" not in content:
        content = content.replace(ORIGINAL_EIGSH, PATCHED_EIGSH)
        changed = True
        print("  Patched: get_spec_embs() -> sparse eigsh")
    elif "eigsh" in content:
        print("  Already patched: sparse eigsh")
    else:
        print("  WARNING: Could not locate eigsh patch target — FunASR may have been updated")

    if "for i in range(A.shape[0]):" in content and "Vectorized" not in content:
        content = content.replace(ORIGINAL_PRUNING, PATCHED_PRUNING)
        changed = True
        print("  Patched: p_pruning() -> vectorized")
    elif "Vectorized" in content:
        print("  Already patched: vectorized pruning")
    else:
        print("  WARNING: Could not locate pruning patch target — FunASR may have been updated")

    if changed:
        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            print(f"  Error: Permission denied writing {path}")
            print("  If FunASR was installed with sudo, try: sudo python3 patch_clustering.py --yes")
            return False
        # Clear bytecode cache
        cache = path.parent / "__pycache__"
        if cache.exists():
            for pyc in cache.glob(f"{path.stem}*.pyc"):
                pyc.unlink()
                print(f"  Cleared: {pyc.name}")

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Patch FunASR's spectral clustering for sparse eigenvalue decomposition.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    target = find_cluster_backend()
    if target is None:
        print("Error: funasr not installed or cluster_backend.py not found")
        sys.exit(1)

    print(f"Found: {target}")

    if not args.yes:
        print(f"\nThis will modify the installed FunASR package file:\n  {target}")
        print("The patch replaces O(N³) eigenvalue decomposition with O(N²·k) for large matrices.")
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted (non-interactive or interrupted).")
            sys.exit(2)
        if resp not in ("y", "yes"):
            print("Aborted.")
            sys.exit(2)

    patch_file(target)
    print("Done.")


if __name__ == "__main__":
    main()
