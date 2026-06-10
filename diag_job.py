"""Diagnostic: print numba / cuda-python / nvvm state to debug the RNNT-loss JIT failure."""
import glob
import os
import sys

for d in sys.path:
    hits = glob.glob(os.path.join(d, "nvidia", "cuda_nvcc", "nvvm", "lib64", "libnvvm.so*"))
    if hits:
        home = os.path.join(d, "nvidia", "cuda_nvcc")
        libdir = os.path.join(home, "nvvm", "lib64")
        os.environ["CUDA_HOME"] = home
        os.environ["LD_LIBRARY_PATH"] = libdir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        un = os.path.join(libdir, "libnvvm.so")
        if not os.path.exists(un):
            try:
                os.symlink(hits[0], un)
            except OSError:
                pass
        print("CUDA_HOME", home, "| libnvvm", [os.path.basename(h) for h in hits], flush=True)
        break

import numba
print("numba", numba.__version__, flush=True)
for pkg in ("numba_cuda", "cuda", "cuda.bindings", "ptxcompiler"):
    try:
        m = __import__(pkg)
        print("OK import", pkg, getattr(m, "__version__", "?"), flush=True)
    except Exception as e:
        print("FAIL import", pkg, repr(e)[:140], flush=True)

from numba import cuda
print("cuda.is_available", cuda.is_available(), flush=True)
try:
    print("detect:", flush=True)
    cuda.detect()
except Exception as e:
    print("detect err", repr(e)[:200], flush=True)
try:
    from numba.cuda.cudadrv import nvvm
    n = nvvm.NVVM()
    print("nvvm get_version", n.get_version(), flush=True)
    print("supported_ccs", nvvm.get_supported_ccs(), flush=True)
except Exception as e:
    print("nvvm err", repr(e)[:300], flush=True)
