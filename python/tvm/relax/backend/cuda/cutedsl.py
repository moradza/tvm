# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Pattern table and BYOC codegen for CuTe DSL AOT GEMM kernels.

Works exactly like the cuBLAS backend:

    from tvm.relax.backend.cuda import cutedsl  # registers patterns + codegen
    patterns = get_patterns_with_prefix("cutedsl")
    mod = FuseOpsByPattern(patterns, annotate_codegen=True)(mod)
    mod = RunCodegen()(mod)

``relax.ext.cutedsl`` (registered below) receives the fused
``Codegen="cutedsl"`` functions, derives one GemmSpec per function straight
from the composite body (n/k from the weight, dtypes from the IR, epilogue
from the composite name), AOT-compiles the kernels in a SUBPROCESS
(content-cached; torch/cutlass never enter this process), and returns a
static-library runtime module. export_library links the kernels INTO the
final artifact — serving needs only ``tvm.runtime.load_module`` plus the
link options from :func:`get_link_options` at export time (the kernels
reference libcute_dsl_runtime / libtvm_ffi / the CUDA driver).

Requires the ``cutedsl-aot`` project (``cute_tvm_aot`` package + builder
CLIs + the Hopper persistent GEMM kernel) importable on sys.path; the heavy
stack (torch, nvidia-cutlass-dsl) is only needed by the builder subprocess.
"""

import glob
import importlib.util
from pathlib import Path

import tvm
import tvm_ffi
from tvm.relax.transform import PatternCheckContext

from ..pattern_registry import register_patterns
from ..patterns import make_matmul_pattern
from ..utils import has_leaking_intermediate_variables

# composite-name suffix -> GemmSpec epilogue (ordering: register the FUSED
# variants LAST — the registry returns patterns most-recently-registered
# first, and FuseOpsByPattern tries them in that order)
_EPILOGUE_BY_SUFFIX = {"_relu": "relu"}


def _cutedsl_aot_home() -> Path:
    """Root of the cutedsl-aot project (parent of the cute_tvm_aot package)."""
    spec = importlib.util.find_spec("cute_tvm_aot")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "cutedsl BYOC needs the cutedsl-aot project on sys.path "
            "(the cute_tvm_aot package plus byoc_gemm_builder_cli.py)"
        )
    return Path(list(spec.submodule_search_locations)[0]).parent


def _static_int(dim):
    return int(dim) if isinstance(dim, tvm.tirx.IntImm) else None


def _gemm_spec_or_none(n, k, in_dtype, out_dtype, epilogue):
    """Validate against what the kernel supports; None = leave for cuBLAS."""
    from cute_tvm_aot.spec import GemmSpec  # deferred: needs sys.path setup

    try:
        return GemmSpec(n=n, k=k, in_dtype=in_dtype, out_dtype=out_dtype,
                        epilogue=epilogue)
    except (ValueError, NotImplementedError):
        return None


#: Dispatch policy, set via :func:`set_dispatch_policy` (explicit API — the
#: pattern checks have no other config channel). A fully-STATIC-M call-site
#: is claimed only when M >= min_static_m: the persistent GEMM wins at large
#: M and loses badly at decode-sized M (M = batch), while symbolic-M sites
#: (prefill) are always claimed, matching the old campaign's
#: symbolic-M-first routing. `shapes`, when set, restricts dispatch to those
#: (n, k) weight shapes.
#: Per-(n, k) kernel specialization (tile / cluster / swizzle) — where the
#: persistent GEMM's performance lives. Defaults are the H100
#: bench_persistent_sweep winners from the TinyLlama campaign; anything not
#: listed builds with GemmSpec defaults (tile (128,128), cluster (1,1),
#: swizzle 1), which is CORRECT but typically ~15-20% off tuned.
_DEFAULT_SPECIALIZATIONS = {
    (11264, 2048): {"tile": [128, 256], "cluster": [2, 2], "swizzle": 8},
    (2048, 5632): {"tile": [128, 256], "cluster": [2, 1], "swizzle": 1},
    (2560, 2048): {"tile": [128, 256], "cluster": [2, 1], "swizzle": 1},
    (2048, 2048): {"tile": [128, 256], "cluster": [2, 1], "swizzle": 1},
}

_POLICY = {"min_static_m": 1024, "shapes": None,
           "specializations": dict(_DEFAULT_SPECIALIZATIONS)}


def set_dispatch_policy(min_static_m: int = None, shapes=None, specializations=None):
    """Configure which matmul call-sites the cutedsl patterns claim and how
    the kernels are specialized.

    Parameters
    ----------
    min_static_m : int, optional
        Minimum M (product of lhs lead dims) for fully-static call-sites.
        Symbolic-M sites are always eligible.
    shapes : iterable of (n, k), optional
        If given, only these weight shapes are claimed; None = all feasible.
    specializations : dict, optional
        {(n, k): {"tile": [m, n], "cluster": [x, y], "swizzle": int}}
        MERGED over the built-in sweep-winner defaults.
    """
    if min_static_m is not None:
        _POLICY["min_static_m"] = int(min_static_m)
    _POLICY["shapes"] = None if shapes is None else {tuple(map(int, s)) for s in shapes}
    if specializations is not None:
        merged = dict(_DEFAULT_SPECIALIZATIONS)
        merged.update({tuple(map(int, k)): dict(v) for k, v in specializations.items()})
        _POLICY["specializations"] = merged


def _check_matmul(context: PatternCheckContext) -> bool:
    """Predicate for cutedsl.matmul_transposed[_relu]."""
    if has_leaking_intermediate_variables(context):
        return False
    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]
    matmul_call = context.annotated_expr["root"]

    if lhs.ty is None or rhs.ty is None:
        return False
    # x is (M, K) or row-major (B, S, K) (flattened to M = B*S inside the
    # kernel wrapper); weight is 2D (N, K).
    if lhs.ty.ndim not in (2, 3) or rhs.ty.ndim != 2:
        return False
    n, k = _static_int(rhs.ty.shape[0]), _static_int(rhs.ty.shape[1])
    if n is None or k is None or _static_int(lhs.ty.shape[-1]) != k:
        return False
    if _POLICY["shapes"] is not None and (n, k) not in _POLICY["shapes"]:
        return False
    # static-M threshold (symbolic lead dims -> always eligible)
    lead = [_static_int(d) for d in list(lhs.ty.shape)[:-1]]
    if all(d is not None for d in lead):
        m = 1
        for d in lead:
            m *= d
        if m < _POLICY["min_static_m"]:
            return False

    # permute_dims must be a real transpose (identity-permute hole on
    # square weights): the matched rhs var binds to the permute call
    perm_var = matmul_call.args[1]
    if perm_var in context.matched_bindings:
        perm_call = context.matched_bindings[perm_var]
        axes = perm_call.attrs["axes"]
        if axes is not None and list(axes) != [1, 0]:
            return False

    in_dt, w_dt = str(lhs.ty.dtype), str(rhs.ty.dtype)
    out_dt = str(matmul_call.ty.dtype)
    if in_dt != w_dt:
        return False
    # epilogue-agnostic feasibility check (relu shares the constraints)
    return _gemm_spec_or_none(n, k, in_dt, out_dt, None) is not None


register_patterns(
    [
        # most-specific LAST (highest priority — see _EPILOGUE_BY_SUFFIX note)
        (
            "cutedsl.matmul_transposed",
            *make_matmul_pattern(with_bias=False, transposed_rhs=True),
            _check_matmul,
        ),
        (
            "cutedsl.matmul_transposed_relu",
            *make_matmul_pattern(with_bias=False, activation="relax.nn.relu",
                                 transposed_rhs=True),
            _check_matmul,
        ),
    ]
)


def _parse_codegen_function(fn) -> dict:
    """One BYOC builder spec entry from a fused Codegen='cutedsl' function."""
    relax = tvm.relax
    sym = fn.attrs["global_symbol"]

    # fused-fn body shape (FuseOpsByPattern, annotate_codegen=True):
    #   local_func = <Function Composite="cutedsl.*">
    #   output     = local_func(<outer params...>)
    inner, inner_var, inner_call = None, None, None
    for block in fn.body.blocks:
        for binding in block.bindings:
            value = getattr(binding, "value", None)
            if isinstance(value, relax.Function) and "Composite" in (value.attrs or {}):
                inner, inner_var = value, binding.var
            elif inner_var is not None and isinstance(value, relax.Call) \
                    and value.op.same_as(inner_var):
                inner_call = value
    assert inner is not None and inner_call is not None, f"unrecognized fused fn {sym}"

    composite = str(inner.attrs["Composite"])
    epilogue = None
    for suffix, epi in _EPILOGUE_BY_SUFFIX.items():
        if composite.endswith(suffix):
            epilogue = epi
    # composite body: [wt = permute_dims(w)] , mm = matmul(x, wt) [, act]
    mm, perm_arg0 = None, {}
    for block in inner.body.blocks:
        for binding in block.bindings:
            value = getattr(binding, "value", None)
            if isinstance(value, relax.Call) and isinstance(value.op, tvm.ir.Op):
                if value.op.name == "relax.permute_dims":
                    perm_arg0[binding.var] = value.args[0]
                elif value.op.name == "relax.matmul":
                    mm = value
    assert mm is not None, f"no matmul in composite of {sym}"
    x_inner, w_inner = mm.args[0], perm_arg0[mm.args[1]]

    def outer_param_index(inner_param):
        inner_idx = list(inner.params).index(inner_param)
        outer_var = inner_call.args[inner_idx]
        return list(fn.params).index(outer_var)

    x_param, w_param = outer_param_index(x_inner), outer_param_index(w_inner)
    w_ty = fn.params[w_param].ty
    return {
        "symbol": str(sym),
        "n": int(w_ty.shape[0]),
        "k": int(w_ty.shape[1]),
        "in_dtype": str(fn.params[x_param].ty.dtype),
        "out_dtype": str(fn.ret_ty.dtype),
        "epilogue": epilogue,
        "x_param": x_param,
        "w_param": w_param,
        "x_ndim": int(fn.params[x_param].ty.ndim),
        # per-shape tile/cluster/swizzle from the policy table (may be {})
        **_POLICY["specializations"].get((int(w_ty.shape[0]), int(w_ty.shape[1])), {}),
    }


@tvm_ffi.register_global_func("relax.ext.cutedsl")
def _cutedsl_codegen(functions, options, constant_names):  # pylint: disable=unused-argument
    """RunCodegen hook: fused functions -> one static-library module."""
    from cute_tvm_aot.cache import ensure_library

    home = _cutedsl_aot_home()
    entries = [_parse_codegen_function(fn) for fn in functions]
    # sample_m only sizes the tracing tensors — M is exported dynamic, so
    # any value works; override via RunCodegen target_options if desired:
    # RunCodegen({"cutedsl": {"sample_m": 4096}})
    sample_m = int(options["sample_m"]) if options and "sample_m" in options else 8192
    obj_path = ensure_library(
        str(home / "byoc_gemm_builder_cli.py"),
        {"sample_m": sample_m, "functions": entries},
    )
    symbols = [e["symbol"] for e in entries]
    return [tvm.runtime.load_static_library(obj_path, func_names=symbols)]


def get_link_options() -> list:
    """Extra linker options for export_library: the kernel objects reference
    libcute_dsl_runtime (CuTe DSL C runtime; no Python) and libtvm_ffi."""
    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    assert spec and spec.submodule_search_locations, "nvidia-cutlass-dsl not installed"
    pkg = list(spec.submodule_search_locations)[0]
    hits = glob.glob(str(Path(pkg) / "cu*" / "lib" / "libcute_dsl_runtime.so"))
    assert hits, f"libcute_dsl_runtime.so not found under {pkg}"
    dsl_libdir = str(Path(hits[0]).parent)

    import tvm_ffi.libinfo as _li

    ffi_libdir = str(Path(_li.find_libtvm_ffi()).parent)
    return [
        f"-L{dsl_libdir}", f"-Wl,-rpath,{dsl_libdir}", "-lcute_dsl_runtime",
        f"-L{ffi_libdir}", f"-Wl,-rpath,{ffi_libdir}", "-ltvm_ffi",
    ]
