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

"""Tests for the CuTe DSL AOT BYOC backend (relax.backend.cuda.cutedsl).

Pattern/predicate tests run on any machine with the cutedsl-aot project on
sys.path (set CUTEDSL_AOT_HOME or have cute_tvm_aot importable). The E2E
test additionally needs an sm_90 GPU + nvidia-cutlass-dsl and builds a real
kernel (content-cached across runs).
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

import tvm
from tvm import relax as R

# --- locate the cutedsl-aot project ----------------------------------------
if importlib.util.find_spec("cute_tvm_aot") is None:
    _home = os.environ.get("CUTEDSL_AOT_HOME")
    if _home:
        sys.path.insert(0, _home)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cute_tvm_aot") is None,
    reason="cutedsl-aot project not on sys.path (set CUTEDSL_AOT_HOME)",
)

M, N, K = 512, 256, 784  # n%8==0, k%8==0: satisfies fp16 TMA alignment
BAD_N = 10  # fails n%8 alignment -> pattern check must reject


def _linear_relu_module(dtype="float16", n=N, k=K, extra_lead_dim=None):
    """main: relu(matmul(x, permute_dims(w))) built with BlockBuilder."""
    bb = R.BlockBuilder()
    x_shape = (M, k) if extra_lead_dim is None else (extra_lead_dim, M, k)
    x = R.Var("x", R.TensorType(x_shape, dtype))
    w = R.Var("w", R.TensorType((n, k), dtype))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            wt = bb.emit(R.op.permute_dims(w))
            mm = bb.emit(R.op.matmul(x, wt))
            out = bb.emit_output(R.op.nn.relu(mm))
        bb.emit_func_output(out)
    return bb.finalize()


def _apply_cutedsl_patterns(mod):
    from tvm.relax.backend.cuda import cutedsl  # noqa: F401  (registration)

    patterns = R.backend.get_patterns_with_prefix("cutedsl")
    assert patterns, "cutedsl pattern registry is empty"
    return R.transform.FuseOpsByPattern(patterns, annotate_codegen=True)(mod)


def test_pattern_priority_fused_relu_first():
    """The registry must yield the relu-fused pattern before the bare one
    (most-recently-registered first); otherwise the bare pattern silently
    steals relu(matmul) sites."""
    from tvm.relax.backend.cuda import cutedsl  # noqa: F401

    names = [p.name for p in R.backend.get_patterns_with_prefix("cutedsl")]
    assert names.index("cutedsl.matmul_transposed_relu") < names.index(
        "cutedsl.matmul_transposed"
    )


def test_fuse_claims_relu_site():
    mod = _apply_cutedsl_patterns(_linear_relu_module())
    txt = mod.script()
    assert 'Codegen": "cutedsl' in txt or "Codegen" in txt
    assert "cutedsl.matmul_transposed_relu" in txt
    # the relu is inside the composite, not a standalone op in main
    assert "R.nn.relu" not in mod["main"].script()


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"n": BAD_N}, "n fails TMA alignment"),
        ({"dtype": "bfloat16"}, "kernel has no bf16"),
        ({"extra_lead_dim": 4}, "rank-3 lhs not supported by v1 BYOC contract"),
    ],
)
def test_check_rejects(kwargs, reason):
    mod = _apply_cutedsl_patterns(_linear_relu_module(**kwargs))
    assert "cutedsl" not in mod["main"].script(), reason


def test_byoc_e2e_numerics():
    """Full pipeline: fuse -> RunCodegen (real kernel build, cached) ->
    compile -> export (kernels linked in) -> load -> numerics vs numpy."""
    torch = pytest.importorskip("torch")  # noqa: F841  (builder needs CUDA torch)
    pytest.importorskip("cutlass")
    if not tvm.cuda(0).exist:
        pytest.skip("no CUDA device")

    from tvm.relax.backend.cuda import cutedsl

    mod = _linear_relu_module()
    mod = _apply_cutedsl_patterns(mod)
    mod = R.transform.RunCodegen()(mod)
    assert "external_mods" in mod.attrs

    device = tvm.cuda(0)
    target = tvm.target.Target.from_device(device)
    with target:
        mod = R.get_pipeline("zero")(mod)
    ex = tvm.compile(mod, target=target)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        so = os.path.join(tmp, "mod.so")
        ex.export_library(so, workspace_dir=tmp, options=cutedsl.get_link_options())
        lib = tvm.runtime.load_module(so)

    vm = R.VirtualMachine(lib, device)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((M, K), np.float32).astype(np.float16)
    w = rng.standard_normal((N, K), np.float32).astype(np.float16)
    out = vm["main"](tvm.runtime.tensor(x, device), tvm.runtime.tensor(w, device))
    got = out.numpy().astype(np.float32)
    ref = np.maximum(x.astype(np.float32) @ w.astype(np.float32).T, 0.0)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 2e-3, f"scaled max-abs err {err:.2e}"
    assert (got >= 0).all(), "relu epilogue produced negatives"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
