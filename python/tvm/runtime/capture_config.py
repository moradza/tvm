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

"""Explicit runtime configuration for whole-step CUDA graph capture.

These are first-class APIs (not env vars): call them from the serving
process before the relevant objects are created.
"""

import tvm_ffi


def set_kv_cache_pinned_aux(enable: bool, slot_elems: int = 4096) -> None:
    """Enable pinned-aux mode for KV caches created AFTER this call.

    In pinned-aux mode every attention auxiliary array kind gets a fixed
    slot in the device aux arena (stable addresses across steps) and aux
    uploads plus the FlashInfer plan run on the compute stream — the three
    properties whole-step CUDA graph capture of decode requires. Without
    it, replayed graphs read aux arrays at stale capture-time offsets and
    silently corrupt KV appends.

    Parameters
    ----------
    enable : bool
        Whether newly created paged KV caches use pinned-aux mode.
    slot_elems : int
        Per-slot capacity in aux elements (default 4096). Every pinned slot
        is uploaded every step, so keep it small; raise it when a single
        aux array can exceed it (e.g. page_indices needs >= the total page
        count of the cache).
    """
    tvm_ffi.get_global_func("vm.builtin.paged_attention_kv_cache_set_pinned_aux")(
        bool(enable), int(slot_elems)
    )


def get_kv_cache_pinned_aux() -> bool:
    """Return whether newly created paged KV caches use pinned-aux mode."""
    return bool(tvm_ffi.get_global_func("vm.builtin.paged_attention_kv_cache_get_pinned_aux")())


def pool_prewarm(device, nbytes: int, count: int) -> None:
    """Pre-stock `count` pooled-allocator entries of `nbytes` each on `device`.

    Steady-state allocations of that size then hit the pool instead of
    cudaMalloc — required before capturing a forward pass with CUDA graphs,
    where a fresh cudaMalloc is illegal. Call once per (size, count) need,
    e.g. ``pool_prewarm(tvm.cuda(0), 4096, 64)``.
    """
    tvm_ffi.get_global_func("vm.builtin.memory_manager.pool_prewarm")(
        device, int(nbytes), int(count)
    )
