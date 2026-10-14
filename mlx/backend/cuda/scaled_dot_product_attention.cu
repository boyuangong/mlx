// Copyright © 2025 Apple Inc.

// Required for using M_LOG2E in MSVC.
#define _USE_MATH_DEFINES

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <fmt/format.h>

#include <algorithm>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

#define PRAGMA_LOOP_UNROLL #pragma unroll

// Whether rows starting at ptr and spaced row_stride elements apart can be
// read N elements per lane with a single vector load.
template <int N, typename T>
__device__ __forceinline__ bool vector_rows(const T* ptr, int64_t row_stride) {
  if constexpr ((N & (N - 1)) == 0) {
    return is_aligned<N>(ptr) && row_stride % N == 0;
  } else {
    return false;
  }
}

// Calls f(i, ptr[N * lane + i]) for i in [0, N), with one vector load when
// vector_rows() allowed it.
template <int N, typename T, typename F>
__device__ __forceinline__ void
for_each_lane_value(const T* ptr, int lane, bool vectorized, F&& f) {
  if constexpr ((N & (N - 1)) == 0) {
    if (vectorized) {
      auto v = unsafe_load_vector<N>(ptr, lane);
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < N; i++) {
        f(i, v[i]);
      }
      return;
    }
  }
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < N; i++) {
    f(i, ptr[N * lane + i]);
  }
}

struct AttnParams {
  int B;
  int H;
  int D;

  int qL;
  int kL;

  int gqa_factor;
  float scale;

  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

// The launch bounds match the block sizes used by the launchers below. At
// D = 512 the 1024-thread kernels sit at the 64 registers per thread such a
// block allows, so let the compiler spill rather than fail to launch.
template <typename T, bool do_causal, int Dk, int Dv>
__global__ void __launch_bounds__(1024) kernel_sdpav_1pass(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    const T* sinks,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 32;
  constexpr int BD = 32;

  static_assert(Dk % BD == 0 && Dv % BD == 0);
  constexpr int qk_per_thread = Dk / BD;
  constexpr int v_per_thread = Dv / BD;

  const int inner_k_stride = BN * int(params.K_strides[2]);
  const int inner_v_stride = BN * int(params.V_strides[2]);

  typedef float U;

  U q[qk_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * M_LOG2E;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;

  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = warp_idx;

  Q += batch_idx * params.Q_strides[0] + // Batch
      head_idx * params.Q_strides[1] + // Head
      q_seq_idx * params.Q_strides[2]; // Sequence

  K += batch_idx * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1] + // Head
      kv_seq_idx * params.K_strides[2]; // Sequence

  V += batch_idx * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1] + // Head
      kv_seq_idx * params.V_strides[2]; // Sequence

  O += batch_idx * params.O_strides[0] + // Batch
      head_idx * params.O_strides[1] + // Head
      q_seq_idx * params.O_strides[2]; // Sequence

  // Read the query and 0 the output accumulator
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = scale_log2 * static_cast<U>(Q[qk_per_thread * lane_idx + i]);
  }

  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0.f;
  }

  U max_score = Limits<U>::finite_min();
  U sum_exp_score = 0.f;
  if (sinks && warp_idx == 0) {
    max_score = M_LOG2E * static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  const bool k_vec = vector_rows<qk_per_thread>(K, params.K_strides[2]);
  const bool v_vec = vector_rows<v_per_thread>(V, params.V_strides[2]);

  // For each key
  for (int i = kv_seq_idx; i < params.kL; i += BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (params.kL - params.qL + q_seq_idx);
    }

    if (use_key) {
      // Compute the i-th score, reading the key straight into the dot
      // product so it does not occupy registers for large head dims.
      U score = 0.f;
      for_each_lane_value<qk_per_thread>(K, lane_idx, k_vec, [&](int j, T x) {
        score += q[j] * static_cast<U>(x);
      });

      // Warp sum
      score = cg::reduce(warp, score, cg::plus<U>());

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = exp2f(max_score - new_max);
      U exp_score = exp2f(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for_each_lane_value<v_per_thread>(V, lane_idx, v_vec, [&](int j, T x) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(x);
      });
    }

    // Move the pointers to the next kv
    K += inner_k_stride;
    V += inner_v_stride;
  }

  if (lane_idx == 0) {
    max_scores[warp_idx] = max_score;
    sum_exp_scores[warp_idx] = sum_exp_score;
  }
  block.sync();

  max_score = max_scores[lane_idx];
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  sum_exp_score =
      cg::reduce(warp, sum_exp_scores[lane_idx] * factor, cg::plus<U>());
  sum_exp_score = sum_exp_score == 0 ? 0 : __frcp_rn(sum_exp_score);

  // Now we need to aggregate all the outputs
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][warp_idx] = o[i];
    block.sync();
    U ot = outputs[warp_idx][lane_idx] * factor;
    o[i] = cg::reduce(warp, ot, cg::plus<U>()) * sum_exp_score;
    block.sync();
  }

  // And write the output
  if (lane_idx == 0) {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      O[v_per_thread * warp_idx + i] = static_cast<T>(o[i]);
    }
  }
}

// `fixed_blocks` instantiates the kernels for exactly 32 key blocks, the
// common case. With the block count known at compile time the first pass
// needs fewer registers, which on sm_86/89 is one more resident thread block
// per SM for head dim 256.
template <typename T, bool do_causal, int Dk, int Dv, bool fixed_blocks>
__global__ void __launch_bounds__(256) kernel_sdpav_2pass_1(
    const T* Q,
    const T* K,
    const T* V,
    const T* sinks,
    float* partials,
    float* sums,
    float* maxs,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  // The keys are split into `blocks` interleaved chunks of BN rows, one
  // thread block each along z for every batch element.
  const int blocks = fixed_blocks ? 32 : gridDim.z / params.B;

  static_assert(Dk % BD == 0 && Dv % BD == 0);
  constexpr int qk_per_thread = Dk / BD;
  constexpr int v_per_thread = Dv / BD;

  const int inner_k_stride = blocks * BN * int(params.K_strides[2]);
  const int inner_v_stride = blocks * BN * int(params.V_strides[2]);

  typedef float U;

  U q[qk_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * 1.44269504089f;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z / blocks;
  const int block_idx = blockIdx.z % blocks;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;

  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = block_idx * BN + warp_idx;

  Q += batch_idx * params.Q_strides[0] + // Batch
      head_idx * params.Q_strides[1] + // Head
      q_seq_idx * params.Q_strides[2]; // Sequence

  K += batch_idx * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1] + // Head
      kv_seq_idx * params.K_strides[2]; // Sequence

  V += batch_idx * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1] + // Head
      kv_seq_idx * params.V_strides[2]; // Sequence

  const int p_stride_s = blocks;
  const int p_stride_h = params.qL * p_stride_s;
  const int p_stride_b = params.H * p_stride_h;
  const int p_offset = batch_idx * p_stride_b + // Batch
      head_idx * p_stride_h + // Head
      q_seq_idx * p_stride_s + // Sequence
      block_idx; // Block

  partials += p_offset * Dv;
  sums += p_offset;
  maxs += p_offset;

  // Read the query and 0 the output accumulator
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = scale_log2 * static_cast<U>(Q[qk_per_thread * lane_idx + i]);
  }

  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0.f;
  }

  U max_score = Limits<U>::finite_min();
  U sum_exp_score = 0.f;
  if (sinks && warp_idx == 0 && block_idx == 0) {
    max_score = M_LOG2E * static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  const bool k_vec = vector_rows<qk_per_thread>(K, params.K_strides[2]);
  const bool v_vec = vector_rows<v_per_thread>(V, params.V_strides[2]);

  // For each key
  for (int i = kv_seq_idx; i < params.kL; i += blocks * BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (params.kL - params.qL + q_seq_idx);
    }

    if (use_key) {
      // Compute the i-th score, reading the key straight into the dot
      // product so it does not occupy registers for large head dims.
      U score = 0.f;
      for_each_lane_value<qk_per_thread>(K, lane_idx, k_vec, [&](int j, T x) {
        score += q[j] * static_cast<U>(x);
      });

      // Warp sum
      score = cg::reduce(warp, score, cg::plus<U>());

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = exp2f(max_score - new_max);
      U exp_score = exp2f(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for_each_lane_value<v_per_thread>(V, lane_idx, v_vec, [&](int j, T x) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(x);
      });
    }

    // Move the pointers to the next kv
    K += inner_k_stride;
    V += inner_v_stride;
  }

  if (lane_idx == 0) {
    max_scores[warp_idx] = max_score;
    sum_exp_scores[warp_idx] = sum_exp_score;
  }

  block.sync();

  max_score = (lane_idx < BN) ? max_scores[lane_idx] : -1e9;
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  sum_exp_score = (lane_idx < BN) ? sum_exp_scores[lane_idx] : 0.f;
  sum_exp_score = cg::reduce(warp, sum_exp_score * factor, cg::plus<U>());

  // Write the sum and new max
  if (warp_idx == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  auto ff = exp2f(max_scores[warp_idx] - new_max);
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[warp_idx][lane_idx] = o[i] * ff;
    block.sync();

    if (warp_idx == 0) {
      U ot = outputs[0][lane_idx];
      PRAGMA_LOOP_UNROLL
      for (int j = 1; j < BN; j++) {
        ot += outputs[j][lane_idx];
        warp.sync();
      }
      o[i] = ot;
    }
    block.sync();
  }

  if (warp_idx == 0) {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      partials[v_per_thread * lane_idx + i] = o[i];
    }
  }
}

// First pass for single-query decode with grouped-query attention. Each
// warp holds HPT query heads that share a key/value head, so every key and
// value row is read once for all of them instead of once per query head.
// The grid runs over (kv head, group of HPT query heads) along x; the
// partials it writes have the same layout as kernel_sdpav_2pass_1.
template <typename T, int Dk, int Dv, int HPT, bool fixed_blocks>
__global__ void __launch_bounds__(256) kernel_sdpav_2pass_1_gqa(
    const T* Q,
    const T* K,
    const T* V,
    const T* sinks,
    float* partials,
    float* sums,
    float* maxs,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  const int blocks = fixed_blocks ? 32 : gridDim.z / params.B;

  static_assert(Dk % BD == 0 && Dv % BD == 0);
  constexpr int qk_per_thread = Dk / BD;
  constexpr int v_per_thread = Dv / BD;

  const int inner_k_stride = blocks * BN * int(params.K_strides[2]);
  const int inner_v_stride = blocks * BN * int(params.V_strides[2]);

  typedef float U;

  U q[HPT][qk_per_thread];
  U o[HPT][v_per_thread];
  U max_score[HPT];
  U sum_exp_score[HPT];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * 1.44269504089f;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  const int groups_per_kv_head = params.gqa_factor / HPT;
  const int batch_idx = blockIdx.z / blocks;
  const int block_idx = blockIdx.z % blocks;
  const int kv_head_idx = blockIdx.x / groups_per_kv_head;
  const int head0 =
      kv_head_idx * params.gqa_factor + (blockIdx.x % groups_per_kv_head) * HPT;
  const int kv_seq_idx = block_idx * BN + warp_idx;

  K += batch_idx * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1] + // Head
      kv_seq_idx * params.K_strides[2]; // Sequence

  V += batch_idx * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1] + // Head
      kv_seq_idx * params.V_strides[2]; // Sequence

  // Read the queries and 0 the output accumulators
  PRAGMA_LOOP_UNROLL
  for (int h = 0; h < HPT; h++) {
    const T* q_ptr =
        Q + batch_idx * params.Q_strides[0] + (head0 + h) * params.Q_strides[1];
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < qk_per_thread; i++) {
      q[h][i] =
          scale_log2 * static_cast<U>(q_ptr[qk_per_thread * lane_idx + i]);
    }
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      o[h][i] = 0.f;
    }
    max_score[h] = Limits<U>::finite_min();
    sum_exp_score[h] = 0.f;
    if (sinks && warp_idx == 0 && block_idx == 0) {
      max_score[h] = M_LOG2E * static_cast<U>(sinks[head0 + h]);
      sum_exp_score[h] = 1.f;
    }
  }

  const bool k_vec = vector_rows<qk_per_thread>(K, params.K_strides[2]);
  const bool v_vec = vector_rows<v_per_thread>(V, params.V_strides[2]);

  // With a single query every key is visible, causal or not.
  for (int i = kv_seq_idx; i < params.kL; i += blocks * BN) {
    U k[qk_per_thread];
    for_each_lane_value<qk_per_thread>(
        K, lane_idx, k_vec, [&](int j, T x) { k[j] = static_cast<U>(x); });

    U exp_score[HPT];
    U factor[HPT];
    PRAGMA_LOOP_UNROLL
    for (int h = 0; h < HPT; h++) {
      U score = 0.f;
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[h][j] * k[j];
      }
      score = cg::reduce(warp, score, cg::plus<U>());
      U new_max = max(max_score[h], score);
      factor[h] = exp2f(max_score[h] - new_max);
      exp_score[h] = exp2f(score - new_max);
      max_score[h] = new_max;
      sum_exp_score[h] = sum_exp_score[h] * factor[h] + exp_score[h];
    }

    for_each_lane_value<v_per_thread>(V, lane_idx, v_vec, [&](int j, T x) {
      U vx = static_cast<U>(x);
      PRAGMA_LOOP_UNROLL
      for (int h = 0; h < HPT; h++) {
        o[h][j] = o[h][j] * factor[h] + exp_score[h] * vx;
      }
    });

    K += inner_k_stride;
    V += inner_v_stride;
  }

  // Combine the warps of the block, one query head at a time.
  const int p_stride_s = blocks;
  const int p_stride_h = params.qL * p_stride_s;
  const int p_stride_b = params.H * p_stride_h;
  PRAGMA_LOOP_UNROLL
  for (int h = 0; h < HPT; h++) {
    const int p_offset =
        batch_idx * p_stride_b + (head0 + h) * p_stride_h + block_idx;

    if (lane_idx == 0) {
      max_scores[warp_idx] = max_score[h];
      sum_exp_scores[warp_idx] = sum_exp_score[h];
    }
    block.sync();

    U m = (lane_idx < BN) ? max_scores[lane_idx] : -1e9;
    U new_max = cg::reduce(warp, m, cg::greater<U>());
    U f = exp2f(m - new_max);
    U s = (lane_idx < BN) ? sum_exp_scores[lane_idx] : 0.f;
    s = cg::reduce(warp, s * f, cg::plus<U>());
    if (warp_idx == 0 && lane_idx == 0) {
      sums[p_offset] = s;
      maxs[p_offset] = new_max;
    }

    auto ff = exp2f(max_scores[warp_idx] - new_max);
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      outputs[warp_idx][lane_idx] = o[h][i] * ff;
      block.sync();
      if (warp_idx == 0) {
        U ot = outputs[0][lane_idx];
        PRAGMA_LOOP_UNROLL
        for (int j = 1; j < BN; j++) {
          ot += outputs[j][lane_idx];
        }
        partials[p_offset * Dv + v_per_thread * lane_idx + i] = ot;
      }
      block.sync();
    }
  }
}

template <typename T, bool do_causal, int Dv, bool fixed_blocks>
__global__ void __launch_bounds__(1024) kernel_sdpav_2pass_2(
    const float* partials,
    const float* sums,
    const float* maxs,
    T* O,
    __grid_constant__ const AttnParams params,
    const int num_blocks) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  const int blocks = fixed_blocks ? 32 : num_blocks;

  static_assert(Dv % BD == 0);
  constexpr int v_per_thread = Dv / BD;

  typedef float U;

  U o[v_per_thread];
  __shared__ U outputs[BN][BD + 1];

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int q_seq_idx = blockIdx.y;

  const int p_stride_s = blocks;
  const int p_stride_h = params.qL * p_stride_s;
  const int p_stride_b = params.H * p_stride_h;
  const int p_offset = batch_idx * p_stride_b + // Batch
      head_idx * p_stride_h + // Head
      q_seq_idx * p_stride_s; // Sequence

  partials += p_offset * Dv;
  sums += p_offset;
  maxs += p_offset;

  O += batch_idx * params.O_strides[0] + // Batch
      head_idx * params.O_strides[1] + // Head
      q_seq_idx * params.O_strides[2]; // Sequence

  // Each warp merges the partial outputs of blocks warp, warp + 32, ... and
  // each lane the sums of blocks lane, lane + 32, ..., both rescaled to a
  // running max. The two loops are independent so their loads overlap.
  U o_max = Limits<U>::finite_min();
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0.f;
  }
  for (int b = warp_idx; b < blocks; b += BN) {
    U m = maxs[b];
    U new_max = max(o_max, m);
    U factor = exp2f(o_max - new_max);
    U p_factor = exp2f(m - new_max);
    o_max = new_max;
    const float* p = partials + b * Dv + v_per_thread * lane_idx;
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = o[i] * factor + p[i] * p_factor;
    }
  }

  U max_score = Limits<U>::finite_min();
  U sum_exp_score = 0.f;
  for (int b = lane_idx; b < blocks; b += BD) {
    U m = maxs[b];
    U new_max = max(max_score, m);
    sum_exp_score = sum_exp_score * exp2f(max_score - new_max) +
        sums[b] * exp2f(m - new_max);
    max_score = new_max;
  }
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  sum_exp_score = cg::reduce(
      warp, sum_exp_score * exp2f(max_score - new_max), cg::plus<U>());
  sum_exp_score = sum_exp_score == 0 ? 0 : __frcp_rn(sum_exp_score);

  // Bring every warp's outputs to the global max.
  U factor = exp2f(o_max - new_max);
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] *= factor;
  }

  // Now we need to aggregate all the outputs
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][warp_idx] = o[i];
    block.sync();
    U ot = outputs[warp_idx][lane_idx];
    o[i] = cg::reduce(warp, ot, cg::plus<U>()) * sum_exp_score;
    block.sync();
  }

  // And write the output
  if (lane_idx == 0) {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      O[v_per_thread * warp_idx + i] = static_cast<T>(o[i]);
    }
  }
}

} // namespace cu

namespace {

// (D_qk, D_v) pairs compiled for the vector kernels. Keep in sync with the
// cases in dispatch_headdims().
constexpr std::pair<int, int> vector_head_dims[] = {
    {64, 64},
    {96, 96},
    {128, 128},
    {256, 256},
    {512, 512},
    {96, 64},
    {192, 128},
};

constexpr bool is_supported_vector_head_dims(int dk, int dv) {
  for (auto [k, v] : vector_head_dims) {
    if (k == dk && v == dv) {
      return true;
    }
  }
  return false;
}

template <typename F>
void dispatch_headdims(int dk, int dv, F&& f) {
#define MLX_SDPAV_CASE(DK, DV)                                               \
  if (dk == DK && dv == DV) {                                                \
    f(std::integral_constant<int, DK>{}, std::integral_constant<int, DV>{}); \
    return;                                                                  \
  }
  MLX_SDPAV_CASE(64, 64)
  MLX_SDPAV_CASE(96, 96)
  MLX_SDPAV_CASE(128, 128)
  MLX_SDPAV_CASE(256, 256)
  MLX_SDPAV_CASE(512, 512)
  MLX_SDPAV_CASE(96, 64)
  MLX_SDPAV_CASE(192, 128)
#undef MLX_SDPAV_CASE
  throw std::invalid_argument(
      fmt::format("[sdpa_vector] Unsupported head dims ({}, {}).", dk, dv));
}

void sdpa_vector_1pass_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks) {
  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  if (sinks) {
    encoder.set_input_array(*sinks);
  }
  encoder.set_output_array(o);

  cu::AttnParams params{
      /* int B = */ q.shape(0),
      /* int H = */ q.shape(1),
      /* int D = */ q.shape(3),

      /* int qL = */ q.shape(2),
      /* int kL = */ k.shape(2),

      /* int gqa_factor = */ q.shape(1) / k.shape(1),
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  dim3 grid_dim(params.H, params.qL, params.B);
  dim3 block_dim(1024, 1, 1);

  dispatch_float_types(o.dtype(), "kernel_sdpav_1pass", [&](auto type_tag) {
    dispatch_bool(do_causal, [&](auto do_causal) {
      dispatch_headdims(q.shape(3), v.shape(3), [&](auto dk, auto dv) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        auto kernel = cu::
            kernel_sdpav_1pass<DataType, do_causal.value, dk.value, dv.value>;
        encoder.add_kernel_node(
            kernel,
            grid_dim,
            block_dim,
            gpu_ptr<DataType>(q),
            gpu_ptr<DataType>(k),
            gpu_ptr<DataType>(v),
            gpu_ptr<DataType>(o),
            sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
            params);
      });
    });
  });
}

// Query heads each warp of kernel_sdpav_2pass_1_gqa holds: as many as keep
// the per-lane query and output accumulators within 64 floats, up to 8.
constexpr int sdpa_vector_gqa_max_heads(int dk, int dv) {
  int budget = 64 / ((dk + dv) / 32);
  int h = 1;
  while (2 * h <= budget && 2 * h <= 8) {
    h *= 2;
  }
  return h;
}

// Heads per warp for the GQA first pass, or 0 to use the per-head kernel:
// single-query decode only, with a power-of-two group that divides the GQA
// factor, of at least 4 heads. Pairs of heads stay on the per-head kernel:
// whether sharing a row between two heads pays off depends on the GPU (it
// helped on an A10G at D = 512 and hurt at D = 256, and the reverse on an
// H100), while groups of 4 or more won on both. MLX_CUDA_SDPA_GQA_PACK=0
// disables packing; a larger value raises the smallest packed group.
int sdpa_vector_gqa_heads(int qL, int gqa_factor, int dk, int dv) {
  static int min_heads = env::get_var("MLX_CUDA_SDPA_GQA_PACK", 4);
  if (min_heads <= 0 || qL != 1) {
    return 0;
  }
  int h = sdpa_vector_gqa_max_heads(dk, dv);
  while (h >= 2 && gqa_factor % h != 0) {
    h /= 2;
  }
  return h >= std::max(min_heads, 4) ? h : 0;
}

// Number of key blocks for the 2-pass kernels. 32 blocks per head fill the
// GPU unless there are only a few heads, so for long contexts keep doubling
// while the first pass has fewer than two thread blocks per SM. Wider heads
// count as several heads since each of their blocks does more work.
int sdpa_vector_2pass_blocks(int num_sms, int heads, int kL, int Dv) {
  constexpr int keys_per_iter = 8; // BN in kernel_sdpav_2pass_1
  constexpr int min_keys_per_warp = 64;
  const int work = heads * std::max(Dv, 256) / 256;
  int blocks = 32;
  while (work * blocks < 2 * num_sms &&
         2 * blocks * keys_per_iter * min_keys_per_warp <= kL) {
    blocks *= 2;
  }
  return blocks;
}

void sdpa_vector_2pass_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks) {
  cu::AttnParams params{
      /* int B = */ q.shape(0),
      /* int H = */ q.shape(1),
      /* int D = */ q.shape(3),

      /* int qL = */ q.shape(2),
      /* int kL = */ k.shape(2),

      /* int gqa_factor = */ q.shape(1) / k.shape(1),
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  const int gqa_heads = sdpa_vector_gqa_heads(
      params.qL, params.gqa_factor, q.shape(3), v.shape(3));
  // A thread block of the GQA first pass covers gqa_heads query heads.
  int blocks = sdpa_vector_2pass_blocks(
      encoder.device().multi_processor_count(),
      params.B * params.H * params.qL / std::max(gqa_heads, 1),
      params.kL,
      v.shape(3));

  // Allocate the intermediates
  Shape intermediate_shape;
  intermediate_shape.reserve(o.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), o.shape().begin(), o.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(o.shape().back());

  array intermediate(intermediate_shape, float32, nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});

  intermediate.set_data(cu::malloc_async(intermediate.nbytes(), encoder));
  sums.set_data(cu::malloc_async(sums.nbytes(), encoder));
  maxs.set_data(cu::malloc_async(maxs.nbytes(), encoder));

  encoder.add_temporary(intermediate);
  encoder.add_temporary(sums);
  encoder.add_temporary(maxs);

  dispatch_float_types(o.dtype(), "kernel_sdpav_2pass", [&](auto type_tag) {
    dispatch_bool(do_causal, [&](auto do_causal) {
      dispatch_headdims(q.shape(3), v.shape(3), [&](auto dk, auto dv) {
        dispatch_bool(blocks == 32, [&](auto fixed_blocks) {
          using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

          {
            encoder.set_input_array(q);
            encoder.set_input_array(k);
            encoder.set_input_array(v);
            if (sinks) {
              encoder.set_input_array(*sinks);
            }

            encoder.set_output_array(intermediate);
            encoder.set_output_array(sums);
            encoder.set_output_array(maxs);

            dim3 block_dim(8 * 32, 1, 1);
            auto launch = [&](auto kernel, dim3 grid_dim) {
              encoder.add_kernel_node(
                  kernel,
                  grid_dim,
                  block_dim,
                  gpu_ptr<DataType>(q),
                  gpu_ptr<DataType>(k),
                  gpu_ptr<DataType>(v),
                  sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
                  gpu_ptr<float>(intermediate),
                  gpu_ptr<float>(sums),
                  gpu_ptr<float>(maxs),
                  params);
            };

            constexpr int max_heads =
                sdpa_vector_gqa_max_heads(dk.value, dv.value);
            dim3 gqa_grid(
                params.H / std::max(gqa_heads, 1), 1, params.B * blocks);
            auto launch_gqa = [&](auto hpt) {
              if constexpr (hpt.value <= max_heads) {
                launch(
                    cu::kernel_sdpav_2pass_1_gqa<
                        DataType,
                        dk.value,
                        dv.value,
                        hpt.value,
                        fixed_blocks.value>,
                    gqa_grid);
              } else {
                throw std::runtime_error(
                    "[sdpa_vector] Too many GQA heads per warp.");
              }
            };
            switch (gqa_heads) {
              case 4:
                launch_gqa(std::integral_constant<int, 4>{});
                break;
              case 8:
                launch_gqa(std::integral_constant<int, 8>{});
                break;
              default:
                launch(
                    cu::kernel_sdpav_2pass_1<
                        DataType,
                        do_causal.value,
                        dk.value,
                        dv.value,
                        fixed_blocks.value>,
                    dim3(params.H, params.qL, params.B * blocks));
            }
          }

          {
            auto kernel = cu::kernel_sdpav_2pass_2<
                DataType,
                do_causal.value,
                dv.value,
                fixed_blocks.value>;

            encoder.set_input_array(intermediate);
            encoder.set_input_array(sums);
            encoder.set_input_array(maxs);
            encoder.set_output_array(o);

            dim3 grid_dim(params.H, params.qL, params.B);
            dim3 block_dim(1024, 1, 1);

            encoder.add_kernel_node(
                kernel,
                grid_dim,
                block_dim,
                gpu_ptr<float>(intermediate),
                gpu_ptr<float>(sums),
                gpu_ptr<float>(maxs),
                gpu_ptr<DataType>(o),
                params,
                blocks);
          }
        });
      });
    });
  });
}

void sdpa_vector_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks) {
  int kL = k.shape(2);

  if (kL > 1024) {
    return sdpa_vector_2pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal, sinks);
  } else {
    return sdpa_vector_1pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal, sinks);
  }
}

} // namespace

bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool output_logsumexp) {
  if (output_logsumexp) {
    return false;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);

  const bool sdpa_supported_head_dim = query_head_dim == k.shape(-1) &&
      is_supported_vector_head_dims(query_head_dim, value_head_dim);

  const bool supported_vector_config =
      sdpa_supported_head_dim && query_sequence_length < 4;

  return supported_vector_config && !has_arr_mask;
}

void sdpa_vector(
    const array& q_pre,
    const array& k_pre,
    const array& v_pre,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks_pre,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(4);
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (sinks_pre) {
    sinks = copy_unless(is_matrix_contiguous, sinks_pre.value());
  }

  // We are in vector mode ie single query
  if (q_pre.shape(2) < 4) {
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    const auto& q = copy_unless(q_copy_unless, q_pre);
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      int64_t str_oD = 1;
      int64_t str_oH = o.shape(3);
      int64_t str_oL = o.shape(1) * str_oH;
      int64_t str_oB = o.shape(2) * str_oL;

      array::Flags flags{
          /* bool contiguous = */ 1,
          /* bool row_contiguous = */ o.shape(2) == 1,
          /* bool col_contiguous = */ o.size() == o.shape(3),
      };

      o.set_data(
          cu::malloc_async(o.nbytes(), encoder),
          o.size(),
          {str_oB, str_oH, str_oL, str_oD},
          flags);
    }

    for (const auto& cp : copies) {
      encoder.add_temporary(cp);
    }

    sdpa_vector_fallback(s, encoder, q, k, v, scale, o, do_causal, sinks);
  }

  // Full attention mode should never reach here
  else {
    throw std::runtime_error("Doesn't support matrix yet.");
  }
}

} // namespace mlx::core
