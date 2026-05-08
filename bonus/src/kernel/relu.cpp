#include "relu.h"
#include <algorithm>
#include <cstdint>
#include <random>
#include <xmmintrin.h>

void initialize_relu(relu_args *args, const size_t size,
                     const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr float mean = 0.0f;
    constexpr float stddev = 1.0f;

    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);

    args->data.resize(size);

    for (auto &value : args->data) {
        value = dist(gen);
    }
}

void naive_relu(std::span<float> data) {
    for (auto &&value : data) {
        if (value < 0.0f) {
            value = 0.0f;
        }
    }
}

void stu_relu(std::span<float> data) {
    float* __restrict__ ptr = data.data();
    const size_t n = data.size();
    const __m128 zero = _mm_setzero_ps();

    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m128 v0 = _mm_loadu_ps(ptr + i);
        __m128 v1 = _mm_loadu_ps(ptr + i + 4);
        __m128 v2 = _mm_loadu_ps(ptr + i + 8);
        __m128 v3 = _mm_loadu_ps(ptr + i + 12);
        const int m0 = _mm_movemask_ps(v0);
        const int m1 = _mm_movemask_ps(v1);
        const int m2 = _mm_movemask_ps(v2);
        const int m3 = _mm_movemask_ps(v3);
        if (m0) _mm_storeu_ps(ptr + i, _mm_max_ps(v0, zero));
        if (m1) _mm_storeu_ps(ptr + i + 4, _mm_max_ps(v1, zero));
        if (m2) _mm_storeu_ps(ptr + i + 8, _mm_max_ps(v2, zero));
        if (m3) _mm_storeu_ps(ptr + i + 12, _mm_max_ps(v3, zero));
    }
    for (; i + 3 < n; i += 4) {
        __m128 v = _mm_loadu_ps(ptr + i);
        if (_mm_movemask_ps(v)) {
            _mm_storeu_ps(ptr + i, _mm_max_ps(v, zero));
        }
    }
    for (; i < n; ++i) {
        if (ptr[i] < 0.0f) {
            ptr[i] = 0.0f;
        }
    }
}

void naive_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    naive_relu(args.data);
}

void stu_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    stu_relu(args.data);
}

bool relu_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<relu_args *>(stu_ctx);
    auto &ref_args = *static_cast<relu_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;

    if (stu_args.data.size() != ref_args.data.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.data.size(),
                  ref_args.data.size());
        return false;
    }

    double max_rel = 0.0;
    size_t worst_i = 0;
    const double atol = 1e-6;

    for (size_t i = 0; i < ref_args.data.size(); ++i) {
        const double r = static_cast<double>(ref_args.data[i]);
        const double s = static_cast<double>(stu_args.data[i]);
        const double err = std::abs(s - r);
        const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;

        if (rel > max_rel) {
            max_rel = rel;
            worst_i = i;
        }

        if (err > (atol + eps * std::abs(r))) {
            debug_log(
                "\tDEBUG: fail at {}: ref={} stu={} err={} rel={} thr={}\n",
                i,
                ref_args.data[i],
                stu_args.data[i],
                err,
                rel,
                (atol + eps * std::abs(r)));
            return false;
        }
    }

    debug_log("\tDEBUG: relu_check passed. max_rel={} at i={}\n",
              max_rel,
              worst_i);
    return true;
}
