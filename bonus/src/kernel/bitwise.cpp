#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    auto* __restrict__ dst = reinterpret_cast<std::uint8_t*>(result.data());
    const auto* __restrict__ sa = reinterpret_cast<const std::uint8_t*>(a.data());
    const auto* __restrict__ sb = reinterpret_cast<const std::uint8_t*>(b.data());

    using vec16 = std::uint8_t __attribute__((vector_size(16)));
    const vec16 mask_a5 = {0xA5u, 0xA5u, 0xA5u, 0xA5u,
                           0xA5u, 0xA5u, 0xA5u, 0xA5u,
                           0xA5u, 0xA5u, 0xA5u, 0xA5u,
                           0xA5u, 0xA5u, 0xA5u, 0xA5u};
    const vec16 mask_99 = {0x99u, 0x99u, 0x99u, 0x99u,
                           0x99u, 0x99u, 0x99u, 0x99u,
                           0x99u, 0x99u, 0x99u, 0x99u,
                           0x99u, 0x99u, 0x99u, 0x99u};

    std::size_t i = 0;
    for (; i + 63 < n; i += 64) {
        vec16 a0, a1, a2, a3;
        vec16 b0, b1, b2, b3;
        std::memcpy(&a0, sa + i, 16);
        std::memcpy(&a1, sa + i + 16, 16);
        std::memcpy(&a2, sa + i + 32, 16);
        std::memcpy(&a3, sa + i + 48, 16);
        std::memcpy(&b0, sb + i, 16);
        std::memcpy(&b1, sb + i + 16, 16);
        std::memcpy(&b2, sb + i + 32, 16);
        std::memcpy(&b3, sb + i + 48, 16);
        a0 = mask_a5 ^ ((a0 | b0) & mask_99);
        a1 = mask_a5 ^ ((a1 | b1) & mask_99);
        a2 = mask_a5 ^ ((a2 | b2) & mask_99);
        a3 = mask_a5 ^ ((a3 | b3) & mask_99);
        std::memcpy(dst + i, &a0, 16);
        std::memcpy(dst + i + 16, &a1, 16);
        std::memcpy(dst + i + 32, &a2, 16);
        std::memcpy(dst + i + 48, &a3, 16);
    }
    for (; i + 15 < n; i += 16) {
        vec16 av, bv;
        std::memcpy(&av, sa + i, 16);
        std::memcpy(&bv, sb + i, 16);
        av = mask_a5 ^ ((av | bv) & mask_99);
        std::memcpy(dst + i, &av, 16);
    }
    for (; i < n; ++i) {
        dst[i] = static_cast<std::uint8_t>(0xA5u ^ ((sa[i] | sb[i]) & 0x99u));
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}
