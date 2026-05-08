#include "blackscholes.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>

#define inv_sqrt_2xPI 0.39894228040143270286
#define p_val 0.2316419
#define coefficient_a1 0.319381530
#define coefficient_a2 -0.356563782
#define coefficient_a3 1.781477937
#define coefficient_a4 -1.821255978
#define coefficient_a5 1.330274429

void initialize_blackscholes(blackscholes_args &args,
                             std::size_t n,
                             std::uint32_t seed) {
    args.call_option_price.assign(n, 0.0f);
    args.put_option_price.assign(n, 0.0f);
    args.epsilon = 5e-3;

    args.spot_price.resize(n);
    args.strike.resize(n);
    args.rate.resize(n);
    args.volatility.resize(n);
    args.time.resize(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> spot_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> strike_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> rate_dist(0.0275f, 0.1f);
    std::uniform_real_distribution<float> vol_dist(0.05f, 0.6f);
    std::uniform_real_distribution<float> time_dist(0.1f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        args.spot_price[i] = spot_dist(rng);
        args.strike[i] = strike_dist(rng);
        args.rate[i] = rate_dist(rng);
        args.volatility[i] = vol_dist(rng);
        args.time[i] = time_dist(rng);
    }
}

void CNDF(float &InputX, float &OutputX) {
    int sign = 0;
    float x = InputX;

    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }

    const float xNPrimeofX = std::exp(-0.5f * x * x) * inv_sqrt_2xPI;
    const float k = 1.0f / (1.0f + p_val * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;

    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;

    OutputX = sign ? (1.0f - local) : local;
}

static inline void naive_BlkSchls_one(float &CallOptionPrice,
                                      float &PutOptionPrice, float spotPrice,
                                      float strike, float rate,
                                      float volatility, float time) {
    const float xSqrtTime = std::sqrt(time);
    const float xLogTerm = std::log(spotPrice / strike);
    const float xPowerTerm = 0.5f * volatility * volatility;

    float xD1 = (rate + xPowerTerm) * time + xLogTerm;
    const float xDen = volatility * xSqrtTime;
    xD1 = xD1 / xDen;
    const float xD2 = xD1 - xDen;

    float d1 = xD1;
    float d2 = xD2;
    float NofXd1 = 0.0f;
    float NofXd2 = 0.0f;

    CNDF(d1, NofXd1);
    CNDF(d2, NofXd2);

    const float FutureValueX = strike * std::exp(-(rate) * (time));
    CallOptionPrice = (spotPrice * NofXd1) - (FutureValueX * NofXd2);

    const float NegNofXd1 = 1.0f - NofXd1;
    const float NegNofXd2 = 1.0f - NofXd2;
    PutOptionPrice = (FutureValueX * NegNofXd2) - (spotPrice * NegNofXd1);
}

void naive_BlkSchls(std::vector<float> &CallOptionPrice,
                    std::vector<float> &PutOptionPrice,
                    const std::vector<float> &spotPrice,
                    const std::vector<float> &strike,
                    const std::vector<float> &rate,
                    const std::vector<float> &volatility,
                    const std::vector<float> &time) {
    size_t n = spotPrice.size();
    for (size_t i = 0; i < n; ++i) {
        naive_BlkSchls_one(CallOptionPrice[i],
                           PutOptionPrice[i],
                           spotPrice[i],
                           strike[i],
                           rate[i],
                           volatility[i],
                           time[i]);
    }
}


static constexpr float kLn2   = 0.69314718056f;
static constexpr float kInvLn2 = 1.44269504089f;

__attribute__((always_inline))
static inline float fast_exp_neg(float x) {
    if (x < -87.0f) return 0.0f;
    const int n = static_cast<int>(x * kInvLn2 - 0.5f);
    const float r = x - static_cast<float>(n) * kLn2;
    const float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f +
                    r * (0.041666667f + r * (0.0083333333f + r * 0.0013888889f)))));
    if (n < -126) return 0.0f;
    union { std::uint32_t u; float f; } v;
    v.u = static_cast<std::uint32_t>(n + 127) << 23;
    return p * v.f;
}


static const float sLogC0 = -19.645704f;
static const float sLogC1 = 0.767002f;
static const float sLogC2 = 0.3717479f;
static const float sLogC3 = 5.2653985f;
static const float sLogC4 = -(1.0f + sLogC0) * (1.0f + sLogC1) /
                              ((1.0f + sLogC2) * (1.0f + sLogC3));
static const float sLog2  = 0.6931472f;

__attribute__((always_inline))
static inline float fast_log(float x) {
    unsigned int ux = std::bit_cast<unsigned int>(x);
    int e = static_cast<int>(ux - 0x3f800000) >> 23;
    ux |= 0x3f800000;
    ux &= 0x3fffffff;
    x = std::bit_cast<float>(ux);
    float a = (x + sLogC0) * (x + sLogC1);
    float b = (x + sLogC2) * (x + sLogC3);
    return (static_cast<float>(e) + sLogC4 + a / b) * sLog2;
}


__attribute__((always_inline))
static inline float fast_cndf(float x) {
    const bool neg = (x < 0.0f);
    x = std::abs(x);
    const float k = 1.0f / (1.0f + static_cast<float>(p_val) * x);
    const float pdf = fast_exp_neg(-0.5f * x * x) * static_cast<float>(inv_sqrt_2xPI);
    const float poly = k * (static_cast<float>(coefficient_a1) +
                       k * (static_cast<float>(coefficient_a2) +
                       k * (static_cast<float>(coefficient_a3) +
                       k * (static_cast<float>(coefficient_a4) +
                       k *  static_cast<float>(coefficient_a5)))));
    const float cdf = 1.0f - poly * pdf;
    return neg ? (1.0f - cdf) : cdf;
}

__attribute__((always_inline))
static inline void stu_BlkSchls_one(float &call, float &put,
                                    float spot, float strike,
                                    float rate, float vol, float t) {
    const float sqrtT  = std::sqrt(t);
    const float logVal = fast_log(spot / strike);
    const float halfV2 = 0.5f * vol * vol;
    const float den    = vol * sqrtT;
    const float invDen = 1.0f / den;
    const float d1     = ((rate + halfV2) * t + logVal) * invDen;
    const float d2     = d1 - den;
    const float Nd1    = fast_cndf(d1);
    const float Nd2    = fast_cndf(d2);
    const float disc   = strike * fast_exp_neg(-rate * t);
    call = spot * Nd1 - disc * Nd2;
    put  = disc * (1.0f - Nd2) - spot * (1.0f - Nd1);
}

void stu_BlkSchls(std::vector<float> &CallOptionPrice,
                  std::vector<float> &PutOptionPrice,
                  const std::vector<float> &spotPrice,
                  const std::vector<float> &strike,
                  const std::vector<float> &rate,
                  const std::vector<float> &volatility,
                  const std::vector<float> &time) {
    const size_t n = spotPrice.size();
    float* __restrict__ call = CallOptionPrice.data();
    float* __restrict__ put  = PutOptionPrice.data();
    const float* __restrict__ s = spotPrice.data();
    const float* __restrict__ k = strike.data();
    const float* __restrict__ r = rate.data();
    const float* __restrict__ v = volatility.data();
    const float* __restrict__ t = time.data();

    for (size_t i = 0; i < n; ++i) {
        stu_BlkSchls_one(call[i], put[i], s[i], k[i], r[i], v[i], t[i]);
    }
}

void naive_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    naive_BlkSchls(args.call_option_price,
                   args.put_option_price,
                   args.spot_price,
                   args.strike,
                   args.rate,
                   args.volatility,
                   args.time);
}

void stu_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    stu_BlkSchls(args.call_option_price,
                 args.put_option_price,
                 args.spot_price,
                 args.strike,
                 args.rate,
                 args.volatility,
                 args.time);
}

bool BlkSchls_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu_args = *static_cast<blackscholes_args *>(stu_ctx);
    auto &ref_args = *static_cast<blackscholes_args *>(ref_ctx);
    const double eps = ref_args.epsilon; // relative tolerance

    if (ref_args.call_option_price.size() != stu_args.call_option_price.size() ||
        ref_args.put_option_price.size() != stu_args.put_option_price.size())
        return false;

    const double atol = 1e-5; // absolute tolerance for near-zero prices
    const size_t n = ref_args.call_option_price.size();
    double max_rel = 0.0, max_abs = 0.0;
    size_t max_idx = 0;
    const char *max_leg = "call";

    for (size_t i = 0; i < n; ++i) {
        const double rc = static_cast<double>(ref_args.call_option_price[i]);
        const double rp = static_cast<double>(ref_args.put_option_price[i]);
        const double sc = static_cast<double>(stu_args.call_option_price[i]);
        const double sp = static_cast<double>(stu_args.put_option_price[i]);

        const double err_c = std::abs(rc - sc);
        const double err_p = std::abs(rp - sp);
        const double rel_c = (err_c - atol) / std::abs(rc);
        const double rel_p = (err_p - atol) / std::abs(rp);

        const bool call_ok = err_c <= (atol + eps * std::abs(rc));
        const bool put_ok = err_p <= (atol + eps * std::abs(rp));

        if (rel_c > max_rel) {
            max_abs = err_c;
            max_rel = rel_c;
            max_idx = i;
            max_leg = "call";
        }
        if (rel_p > max_rel) {
            max_abs = err_p;
            max_rel = rel_p;
            max_idx = i;
            max_leg = "put";
        }

        if (!call_ok || !put_ok) {
            debug_log("\tDEBUG: fail idx={} | call ref={} stu={} err={} thr={} | put ref={} stu={} err={} thr={}\n",
                      i,
                      rc,
                      sc,
                      err_c,
                      (atol + eps * std::abs(rc)),
                      rp,
                      sp,
                      err_p,
                      (atol + eps * std::abs(rp)));
            return false;
        }
    }
    debug_log("\tBlkSchls_check passed: n={}, max_rel_err={}, max_abs_err={} at idx={} ({})\n",
              n,
              max_rel,
              max_abs,
              max_idx,
              max_leg);

    return true;
}
