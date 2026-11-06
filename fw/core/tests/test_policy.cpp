#include "nf/policy.hpp"
#include "model_fixture.hpp"
#include "nftest.hpp"

using namespace nf;

namespace {
// Independent reference: floor division by 2^s using only well-defined
// (truncating) integer division.
int64_t ref_rshift_round(int64_t v, int s) {
  int64_t d = static_cast<int64_t>(1) << s;
  int64_t t = v + (d >> 1);
  int64_t q = t / d;
  if ((t % d != 0) && (t < 0)) --q;
  return q;
}
}  // namespace

TEST(rshift_round_edge_cases) {
  CHECK_EQ(rshift_round(0, 1), 0);
  CHECK_EQ(rshift_round(1, 1), 1);    // 0.5 -> 1 (half rounds up)
  CHECK_EQ(rshift_round(-1, 1), 0);   // -0.5 -> 0 (half rounds toward +inf)
  CHECK_EQ(rshift_round(-2, 1), -1);
  CHECK_EQ(rshift_round(-3, 1), -1);  // -1.5 -> -1
  CHECK_EQ(rshift_round(-4, 1), -2);
  CHECK_EQ(rshift_round(5, 1), 3);    // 2.5 -> 3
  CHECK_EQ(rshift_round(-5, 1), -2);  // -2.5 -> -2
  CHECK_EQ(rshift_round(-6, 2), -1);  // -1.5 -> -1
  CHECK_EQ(rshift_round(-7, 2), -2);  // -1.75 -> -2
  CHECK_EQ(rshift_round(-9, 3), -1);  // -1.125 -> -1
  CHECK_EQ(rshift_round(-12, 3), -1); // -1.5 -> -1
  CHECK_EQ(rshift_round(-13, 3), -2);
  const int64_t big = static_cast<int64_t>(1) << 62;
  CHECK_EQ(rshift_round(-big, 62), -1);
  CHECK_EQ(rshift_round(-big / 2, 62), 0);   // exactly -0.5 -> 0
  CHECK_EQ(rshift_round(-big / 2 - 1, 62), -1);
  CHECK_EQ(rshift_round(big, 62), 1);
  // int32 * int32 extremes (the products the kernel actually forms).
  const int64_t p = static_cast<int64_t>(INT32_MIN) * INT32_MAX;
  CHECK_EQ(rshift_round(p, 62), ref_rshift_round(p, 62));
  CHECK_EQ(rshift_round(p, 31), ref_rshift_round(p, 31));
  CHECK_EQ(rshift_round(-p, 40), ref_rshift_round(-p, 40));
  // Randomised against the reference for every shift.
  nftest::Lcg r(7);
  int mism = 0;
  for (int s = 1; s <= 62; ++s)
    for (int k = 0; k < 2000; ++k) {
      int32_t a = static_cast<int32_t>((static_cast<uint32_t>(r.uni() * 4294967295.0f)));
      int32_t m = static_cast<int32_t>((static_cast<uint32_t>(r.uni() * 4294967295.0f)));
      int64_t v = static_cast<int64_t>(a) * m;
      if (rshift_round(v, s) != ref_rshift_round(v, s)) ++mism;
      if (rshift_round(k - 1000, s) != ref_rshift_round(k - 1000, s)) ++mism;
    }
  CHECK_EQ(mism, 0);
}

TEST(quantize_input_rounding) {
  CHECK_EQ(quantize_input(0.5f), 1);
  CHECK_EQ(quantize_input(-0.5f), -1);   // half away from zero (NOT numpy's half-to-even)
  CHECK_EQ(quantize_input(1.5f), 2);
  CHECK_EQ(quantize_input(2.5f), 3);
  CHECK_EQ(quantize_input(-2.5f), -3);
  CHECK_EQ(quantize_input(0.49999997f), 0);
  CHECK_EQ(quantize_input(126.6f), 127);
  CHECK_EQ(quantize_input(500.0f), 127);
  CHECK_EQ(quantize_input(-500.0f), -127);  // never -128 at the input
  CHECK_EQ(quantize_input(INFINITY), 127);
  CHECK_EQ(quantize_input(-INFINITY), -127);
  CHECK_EQ(quantize_input(NAN), 0);
}

TEST(crc32_known_vector) {
  const char* s = "123456789";
  CHECK_EQ(crc32_update(0, s, 9), 0xCBF43926u);
  // Incremental == one-shot.
  uint32_t c = crc32_update(0, s, 4);
  c = crc32_update(c, s + 4, 5);
  CHECK_EQ(c, 0xCBF43926u);
}

TEST(float_mlp_hand_computed) {
  // obs_dim 19 -> 2 hidden (tanh) -> 4 (tanh); only obs[0], obs[1] matter.
  static float mean[19] = {}, stdv[19];
  for (float& s : stdv) s = 1.0f;
  mean[0] = 1.0f; stdv[1] = 2.0f;
  static float W0[2 * 19] = {}, b0[2] = {0.1f, -0.2f};
  W0[0] = 0.5f; W0[19 + 1] = -1.0f;
  static float W1[4 * 2] = {1, 0, 0, 1, 1, 1, -1, 0.5f}, b1[4] = {0, 0, 0.3f, 0};
  static FloatLayer L[2] = {{19, 2, W0, b0, Act::Tanh}, {2, 4, W1, b1, Act::Tanh}};
  FloatModel m{19, 4, 2, mean, stdv, L, 0};
  CHECK(validate(m));
  float obs[19] = {};
  obs[0] = 2.0f; obs[1] = 1.0f;  // xn = [1, 0.5]
  float out[4];
  CHECK(mlp_forward(m, obs, out));
  float h0 = std::tanh(0.5f * 1.0f + 0.1f), h1 = std::tanh(-1.0f * 0.5f - 0.2f);
  CHECK_NEAR(out[0], std::tanh(h0), 1e-6);
  CHECK_NEAR(out[1], std::tanh(h1), 1e-6);
  CHECK_NEAR(out[2], std::tanh(h0 + h1 + 0.3f), 1e-6);
  CHECK_NEAR(out[3], std::tanh(-h0 + 0.5f * h1), 1e-6);
}

TEST(validate_rejects_bad_models) {
  nftest::MlpFixture f;
  int hidden[2] = {64, 64};
  f.build(hidden, 2, 1);
  CHECK(validate(f.fm));
  CHECK(validate(f.qm));
  FloatModel bad = f.fm;
  bad.n_layers = 2;  // chain ends at 64 != 4
  CHECK(!validate(bad));
  Int8Model qb = f.qm;
  qb.s_in = 0.0f;
  CHECK(!validate(qb));
  f.S[0][3] = 63;
  CHECK(!validate(f.qm));
  f.S[0][3] = 0;
  CHECK(!validate(f.qm));
}

TEST(crc_detects_weight_change) {
  nftest::MlpFixture f;
  int hidden[2] = {64, 64};
  f.build(hidden, 2, 2);
  CHECK_EQ(policy_crc32(f.fm), f.fm.crc32);
  CHECK_EQ(policy_crc32(f.qm), f.qm.crc32);
  f.W[1][100] += 1e-3f;
  CHECK(policy_crc32(f.fm) != f.fm.crc32);
  f.lut[0][7] ^= 1;
  CHECK(policy_crc32(f.qm) != f.qm.crc32);
}

TEST(int8_mlp_matches_float_within_quant_tolerance) {
  int configs[3][2] = {{64, 64}, {32, 64}, {64, 128}};
  for (auto& h : configs) {
    nftest::MlpFixture f;
    f.build(h, 2, 42u + static_cast<uint32_t>(h[1]));
    nftest::Lcg r(99);
    float max_err = 0, sum_err = 0;
    const int N = 500;
    for (int n = 0; n < N; ++n) {
      float obs[kObsDim], a[kActDim], q[kActDim];
      f.random_obs(r, obs);
      mlp_forward(f.fm, obs, a);
      mlp_forward_int8(f.qm, obs, q);
      for (int j = 0; j < kActDim; ++j) {
        float e = std::fabs(a[j] - q[j]);
        max_err = std::fmax(max_err, e);
        sum_err += e;
      }
    }
    float mean_err = sum_err / (N * kActDim);
    std::printf("  int8 vs float [%d,%d]: mean |err| %.4f, max %.4f\n", h[0], h[1], static_cast<double>(mean_err),
                static_cast<double>(max_err));
    CHECK(mean_err < 0.02f);
    CHECK(max_err < 0.08f);
  }
}

TEST(int8_mlp_bit_exact_vs_reference_loops) {
  // Re-derive every intermediate with plain int64 arithmetic and the
  // division-based rshift reference; the last-layer accumulators must match.
  nftest::MlpFixture f;
  int hidden[2] = {64, 64};
  f.build(hidden, 2, 5);
  nftest::Lcg r(3);
  int mism = 0;
  for (int n = 0; n < 300; ++n) {
    float obs[kObsDim];
    f.random_obs(r, obs);
    if (n == 0) obs[3] = 1e6f;  // saturating input
    int8_t xq_kernel[kObsDim];
    int32_t acc_kernel[kActDim];
    float out_kernel[kActDim];
    mlp_forward_int8(f.qm, obs, out_kernel, acc_kernel, xq_kernel);
    int8_t x[128];
    for (int i = 0; i < kObsDim; ++i) {
      float xn = (obs[i] - f.mean[static_cast<size_t>(i)]) / f.stdv[static_cast<size_t>(i)];
      x[i] = quantize_input(xn / f.s_in);
      if (x[i] != xq_kernel[i]) ++mism;
    }
    for (int l = 0; l < f.n_layers; ++l) {
      const Int8Layer& L = f.ql[l];
      int8_t y[128];
      for (int j = 0; j < L.out; ++j) {
        int64_t acc = L.b[j];
        for (int i = 0; i < L.in; ++i) acc += static_cast<int64_t>(L.W[j * L.in + i]) * x[i];
        if (l + 1 == f.n_layers) {
          if (static_cast<int32_t>(acc) != acc_kernel[j]) ++mism;
          float o = std::tanh(static_cast<float>(acc) * f.qm.s_x_last * L.s_w[j]);
          if (std::fabs(o - out_kernel[j]) > 1e-6f) ++mism;
        } else {
          int64_t v = ref_rshift_round(acc * L.M[j], L.S[j]);
          v = v < -128 ? -128 : (v > 127 ? 127 : v);
          y[j] = L.lut[v + 128];
        }
      }
      for (int j = 0; j < L.out; ++j) x[j] = y[j];
    }
  }
  CHECK_EQ(mism, 0);
}

TEST(int8_single_layer_uses_s_in) {
  nftest::MlpFixture f;
  f.build(nullptr, 0, 11);
  CHECK(validate(f.qm));
  CHECK_NEAR(f.qm.s_x_last, f.s_in, 0);
  nftest::Lcg r(1);
  float obs[kObsDim], a[kActDim], q[kActDim];
  f.random_obs(r, obs);
  mlp_forward(f.fm, obs, a);
  mlp_forward_int8(f.qm, obs, q);
  for (int j = 0; j < kActDim; ++j) CHECK_NEAR(a[j], q[j], 0.05);
}

TEST(obs_layout_and_ctbr_decode) {
  VehicleState s;
  s.pos = {3.0f, -0.5f, 1.2f};
  s.vel = {0.1f, -7.0f, 0.3f};
  s.q = Quat::from_euler(0.1f, 0.2f, 0.3f);
  s.omega = {1, 2, 3};
  float prev[4] = {0.1f, 0.2f, 0.3f, 0.4f}, obs[kObsDim];
  build_obs(s, {0, 0, 1}, prev, obs);
  CHECK_NEAR(obs[0], 2.0f, 0);   // clipped
  CHECK_NEAR(obs[1], -0.5f, 0);
  CHECK_NEAR(obs[2], 0.2f, 1e-6);
  CHECK_NEAR(obs[4], -5.0f, 0);  // clipped
  Mat3 R = s.q.to_mat();
  CHECK_NEAR(obs[6], R(0, 0), 0); CHECK_NEAR(obs[7], R(1, 0), 0); CHECK_NEAR(obs[8], R(2, 0), 0);
  CHECK_NEAR(obs[9], R(0, 1), 0); CHECK_NEAR(obs[10], R(1, 1), 0); CHECK_NEAR(obs[11], R(2, 1), 0);
  CHECK_NEAR(obs[14], 3.0f, 0);
  CHECK_NEAR(obs[18], 0.4f, 0);
  float a[4] = {0.0f, 1.0f, -0.5f, 2.0f};
  CtbrCommand c = decode_ctbr(a);
  CHECK_NEAR(c.thrust_n, 2.0f * vehicle::kThrustMax, 1e-5);
  CHECK_NEAR(c.rate_sp.x, vehicle::kMaxBodyRate[0], 1e-6);
  CHECK_NEAR(c.rate_sp.y, -0.5f * vehicle::kMaxBodyRate[1], 1e-6);
  CHECK_NEAR(c.rate_sp.z, vehicle::kMaxBodyRate[2], 1e-6);  // clamped
  float b[4] = {-1.0f, NAN, 0, 0};
  c = decode_ctbr(b);
  CHECK_NEAR(c.thrust_n, 0.0f, 0);
  CHECK_NEAR(c.rate_sp.x, 0.0f, 0);
  float e[4];
  CtbrCommand c2{7.0f, {1.0f, -2.0f, 0.5f}};
  encode_ctbr(c2, e);
  CtbrCommand c3 = decode_ctbr(e);
  CHECK_NEAR(c3.thrust_n, 7.0f, 1e-5);
  CHECK_NEAR(c3.rate_sp.y, -2.0f, 1e-5);
}

TEST(policy_runner_prev_action) {
  nftest::MlpFixture f;
  int hidden[2] = {64, 64};
  f.build(hidden, 2, 8);
  PolicyRunner pr;
  pr.set_model(&f.fm);
  CHECK(pr.valid());
  CHECK_EQ(pr.computed_crc(), pr.expected_crc());
  VehicleState s;
  float a[4];
  CHECK(pr.run(s, {0, 0, 1}, a));
  for (int i = 0; i < 4; ++i) CHECK_NEAR(pr.prev_action()[i], a[i], 0);
  float a2[4];
  pr.run(s, {0, 0, 1}, a2);
  for (int i = 0; i < 4; ++i) CHECK_NEAR(pr.last_obs()[15 + i], a[i], 0);
}

NFTEST_MAIN()
