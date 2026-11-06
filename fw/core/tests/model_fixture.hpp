// Builds random float MLPs and a reference int8 quantization of them, laid out
// exactly like the codegen will emit (plain arrays + nf::FloatModel/Int8Model).
// The quantizer here mirrors INTERFACES.md section 7 so the C++ int8 kernel can be
// checked against the float model; the Python golden model is the real reference.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "nf/policy.hpp"

namespace nftest {

struct Lcg {
  uint32_t s;
  explicit Lcg(uint32_t seed) : s(seed) {}
  float uni() { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 16777216.0f; }
  float sym(float a) { return (2.0f * uni() - 1.0f) * a; }
};

struct MlpFixture {
  static constexpr int kMaxL = 4;
  int n_layers = 0;
  int dims[kMaxL + 1] = {};
  std::vector<float> mean, stdv;
  std::vector<float> W[kMaxL], b[kMaxL];
  nf::FloatLayer fl[kMaxL];
  nf::FloatModel fm{};
  // int8
  std::vector<int8_t> Wq[kMaxL];
  std::vector<int32_t> bq[kMaxL], M[kMaxL];
  std::vector<uint8_t> S[kMaxL];
  std::vector<int8_t> lut[kMaxL];
  std::vector<float> sw[kMaxL];
  nf::Int8Layer ql[kMaxL];
  nf::Int8Model qm{};
  float s_in = 4.0f / 127.0f;
  float s_pre = 3.0f / 127.0f;

  // hidden: e.g. {64, 64}
  void build(const int* hidden, int n_hidden, uint32_t seed) {
    Lcg r(seed);
    n_layers = n_hidden + 1;
    dims[0] = nf::kObsDim;
    for (int i = 0; i < n_hidden; ++i) dims[i + 1] = hidden[i];
    dims[n_layers] = nf::kActDim;
    mean.resize(nf::kObsDim);
    stdv.resize(nf::kObsDim);
    for (int i = 0; i < nf::kObsDim; ++i) { mean[i] = r.sym(0.2f); stdv[i] = 0.5f + r.uni(); }
    for (int l = 0; l < n_layers; ++l) {
      int in = dims[l], out = dims[l + 1];
      float a = 1.5f / std::sqrt(static_cast<float>(in));
      W[l].resize(static_cast<size_t>(in * out));
      b[l].resize(static_cast<size_t>(out));
      for (float& w : W[l]) w = r.sym(a);
      for (float& v : b[l]) v = r.sym(0.1f);
      fl[l] = {static_cast<uint16_t>(in), static_cast<uint16_t>(out), W[l].data(), b[l].data(), nf::Act::Tanh};
    }
    fm = {nf::kObsDim, nf::kActDim, static_cast<uint16_t>(n_layers), mean.data(), stdv.data(), fl, 0};
    fm.crc32 = nf::policy_crc32(fm);
    quantize();
  }

  static int32_t rnd(double v) { return static_cast<int32_t>(v >= 0 ? std::floor(v + 0.5) : -std::floor(-v + 0.5)); }

  void quantize() {
    float s_x = s_in;
    for (int l = 0; l < n_layers; ++l) {
      int in = dims[l], out = dims[l + 1];
      bool last = l + 1 == n_layers;
      Wq[l].resize(static_cast<size_t>(in * out));
      bq[l].resize(static_cast<size_t>(out));
      sw[l].resize(static_cast<size_t>(out));
      M[l].resize(static_cast<size_t>(out));
      S[l].resize(static_cast<size_t>(out));
      for (int j = 0; j < out; ++j) {
        float mx = 1e-8f;
        for (int i = 0; i < in; ++i) mx = std::fmax(mx, std::fabs(W[l][static_cast<size_t>(j * in + i)]));
        sw[l][static_cast<size_t>(j)] = mx / 127.0f;
        for (int i = 0; i < in; ++i)
          Wq[l][static_cast<size_t>(j * in + i)] =
              static_cast<int8_t>(rnd(static_cast<double>(W[l][static_cast<size_t>(j * in + i)] / sw[l][static_cast<size_t>(j)])));
        const double swj = static_cast<double>(sw[l][static_cast<size_t>(j)]);
        bq[l][static_cast<size_t>(j)] =
            rnd(static_cast<double>(b[l][static_cast<size_t>(j)]) / (static_cast<double>(s_x) * swj));
        double m = static_cast<double>(s_x) * swj / static_cast<double>(s_pre);
        int sh = static_cast<int>(std::floor(30.0 - std::log2(m)));
        sh = sh < 1 ? 1 : (sh > 62 ? 62 : sh);
        S[l][static_cast<size_t>(j)] = static_cast<uint8_t>(sh);
        M[l][static_cast<size_t>(j)] = rnd(m * std::ldexp(1.0, sh));
      }
      lut[l].resize(256);
      for (int k = 0; k < 256; ++k) lut[l][static_cast<size_t>(k)] = static_cast<int8_t>(rnd(127.0 * std::tanh((k - 128) * static_cast<double>(s_pre))));
      ql[l] = {static_cast<uint16_t>(in), static_cast<uint16_t>(out), Wq[l].data(), bq[l].data(),
               last ? nullptr : M[l].data(), last ? nullptr : S[l].data(), last ? nullptr : lut[l].data(),
               last ? sw[l].data() : nullptr};
      if (!last) s_x = 1.0f / 127.0f;
    }
    float s_x_last = n_layers > 1 ? 1.0f / 127.0f : s_in;
    qm = {nf::kObsDim, nf::kActDim, static_cast<uint16_t>(n_layers), mean.data(), stdv.data(), ql, s_in, s_x_last, 0};
    qm.crc32 = nf::policy_crc32(qm);
  }

  void random_obs(Lcg& r, float* obs) const {
    for (int i = 0; i < nf::kObsDim; ++i) obs[i] = mean[static_cast<size_t>(i)] + stdv[static_cast<size_t>(i)] * r.sym(2.0f);
  }
};

}  // namespace nftest
