// Bit-exact check of the generated policy against the Python golden model.
//
//   test_policy_bitexact <vectors.bin>
//
// The policy comes from tools/codegen.py (nf_policy_gen.hpp/.cpp, namespace
// nf_gen); the vectors from tools/gen_vectors.py (layout documented there).
// Asserts:
//   * validate() on both models, policy_crc32() == embedded CRC == Python CRC
//   * x_q and last-layer int32 accumulators bit-exact for every row
//   * int8-path float output within 1e-6 of the golden
//   * float model within 1e-5 of the numpy float32 reference (NaN <-> NaN)
//   * nf::rshift_round == Python-int reference for all K cases
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nf/policy.hpp"
#include "nf_policy_gen.hpp"

namespace {

int g_fail = 0;

#define EXPECT(cond, ...)                         \
  do {                                            \
    if (!(cond)) {                                \
      if (g_fail < 20) {                          \
        std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        std::printf(__VA_ARGS__);                 \
        std::printf("\n");                        \
      }                                           \
      ++g_fail;                                   \
    }                                             \
  } while (0)

struct Header {
  char magic[8];
  uint32_t n, obs_dim, act_dim, crc_f32, crc_i8, k, reserved;
};

template <typename T>
bool read_arr(std::FILE* f, std::vector<T>& v, size_t n) {
  v.resize(n);
  return n == 0 || std::fread(v.data(), sizeof(T), n, f) == n;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s vectors.bin\n", argv[0]);
    return 2;
  }
  std::FILE* f = std::fopen(argv[1], "rb");
  if (!f) {
    std::printf("cannot open %s\n", argv[1]);
    return 2;
  }
  Header h;
  static_assert(sizeof(Header) == 36, "header packing");
  if (std::fread(&h, sizeof h, 1, f) != 1 || std::memcmp(h.magic, "NFVEC01", 8) != 0) {
    std::printf("bad header\n");
    return 2;
  }
  const size_t N = h.n, od = h.obs_dim, ad = h.act_dim, K = h.k;
  std::vector<float> obs, out_q, out_f;
  std::vector<int8_t> xq;
  std::vector<int32_t> acc, rs_s;
  std::vector<int64_t> rs_v, rs_r;
  bool ok = read_arr(f, obs, N * od) && read_arr(f, xq, N * od) && read_arr(f, acc, N * ad) &&
            read_arr(f, out_q, N * ad) && read_arr(f, out_f, N * ad) && read_arr(f, rs_v, K) &&
            read_arr(f, rs_s, K) && read_arr(f, rs_r, K);
  char extra;
  ok = ok && std::fread(&extra, 1, 1, f) == 0;
  std::fclose(f);
  if (!ok) {
    std::printf("truncated/oversized vector file\n");
    return 2;
  }

  const nf::FloatModel& F = nf_gen::kPolicyF32;
  const nf::Int8Model& Q = nf_gen::kPolicyI8;
  std::printf("policy %s: %u layers, %zu vectors, %zu rshift cases\n", nf_gen::kPolicyName, Q.n_layers, N, K);

  // ---- structure + CRC ------------------------------------------------------
  EXPECT(nf::validate(F), "validate(FloatModel)");
  EXPECT(nf::validate(Q), "validate(Int8Model)");
  EXPECT(F.obs_dim == od && F.act_dim == ad, "dims %u/%u vs file %zu/%zu", F.obs_dim, F.act_dim, od, ad);
  const uint32_t cf = nf::policy_crc32(F), ci = nf::policy_crc32(Q);
  EXPECT(cf == F.crc32, "f32 crc computed 0x%08X embedded 0x%08X", cf, F.crc32);
  EXPECT(ci == Q.crc32, "i8 crc computed 0x%08X embedded 0x%08X", ci, Q.crc32);
  EXPECT(cf == h.crc_f32, "f32 crc C++ 0x%08X python 0x%08X", cf, h.crc_f32);
  EXPECT(ci == h.crc_i8, "i8 crc C++ 0x%08X python 0x%08X", ci, h.crc_i8);
  if (g_fail) {
    std::printf("structural failures, aborting\n");
    return 1;
  }

  // ---- rshift_round -------------------------------------------------------
  size_t rs_bad = 0;
  for (size_t k = 0; k < K; ++k) {
    int64_t r = nf::rshift_round(rs_v[k], rs_s[k]);
    if (r != rs_r[k]) {
      ++rs_bad;
      EXPECT(false, "rshift_round(%lld, %d) = %lld, python %lld", (long long)rs_v[k], rs_s[k], (long long)r,
             (long long)rs_r[k]);
    }
  }

  // ---- inference ------------------------------------------------------------
  size_t xq_bad = 0, acc_bad = 0, outq_bad = 0, outf_bad = 0, nan_rows = 0;
  double max_dq = 0, max_df = 0;
  std::vector<float> yq(ad), yf(ad);
  std::vector<int32_t> la(ad);
  std::vector<int8_t> lx(od);
  for (size_t n = 0; n < N; ++n) {
    const float* o = &obs[n * od];
    bool okq = nf::mlp_forward_int8(Q, o, yq.data(), la.data(), lx.data());
    bool okf = nf::mlp_forward(F, o, yf.data());
    EXPECT(okq && okf, "forward returned false at row %zu", n);
    for (size_t i = 0; i < od; ++i)
      if (lx[i] != xq[n * od + i]) {
        ++xq_bad;
        EXPECT(false, "row %zu x_q[%zu]: C++ %d golden %d (obs %.9g)", n, i, lx[i], xq[n * od + i], (double)o[i]);
      }
    bool row_nan = false;
    for (size_t j = 0; j < ad; ++j) {
      const size_t k = n * ad + j;
      if (la[j] != acc[k]) {
        ++acc_bad;
        EXPECT(false, "row %zu acc[%zu]: C++ %d golden %d", n, j, la[j], acc[k]);
      }
      const double dq = std::fabs((double)yq[j] - (double)out_q[k]);
      if (!(dq <= 1e-6)) {
        ++outq_bad;
        EXPECT(false, "row %zu out_q[%zu]: C++ %.9g golden %.9g", n, j, (double)yq[j], (double)out_q[k]);
      } else if (dq > max_dq) {
        max_dq = dq;
      }
      const bool cn = std::isnan(yf[j]), pn = std::isnan(out_f[k]);
      if (cn || pn) {
        row_nan = true;
        if (cn != pn) {
          ++outf_bad;
          EXPECT(false, "row %zu out_f[%zu]: NaN mismatch C++ %.9g numpy %.9g", n, j, (double)yf[j],
                 (double)out_f[k]);
        }
      } else {
        const double df = std::fabs((double)yf[j] - (double)out_f[k]);
        if (!(df <= 1e-5)) {
          ++outf_bad;
          EXPECT(false, "row %zu out_f[%zu]: C++ %.9g numpy %.9g", n, j, (double)yf[j], (double)out_f[k]);
        } else if (df > max_df) {
          max_df = df;
        }
      }
    }
    nan_rows += row_nan;
  }

  std::printf("crc f32 0x%08X i8 0x%08X (match python)\n", cf, ci);
  std::printf("rshift mismatches %zu/%zu\n", rs_bad, K);
  std::printf("x_q mismatches %zu/%zu, acc mismatches %zu/%zu\n", xq_bad, N * od, acc_bad, N * ad);
  std::printf("int8 out: %zu outside 1e-6 (max |d| %.3g)\n", outq_bad, max_dq);
  std::printf("float out: %zu outside 1e-5 (max |d| %.3g), %zu rows with NaN (matched)\n", outf_bad, max_df,
              nan_rows);
  std::printf("%s\n", g_fail ? "FAILED" : "PASSED");
  return g_fail ? 1 : 0;
}
