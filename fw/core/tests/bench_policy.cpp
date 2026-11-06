// Rough host timing of the 2x64 MLP (float32 and int8). Not a target WCET.
#include <chrono>
#include <cstdio>

#include "model_fixture.hpp"
#include "nf/policy.hpp"

int main() {
  nftest::MlpFixture f;
  int hidden[2] = {64, 64};
  f.build(hidden, 2, 1234);
  float obs[nf::kObsDim], out[nf::kActDim];
  nftest::Lcg r(1);
  f.random_obs(r, obs);
  const int N = 20000;
  volatile float sink = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) { obs[0] += 1e-6f; nf::mlp_forward(f.fm, obs, out); sink = sink + out[0]; }
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) { obs[0] += 1e-6f; nf::mlp_forward_int8(f.qm, obs, out); sink = sink + out[0]; }
  auto t2 = std::chrono::steady_clock::now();
  double f_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  double q_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / N;
  std::printf("2x64 MLP host per-call: float32 %.3f us, int8 %.3f us\n", f_us, q_us);
  return 0;
}
