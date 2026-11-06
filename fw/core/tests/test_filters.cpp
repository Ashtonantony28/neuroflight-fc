#include "nf/filters.hpp"
#include "nftest.hpp"

using namespace nf;

namespace {
constexpr float kFs = 1000.0f;

// Simulated steady-state gain of a sine through the filter (amplitude ratio).
template <class F>
float sine_gain(F& filt, float f_hz, float fs = kFs) {
  const int n = static_cast<int>(fs * 2.0f);  // 2 s
  float peak = 0.0f;
  for (int i = 0; i < n; ++i) {
    float x = std::sin(2.0f * kPi * f_hz * static_cast<float>(i) / fs);
    float y = filt.apply(x);
    if (i > n / 2) peak = std::fmax(peak, std::fabs(y));
  }
  return peak;
}
float db(float g) { return 20.0f * std::log10(g); }
}  // namespace

TEST(biquad_lowpass_response) {
  const float fc = 100.0f;
  struct { float f, expect_db, tol; } pts[] = {{10, 0.0f, 0.1f}, {100, -3.01f, 0.3f}, {300, -25.1f, 0.5f}  /* bilinear: extra attenuation vs analog -19 dB */};
  for (auto& p : pts) {
    Biquad b; b.init_lowpass(fc, kFs);
    float g = sine_gain(b, p.f);
    CHECK_NEAR(db(g), p.expect_db, p.tol);
    CHECK_NEAR(db(b.magnitude(p.f, kFs)), db(g), 0.2);
  }
  Biquad b; b.init_lowpass(fc, kFs);
  CHECK(db(sine_gain(b, 450.0f)) < -30.0f);
}

TEST(pt1_response) {
  Pt1 p; p.init(50.0f, kFs);
  CHECK_NEAR(sine_gain(p, 5.0f), 1.0f, 0.02);
  Pt1 p2; p2.init(50.0f, kFs);
  CHECK_NEAR(db(sine_gain(p2, 50.0f)), -3.65f, 0.3);  // k = dt/(RC+dt) discretisation: -3.65 dB at fc
  Pt1 p3; p3.init(50.0f, kFs);
  CHECK(db(sine_gain(p3, 400.0f)) < -15.0f);
  // Step response: reaches 63% after one RC-ish time constant.
  Pt1 s; s.init(10.0f, kFs);
  float y = 0;
  for (int i = 0; i < 16; ++i) y = s.apply(1.0f);  // tau = 15.9 ms
  CHECK_NEAR(y, 0.632f, 0.03);
}

TEST(notch_response) {
  Biquad n; n.init_notch(200.0f, kFs, 3.0f);
  CHECK(db(sine_gain(n, 200.0f)) < -30.0f);
  Biquad n2; n2.init_notch(200.0f, kFs, 3.0f);
  CHECK_NEAR(db(sine_gain(n2, 20.0f)), 0.0f, 0.3);
  Biquad n3; n3.init_notch(200.0f, kFs, 3.0f);
  CHECK_NEAR(db(sine_gain(n3, 450.0f)), 0.0f, 1.0);
}

TEST(dynamic_notch_tracks_center) {
  // Retune the notch while filtering a tone that moves with it.
  Biquad n; n.init_notch(150.0f, kFs, 3.0f);
  float f = 150.0f, phase = 0.0f, peak = 0.0f;
  for (int i = 0; i < 4000; ++i) {
    f = 150.0f + 100.0f * static_cast<float>(i) / 4000.0f;  // sweep 150 -> 250 Hz
    n.update_notch(f, kFs, 3.0f);
    phase += 2.0f * kPi * f / kFs;
    float y = n.apply(std::sin(phase));
    if (i > 500) peak = std::fmax(peak, std::fabs(y));
  }
  CHECK(peak < 0.1f);
  // A fixed notch at 150 Hz would pass the 250 Hz tone.
  CHECK(n.magnitude(250.0f, kFs) < 0.01f);
}

TEST(rpm_notch_bank) {
  RpmNotchBank bank; bank.init(kFs, 3.0f, 80.0f);
  float hz[4] = {180, 190, 200, 210};
  bank.update(hz);
  for (int m = 0; m < 4; ++m) CHECK(bank.active(m));
  // Tone at motor 2 frequency is removed on all axes; low-frequency motion passes.
  float peak = 0.0f, peak_lo = 0.0f;
  for (int i = 0; i < 2000; ++i) {
    float t = static_cast<float>(i) / kFs;
    float s = std::sin(2 * kPi * 200.0f * t);
    Vec3 y = bank.apply({s, s, s});
    if (i > 1000) peak = std::fmax(peak, std::fabs(y.y));
  }
  RpmNotchBank b2; b2.init(kFs, 3.0f, 80.0f); b2.update(hz);
  for (int i = 0; i < 2000; ++i) {
    float t = static_cast<float>(i) / kFs;
    Vec3 y = b2.apply({std::sin(2 * kPi * 5.0f * t), 0, 0});
    if (i > 1000) peak_lo = std::fmax(peak_lo, std::fabs(y.x));
  }
  CHECK(peak < 0.05f);
  CHECK_NEAR(peak_lo, 1.0f, 0.03);
  // Below min_hz / NaN -> bypassed.
  float off[4] = {10, NAN, 200, 200};
  bank.update(off);
  CHECK(!bank.active(0)); CHECK(!bank.active(1)); CHECK(bank.active(2));
}

NFTEST_MAIN()
