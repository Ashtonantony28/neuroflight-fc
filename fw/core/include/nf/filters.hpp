// Digital filters: PT1, biquad low-pass / notch (RBJ cookbook, direct form II
// transposed), dynamic RPM notch bank. All float32, no heap.
#pragma once

#include "nf/math.hpp"

namespace nf {

// First-order low-pass, y += k (x - y), k = dt / (RC + dt).
class Pt1 {
 public:
  void init(float cutoff_hz, float sample_hz);
  void set_cutoff(float cutoff_hz, float sample_hz);
  float apply(float x) { y_ += k_ * (x - y_); return y_; }
  void reset(float v = 0.0f) { y_ = v; }
  float value() const { return y_; }

 private:
  float k_ = 1.0f, y_ = 0.0f;
};

class Biquad {
 public:
  void init_lowpass(float cutoff_hz, float sample_hz, float q = 0.70710678f);
  void init_notch(float center_hz, float sample_hz, float q);
  // Retune a notch in place WITHOUT resetting the delay state (for RPM tracking).
  void update_notch(float center_hz, float sample_hz, float q);
  void init_passthrough();
  float apply(float x) {
    float y = b0_ * x + s1_;
    s1_ = b1_ * x - a1_ * y + s2_;
    s2_ = b2_ * x - a2_ * y;
    return y;
  }
  void reset() { s1_ = s2_ = 0.0f; }
  // Analytic |H(e^jw)| at f (for diagnostics/tests).
  float magnitude(float f_hz, float sample_hz) const;

 private:
  void set_notch_coeffs(float center_hz, float sample_hz, float q);
  float b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
  float s1_ = 0, s2_ = 0;
};

// Vec3 convenience wrappers.
struct Pt1Vec3 {
  Pt1 f[3];
  void init(float c, float fs) { for (auto& e : f) e.init(c, fs); }
  Vec3 apply(const Vec3& v) { return {f[0].apply(v.x), f[1].apply(v.y), f[2].apply(v.z)}; }
  void reset(const Vec3& v) { f[0].reset(v.x); f[1].reset(v.y); f[2].reset(v.z); }
};
struct BiquadLpfVec3 {
  Biquad f[3];
  void init(float c, float fs) { for (auto& e : f) e.init_lowpass(c, fs); }
  Vec3 apply(const Vec3& v) { return {f[0].apply(v.x), f[1].apply(v.y), f[2].apply(v.z)}; }
};

// RPM filter: one notch per motor per axis at the motor's rotation frequency
// (fundamental). Notches whose frequency is below min_hz (or invalid) are
// bypassed; center is clamped to max_ratio * Nyquist.
class RpmNotchBank {
 public:
  static constexpr int kMotors = 4;
  void init(float sample_hz, float q = 3.0f, float min_hz = 80.0f);
  // motor_hz: rotation frequency of each motor in Hz (eRPM / pole_pairs / 60).
  void update(const float motor_hz[kMotors]);
  Vec3 apply(const Vec3& v);
  bool active(int motor) const { return active_[motor]; }

 private:
  Biquad n_[kMotors][3];
  bool active_[kMotors] = {false, false, false, false};
  float fs_ = 1000.0f, q_ = 3.0f, min_hz_ = 80.0f;
};

}  // namespace nf
