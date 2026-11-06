#include "nf/filters.hpp"

namespace nf {

void Pt1::init(float cutoff_hz, float sample_hz) {
  set_cutoff(cutoff_hz, sample_hz);
  y_ = 0.0f;
}

void Pt1::set_cutoff(float cutoff_hz, float sample_hz) {
  if (cutoff_hz <= 0.0f) { k_ = 1.0f; return; }
  float rc = 1.0f / (2.0f * kPi * cutoff_hz);
  float dt = 1.0f / sample_hz;
  k_ = dt / (rc + dt);
}

void Biquad::init_passthrough() {
  b0_ = 1; b1_ = b2_ = a1_ = a2_ = 0;
  reset();
}

void Biquad::init_lowpass(float fc, float fs, float q) {
  fc = clampf(fc, 1e-3f, 0.49f * fs);
  float w0 = 2.0f * kPi * fc / fs;
  float sn = std::sin(w0), cs = std::cos(w0);
  float alpha = sn / (2.0f * q);
  float a0 = 1.0f + alpha;
  b0_ = (1.0f - cs) * 0.5f / a0;
  b1_ = (1.0f - cs) / a0;
  b2_ = b0_;
  a1_ = -2.0f * cs / a0;
  a2_ = (1.0f - alpha) / a0;
  reset();
}

void Biquad::set_notch_coeffs(float fc, float fs, float q) {
  fc = clampf(fc, 1e-3f, 0.48f * fs);
  float w0 = 2.0f * kPi * fc / fs;
  float sn = std::sin(w0), cs = std::cos(w0);
  float alpha = sn / (2.0f * q);
  float a0 = 1.0f + alpha;
  b0_ = 1.0f / a0;
  b1_ = -2.0f * cs / a0;
  b2_ = b0_;
  a1_ = b1_;
  a2_ = (1.0f - alpha) / a0;
}

void Biquad::init_notch(float fc, float fs, float q) {
  set_notch_coeffs(fc, fs, q);
  reset();
}

void Biquad::update_notch(float fc, float fs, float q) { set_notch_coeffs(fc, fs, q); }

float Biquad::magnitude(float f, float fs) const {
  // H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2), z = e^jw
  float w = 2.0f * kPi * f / fs;
  float c1 = std::cos(w), s1 = std::sin(w), c2 = std::cos(2 * w), s2 = std::sin(2 * w);
  float nr = b0_ + b1_ * c1 + b2_ * c2, ni = -(b1_ * s1 + b2_ * s2);
  float dr = 1.0f + a1_ * c1 + a2_ * c2, di = -(a1_ * s1 + a2_ * s2);
  return std::sqrt((nr * nr + ni * ni) / (dr * dr + di * di));
}

void RpmNotchBank::init(float sample_hz, float q, float min_hz) {
  fs_ = sample_hz; q_ = q; min_hz_ = min_hz;
  for (int m = 0; m < kMotors; ++m) {
    active_[m] = false;
    for (auto& f : n_[m]) f.init_notch(min_hz_, fs_, q_);
  }
}

void RpmNotchBank::update(const float motor_hz[kMotors]) {
  for (int m = 0; m < kMotors; ++m) {
    float f = motor_hz[m];
    bool ok = is_finite(f) && f >= min_hz_;
    if (ok) {
      f = clampf(f, min_hz_, 0.45f * fs_);
      for (auto& n : n_[m]) n.update_notch(f, fs_, q_);
    }
    if (ok && !active_[m])
      for (auto& n : n_[m]) n.reset();  // fresh state when (re)activating
    active_[m] = ok;
  }
}

Vec3 RpmNotchBank::apply(const Vec3& v) {
  Vec3 o = v;
  for (int m = 0; m < kMotors; ++m) {
    if (!active_[m]) continue;
    o = {n_[m][0].apply(o.x), n_[m][1].apply(o.y), n_[m][2].apply(o.z)};
  }
  return o;
}

}  // namespace nf
