#include "nf/estimator.hpp"

namespace nf {

void Mahony::init(const Config& c) {
  c_ = c;
  reset();
}

void Mahony::reset(const Quat& q) {
  q_ = q.normalized();
  bias_ = {};
  innov_ = 0.0f;
  t_ = 0.0f;
}

void Mahony::init_from_accel(const Vec3& a) {
  Vec3 u = normalized(a);
  if (norm(u) < 0.5f) return;
  // Rotation taking body "up" estimate u to... we want R^T e_z = u.
  float roll = std::atan2(u.y, u.z);
  float pitch = std::atan2(-u.x, std::sqrt(u.y * u.y + u.z * u.z));
  q_ = Quat::from_euler(roll, pitch, 0.0f);
  innov_ = 0.0f;
}

void Mahony::update(const Vec3& gyro, const Vec3& accel, float dt) {
  if (!(dt > 0.0f) || dt > 0.1f) return;
  t_ += dt;
  Vec3 omega = gyro - bias_;
  float an = norm(accel);
  float g = c_.gravity;
  const bool calm = norm(omega) < c_.max_rate_for_accel || t_ < c_.init_time_s;
  if (calm && is_finite(an) && an > c_.accel_min_g * g && an < c_.accel_max_g * g) {
    Vec3 a = accel / an;
    Vec3 v = q_.to_mat().row(2);  // predicted up direction in B = R^T e_z
    Vec3 e = cross(a, v);
    float innov_inst = std::atan2(norm(e), dot(a, v));
    float k = dt / (1.0f / (2.0f * kPi * c_.innov_lpf_hz) + dt);
    innov_ += k * (innov_inst - innov_);
    float kp = t_ < c_.init_time_s ? c_.kp_init : c_.kp;
    if (c_.ki > 0.0f && t_ >= c_.init_time_s) {
      bias_ -= e * (c_.ki * dt);
      bias_ = clamp(bias_, {c_.bias_limit, c_.bias_limit, c_.bias_limit});
    }
    omega += e * kp;
  }
  Quat qn = q_.integrated(omega, dt);
  if (qn.finite()) q_ = qn;
}

void Mahony::set_bias(const Vec3& b) {
  if (!finite(b)) return;
  bias_ = clamp(b, {c_.bias_limit, c_.bias_limit, c_.bias_limit});
}

// ---------------------------------------------------------------- GyroCalibrator
void GyroCalibrator::init(const Config& c, float sample_hz) {
  c_ = c;
  window_n_ = static_cast<int>(c_.window_s * sample_hz + 0.5f);
  if (window_n_ < 2) window_n_ = 2;
  reset();
}

void GyroCalibrator::reset() {
  restart();
  bias_ = {};
  done_ = motion_ = false;
  windows_ = rejected_ = 0;
  avg_n_ = 0;
}

void GyroCalibrator::reject() {
  done_ = false;
  motion_ = true;
  avg_n_ = 0;
  ++rejected_;
  restart();
}

bool GyroCalibrator::update(const Vec3& gyro, const Vec3& accel) {
  const float g = c_.gravity;
  if (!finite(gyro) || !finite(accel)) { reject(); return false; }
  const float an = norm(accel);
  if (std::fabs(gyro.x) > c_.motion_gyro || std::fabs(gyro.y) > c_.motion_gyro ||
      std::fabs(gyro.z) > c_.motion_gyro || std::fabs(an - g) > c_.motion_accel) {
    reject();
    return false;
  }
  if (n_ == 0) {
    ref_g_ = gyro;
    ref_a_ = accel;
    sg_ = sg2_ = sa_ = sa2_ = {};
  }
  const Vec3 dg = gyro - ref_g_, da = accel - ref_a_;
  sg_ += dg;
  sg2_ += hadamard(dg, dg);
  sa_ += da;
  sa2_ += hadamard(da, da);
  if (++n_ < window_n_) return false;

  const float inv = 1.0f / static_cast<float>(n_);
  const Vec3 mg = sg_ * inv, ma = sa_ * inv;
  bool still = true;
  for (int i = 0; i < 3; ++i) {
    const float vg = sg2_[i] * inv - mg[i] * mg[i];
    const float va = sa2_[i] * inv - ma[i] * ma[i];
    if (!(vg <= sq(c_.max_gyro_std)) || !(va <= sq(c_.max_accel_std))) still = false;
    if (!(std::fabs(ref_g_[i] + mg[i]) <= c_.max_gyro_mean)) still = false;
  }
  if (!(std::fabs(norm(ref_a_ + ma) - g) <= c_.max_accel_err)) still = false;
  if (!still) { reject(); return false; }
  if (avg_n_ < c_.max_avg_windows) ++avg_n_;
  bias_ += ((ref_g_ + mg) - bias_) * (1.0f / static_cast<float>(avg_n_));  // avg_n_ == 1: replace
  done_ = true;
  motion_ = false;
  ++windows_;
  restart();
  return true;
}

void TranslationKf::init(const Config& c) {
  c_ = c;
  reset();
}

void TranslationKf::reset(const Vec3& p, const Vec3& v) {
  for (int i = 0; i < 3; ++i) {
    x_[i] = Axis{};
    x_[i].p = p[i];
    x_[i].v = v[i];
    x_[i].P[0][0] = 0.01f;
    x_[i].P[1][1] = 0.01f;
  }
  nis_lpf_ = 0.0f;
  height_age_ = flow_age_ = 1e3f;
  rejected_ = 0;
}

void TranslationKf::predict_axis(Axis& a, float acc, float dt) {
  a.p += a.v * dt + 0.5f * acc * dt * dt;
  a.v += acc * dt;
  // P = F P F^T + Q, F = [[1, dt],[0, 1]], Q from white accel noise.
  float p00 = a.P[0][0], p01 = a.P[0][1], p10 = a.P[1][0], p11 = a.P[1][1];
  float q = c_.accel_noise * c_.accel_noise;
  float n00 = p00 + dt * (p10 + p01) + dt * dt * p11 + q * dt * dt * dt / 3.0f;
  float n01 = p01 + dt * p11 + q * dt * dt / 2.0f;
  float n11 = p11 + q * dt;
  a.P[0][0] = n00; a.P[0][1] = n01; a.P[1][0] = n01; a.P[1][1] = n11;
}

bool TranslationKf::update_axis(Axis& a, int idx, float z, float r2) {
  float hx = idx == 0 ? a.p : a.v;
  float y = z - hx;
  float s = a.P[idx][idx] + r2;
  float nis = y * y / s;
  float k = dt_last_ / (1.0f / (2.0f * kPi * c_.nis_lpf_hz) + dt_last_);
  nis_lpf_ += k * (nis - nis_lpf_);
  if (!(nis <= c_.nis_gate)) { ++rejected_; return false; }
  float k0 = a.P[0][idx] / s, k1 = a.P[1][idx] / s;
  a.p += k0 * y;
  a.v += k1 * y;
  float pi0 = a.P[idx][0], pi1 = a.P[idx][1];
  a.P[0][0] -= k0 * pi0; a.P[0][1] -= k0 * pi1;
  a.P[1][0] -= k1 * pi0; a.P[1][1] -= k1 * pi1;
  float sym = 0.5f * (a.P[0][1] + a.P[1][0]);
  a.P[0][1] = a.P[1][0] = sym;
  return true;
}

void TranslationKf::predict(const Vec3& accel_b, const Quat& q, float dt) {
  if (!(dt > 0.0f) || dt > 0.1f) return;
  dt_last_ = dt;
  Vec3 aw = q.rotate(accel_b);
  aw.z -= c_.gravity;
  if (!finite(aw)) aw = {};
  for (int i = 0; i < 3; ++i) predict_axis(x_[i], aw[i], dt);
  height_age_ += dt;
  flow_age_ += dt;
}

void TranslationKf::correct(const FlowSample& flow, const RangeSample& range, const Quat& q, const Vec3& omega) {
  Mat3 R = q.to_mat();
  float r22 = R(2, 2);
  float tilt = std::acos(clampf(r22, -1.0f, 1.0f));
  if (range.valid && is_finite(range.range_m) && range.range_m >= 0.0f && range.range_m < c_.max_range &&
      tilt < c_.max_tilt_range) {
    float h = range.range_m * r22;
    if (update_axis(x_[2], 0, h, c_.range_noise * c_.range_noise)) height_age_ = 0.0f;
  }
  float h = x_[2].p;
  if (flow.valid && flow.quality >= c_.min_quality && h > c_.min_height_flow && height_valid() &&
      is_finite(flow.flow_x) && is_finite(flow.flow_y) && tilt < c_.max_tilt_range) {
    float hb = h / r22;  // distance along the optical axis
    Vec3 vb_est = R.tmul(vel());
    Vec3 vb{hb * (flow.flow_x + omega.y), hb * (flow.flow_y - omega.x), vb_est.z};
    Vec3 vw = R * vb;
    float r2 = c_.flow_vel_noise * c_.flow_vel_noise;
    bool ok0 = update_axis(x_[0], 1, vw.x, r2);
    bool ok1 = update_axis(x_[1], 1, vw.y, r2);
    if (ok0 && ok1) flow_age_ = 0.0f;
  }
}

bool TranslationKf::healthy() const {
  Vec3 p = pos(), v = vel();
  return finite(p) && finite(v) && nis_lpf_ < 0.5f * c_.nis_gate;
}

}  // namespace nf
