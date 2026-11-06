#include "nf/control.hpp"

namespace nf {

namespace {
// Motor sign patterns from INTERFACES.md section 2.
constexpr float kRollSign[4] = {+1, -1, -1, +1};
constexpr float kPitchSign[4] = {-1, -1, +1, +1};
constexpr float kYawSign[4] = {+1, -1, +1, -1};

float spread(const float u[4]) {
  float lo = u[0], hi = u[0];
  for (int i = 1; i < 4; ++i) { lo = u[i] < lo ? u[i] : lo; hi = u[i] > hi ? u[i] : hi; }
  return hi - lo;
}
}  // namespace

// ---------------------------------------------------------------- Mixer
void Mixer::init(const MixerParams& p) {
  p_ = p;
  t_min_ = cmd_to_thrust(p_.idle_command);
}

float Mixer::cmd_to_thrust(float c) const {
  float w = clampf(c, 0.0f, 1.0f);
  return p_.thrust_max * (p_.curve_a * w * w + (1.0f - p_.curve_a) * w);
}

float Mixer::thrust_to_cmd(float t) const {
  float tn = clampf(t / p_.thrust_max, 0.0f, 1.0f);
  float a = p_.curve_a, b = 1.0f - a;
  float w = a > 1e-6f ? (-b + std::sqrt(b * b + 4.0f * a * tn)) / (2.0f * a) : tn / b;
  return clampf(w, p_.idle_command, 1.0f);
}

Vec3 Mixer::torque_from_thrusts(const float t[4]) const {
  Vec3 tau{};
  for (int i = 0; i < 4; ++i) {
    tau.x += p_.arm_xy * kRollSign[i] * t[i];
    tau.y += p_.arm_xy * kPitchSign[i] * t[i];
    tau.z += p_.yaw_k * kYawSign[i] * t[i];
  }
  return tau;
}

MixerResult Mixer::mix(float thrust_n, const Vec3& torque) const {
  MixerResult r;
  const float lo = t_min_, hi = p_.thrust_max, rng = hi - lo;
  const float tau_x = is_finite(torque.x) ? torque.x : 0.0f;
  const float tau_y = is_finite(torque.y) ? torque.y : 0.0f;
  const float tau_z = is_finite(torque.z) ? torque.z : 0.0f;
  const float base = is_finite(thrust_n) ? thrust_n / 4.0f : 0.0f;

  // (1) roll/pitch differential thrusts, scaled to fit the range.
  float t_rp[4], t_y[4], T[4];
  for (int i = 0; i < 4; ++i)
    t_rp[i] = (tau_x / (4.0f * p_.arm_xy)) * kRollSign[i] + (tau_y / (4.0f * p_.arm_xy)) * kPitchSign[i];
  float span = spread(t_rp);
  float scale_rp = rng / (span > 1e-9f ? span : 1e-9f);
  if (scale_rp > 1.0f) scale_rp = 1.0f;
  float rp_min = 1e30f, rp_max = -1e30f;
  for (float& v : t_rp) {
    v *= scale_rp;
    rp_min = v < rp_min ? v : rp_min;
    rp_max = v > rp_max ? v : rp_max;
  }
  // (2) collective shifted so roll/pitch fit: min(max(base, lo - min), hi - max).
  float base_c = base > lo - rp_min ? base : lo - rp_min;
  base_c = base_c < hi - rp_max ? base_c : hi - rp_max;
  for (int i = 0; i < 4; ++i) T[i] = base_c + t_rp[i];
  // (3) yaw with the largest s in [0,1] such that lo <= T + s*t_y <= hi.
  float s_y = 1e30f;
  for (int i = 0; i < 4; ++i) {
    t_y[i] = (tau_z / (4.0f * p_.yaw_k)) * kYawSign[i];
    float room = t_y[i] > 0.0f ? hi - T[i] : T[i] - lo;
    if (room < 0.0f) room = 0.0f;
    float ay = std::fabs(t_y[i]);
    float si = room / (ay > 1e-12f ? ay : 1e-12f);
    s_y = si < s_y ? si : s_y;
  }
  if (s_y > 1.0f) s_y = 1.0f;
  for (int i = 0; i < 4; ++i) {
    r.thrust[i] = clampf(T[i] + s_y * t_y[i], lo, hi);
    r.cmd[i] = thrust_to_cmd(r.thrust[i]);
  }
  r.rp_scale = scale_rp;
  r.yaw_scale = s_y;
  r.rp_saturated = scale_rp < 1.0f;
  r.yaw_saturated = s_y < 1.0f;
  r.thrust_saturated = std::fabs(base_c - base) > 1e-6f;
  return r;
}

// ---------------------------------------------------------------- RatePid
void RatePid::init(const RatePidConfig& c, float fs) {
  c_ = c;
  dt_ = 1.0f / fs;
  d_alpha_ = 1.0f - std::exp(-2.0f * kPi * c_.d_lpf_hz * dt_);
  for (int i = 0; i < 3; ++i) sp_lpf_[i].init(c_.relax_lpf_hz, fs);
  reset();
}

void RatePid::reset() {
  i_ = {};
  prev_rate_ = {};
  d_filt_ = {};
  e_ = {};
  first_ = true;
  for (int i = 0; i < 3; ++i) { sp_lpf_[i].reset(); relax_[i] = 1.0f; }
}

Vec3 RatePid::update(const Vec3& sp, const Vec3& rate) {
  if (first_) {  // == sim reset(idx, gyro): no derivative kick on the first step
    prev_rate_ = rate;
    for (int i = 0; i < 3; ++i) sp_lpf_[i].reset(sp[i]);
    first_ = false;
  }
  Vec3 alpha{};
  for (int ax = 0; ax < 3; ++ax) {
    e_[ax] = sp[ax] - rate[ax];
    float u = c_.kp[ax] * e_[ax];
    if (c_.kd[ax] != 0.0f) {
      float d = -(rate[ax] - prev_rate_[ax]) / dt_;
      d_filt_[ax] = d_filt_[ax] + d_alpha_ * (d - d_filt_[ax]);
      u += c_.kd[ax] * d_filt_[ax];
    }
    u += c_.ki[ax] * i_[ax];
    if (c_.out_limit > 0.0f) u = clampf(u, -c_.out_limit, c_.out_limit);
    alpha[ax] = u;
    if (c_.relax_threshold > 0.0f) {
      float hp = sp[ax] - sp_lpf_[ax].apply(sp[ax]);
      relax_[ax] = clampf(1.0f - std::fabs(hp) / c_.relax_threshold, 0.0f, 1.0f);
    }
  }
  prev_rate_ = rate;
  return hadamard(alpha, c_.inertia);
}

void RatePid::integrate(bool mixer_saturated) {
  if (mixer_saturated) return;  // frozen while the mixer saturates
  for (int ax = 0; ax < 3; ++ax)
    i_[ax] = clampf(i_[ax] + e_[ax] * relax_[ax] * dt_, -c_.i_limit[ax], c_.i_limit[ax]);
}

// ---------------------------------------------------------------- Attitude
Vec3 attitude_p(const Quat& q, const Quat& q_sp, const AttitudeConfig& c) {
  Quat qe = q.conj() * q_sp;  // error expressed in body frame
  float sgn = qe.w >= 0.0f ? 2.0f : -2.0f;
  Vec3 e = qe.vec() * sgn;  // ~ rotation vector for small errors
  Vec3 r = hadamard(e, c.kp);
  return clamp(r, c.rate_limit);
}

// ---------------------------------------------------------------- Cascade
void CascadeController::init(const CascadeConfig& c) {
  c_ = c;
  reset();
}

void CascadeController::reset() {
  vi_ = {};
  q_sp_ = {};
}

CtbrCommand CascadeController::update(const VehicleState& s, const Vec3& pos_sp, float yaw_sp, CascadeMode mode,
                                      float dt) {
  const bool use_xy = s.pos_valid && mode != CascadeMode::LEVEL;
  const bool use_z = s.height_valid && mode != CascadeMode::LEVEL;
  Vec3 vel_sp{};
  if (use_xy) {
    vel_sp.x = c_.pos_kp.x * (pos_sp.x - s.pos.x);
    vel_sp.y = c_.pos_kp.y * (pos_sp.y - s.pos.y);
  }
  if (mode == CascadeMode::HOLD && use_z)
    vel_sp.z = c_.pos_kp.z * (pos_sp.z - s.pos.z);
  else
    vel_sp.z = -c_.descent_rate;  // DESCENT, LEVEL, or no height: sink slowly
  vel_sp = clamp(vel_sp, c_.vel_limit);
  if (mode == CascadeMode::DESCENT) vel_sp.z = -c_.descent_rate;

  Vec3 ev = vel_sp - s.vel;
  if (!use_xy) { ev.x = 0.0f; ev.y = 0.0f; }
  if (!finite(ev)) ev = {};
  Vec3 acc = hadamard(c_.vel_kp, ev);
  if (dt > 0.0f) {
    vi_ += hadamard(c_.vel_ki, ev) * dt;
    vi_ = clamp(vi_, c_.vel_i_limit);
  }
  if (!use_xy) { vi_.x = 0.0f; vi_.y = 0.0f; }
  acc += vi_;
  acc.z = clampf(acc.z, c_.min_acc_z, c_.max_acc_z);

  // Desired thrust vector (W), tilt-limited.
  Vec3 f{acc.x, acc.y, acc.z + c_.gravity};
  f *= c_.mass;
  if (f.z < 0.1f * c_.mass * c_.gravity) f.z = 0.1f * c_.mass * c_.gravity;
  float h = std::sqrt(f.x * f.x + f.y * f.y);
  float hmax = f.z * std::tan(c_.max_tilt);
  if (h > hmax && h > 1e-9f) { f.x *= hmax / h; f.y *= hmax / h; }

  Vec3 zb = normalized(f);
  Vec3 xc{std::cos(yaw_sp), std::sin(yaw_sp), 0.0f};
  Vec3 yb = normalized(cross(zb, xc));
  Vec3 xb = cross(yb, zb);
  Mat3 R;
  for (int i = 0; i < 3; ++i) { R.m[i][0] = xb[i]; R.m[i][1] = yb[i]; R.m[i][2] = zb[i]; }
  q_sp_ = Quat::from_mat(R);

  CtbrCommand cmd;
  Vec3 zb_now = s.q.to_mat().col(2);
  cmd.thrust_n = clampf(dot(f, zb_now), 0.0f, 1e3f);
  cmd.rate_sp = attitude_p(s.q, q_sp_, c_.att);
  return cmd;
}

}  // namespace nf
