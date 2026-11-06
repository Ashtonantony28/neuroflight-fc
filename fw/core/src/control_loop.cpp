#include "nf/control_loop.hpp"

namespace nf {

uint32_t arming_checks(const ArmingInput& in, const ArmingConfig& c) {
  uint32_t f = 0;
  if (in.kill) f |= ARM_FAIL_KILL;
  if (!in.rc_link_ok) f |= ARM_FAIL_RC_LINK;
  if (!(in.throttle <= c.max_throttle)) f |= ARM_FAIL_THROTTLE;
  if (in.imu_b_valid) {
    if (!in.imu_ab_ready || !(norm(in.gyro_ab_diff) <= c.max_gyro_diff) ||
        !(norm(in.accel_ab_diff) <= c.max_accel_diff))
      f |= ARM_FAIL_IMU_DISAGREE;
  } else if (c.require_imu_b) {
    f |= ARM_FAIL_IMU_B_MISSING;
  }
  if (c.require_gyro_cal) {
    if (in.gyro_motion) f |= ARM_FAIL_MOTION;
    if (!in.gyro_cal_done) f |= ARM_FAIL_GYRO_CAL;
  }
  if (!in.est_converged) f |= ARM_FAIL_ESTIMATOR;
  if (!(in.tilt_rad <= c.max_tilt_rad)) f |= ARM_FAIL_TILT;
  if (c.require_vbat && (!in.vbat_valid || !(in.vbat >= c.min_vbat))) f |= ARM_FAIL_VBAT;
  if (in.vbat_valid && !(in.vbat >= c.min_vbat)) f |= ARM_FAIL_VBAT;
  if (in.want_policy && !in.policy_ok) f |= ARM_FAIL_POLICY;
  if (in.fault_flags) f |= ARM_FAIL_FAULTS;
  return f;
}

void ControlLoop::init(const ControlLoopConfig& c) {
  c_ = c;
  const float fs = static_cast<float>(c_.rate_hz);
  ctrl_div_ = c_.control_hz > 0 && c_.rate_hz >= c_.control_hz ? c_.rate_hz / c_.control_hz : 1;
  gyro_lpf_.init(c_.gyro_lpf_hz, fs);
  acc_lpf_.init(c_.accel_lpf_hz, fs);
  rpm_.init(fs);
  cal_.init(c_.gyro_cal, fs);
  for (int i = 0; i < 3; ++i) {
    gdiff_[i].init(c_.arming.imu_diff_lpf_hz, fs);
    adiff_[i].init(c_.arming.imu_diff_lpf_hz, fs);
  }
  imu_diff_n_ = 0;
  imu_diff_min_n_ = static_cast<int>(c_.arming.imu_diff_min_s * fs + 0.5f);
  hover_thrust_.init(c_.hover_est_lpf_hz, fs);
  hover_thrust_.reset(c_.cascade.mass * c_.cascade.gravity);
  mahony_.init(c_.mahony);
  kf_.init(c_.kf);
  pid_.init(c_.rate, fs);
  cascade_.init(c_.cascade);
  mixer_.init(c_.mixer);
  SupervisorConfig sc = c_.sup;
  sc.nominal_period_us = static_cast<uint32_t>(1000000 / c_.rate_hz);
  sup_.init(sc);
  target_ = {0.0f, 0.0f, c_.hold_height_m};
  first_ = true;
  armed_ = false;
}

void ControlLoop::set_policy(const FloatModel* m) {
  policy_.set_model(m);
  policy_crc_ok_ = m && policy_.valid() && policy_.computed_crc() == policy_.expected_crc();
}
void ControlLoop::set_policy(const Int8Model* m) {
  policy_.set_model(m);
  policy_crc_ok_ = m && policy_.valid() && policy_.computed_crc() == policy_.expected_crc();
}
bool ControlLoop::policy_ok() const { return policy_crc_ok_; }

void ControlLoop::set_target(const Vec3& pos, float yaw) {
  target_ = pos;
  target_yaw_ = yaw;
  if (sup_.mode() != FlightMode::FALLBACK_HOLD) { hold_sp_ = pos; hold_yaw_ = yaw; }
}

void ControlLoop::do_arm(uint64_t now, bool use_policy) {
  armed_ = true;
  // Arming point becomes the horizontal origin; z stays "height above ground".
  // Horizontal velocity is unobservable on the ground (no flow below
  // min_height_flow) and random-walks while disarmed, so it is zeroed: the
  // vehicle is at rest when armed (WS6 SITL fix: after ~60 s disarmed it had
  // drifted to ~5 m/s and tripped FAULT_ESTIMATOR right after takeoff).
  kf_.reset({0.0f, 0.0f, kf_.pos().z}, {0.0f, 0.0f, kf_.vel().z});
  hover_thrust_.reset(c_.cascade.mass * c_.cascade.gravity);
  pid_.reset();
  cascade_.reset();
  policy_.reset();
  for (float& a : action_) a = 0.0f;
  hold_sp_ = target_;
  hold_yaw_ = target_yaw_;
  sup_.arm(now, use_policy);
  prev_mode_ = sup_.mode();
}

void ControlLoop::latch_hold_point() {
  // Hold where we are, pulled inside the fence and above the floor.
  const SupervisorConfig& sc = sup_.config();
  Vec3 p = state_.pos;
  float r = std::sqrt(sq(p.x) + sq(p.y));
  float rmax = 0.8f * sc.fence_radius_m;
  if (r > rmax && r > 1e-6f) { p.x *= rmax / r; p.y *= rmax / r; }
  p.z = clampf(p.z, sc.height_floor_m + 0.25f, sc.fence_ceiling_m - 0.3f);
  if (!finite(p)) p = target_;
  hold_sp_ = p;
  hold_yaw_ = state_.q.yaw();
  // Bumpless transfer: start the fallback from the thrust that is holding the
  // vehicle up now instead of from nominal weight (was a ~0.35 m sag).
  if (c_.bumpless_handover) cascade_.seed_hover_thrust(hover_thrust_.value());
  else cascade_.reset();
}

MotorOutput ControlLoop::step(const ImuSample& imu, const SensorSample& sens, const RcInput& rc, uint64_t now) {
  MotorOutput out;
  float dt = 1.0f / static_cast<float>(c_.rate_hz);
  if (!first_ && now > last_us_) dt = clampf(static_cast<float>(now - last_us_) * 1e-6f, 1e-5f, 0.02f);
  first_ = false;
  last_us_ = now;
  ++step_count_;

  // ---- 1. sensing
  // At-rest gyro calibration while disarmed (raw samples; DC is unchanged by
  // the filters below). Each accepted window re-seeds the Mahony bias.
  if (!armed_ && cal_.update(imu.gyro, imu.accel)) mahony_.set_bias(cal_.bias());
  // IMU A/B agreement on low-passed differences (arming check).
  {
    const Vec3 gd = (imu.gyro - cal_.bias()) - imu.gyro_b;
    const Vec3 ad = imu.accel - imu.accel_b;
    if (imu.b_valid && finite(gd) && finite(ad)) {
      if (imu_diff_n_ == 0)
        for (int i = 0; i < 3; ++i) { gdiff_[i].reset(gd[i]); adiff_[i].reset(ad[i]); }
      for (int i = 0; i < 3; ++i) { gdiff_[i].apply(gd[i]); adiff_[i].apply(ad[i]); }
      if (imu_diff_n_ < imu_diff_min_n_) ++imu_diff_n_;
    } else {
      imu_diff_n_ = 0;
    }
  }
  if (c_.rpm_filter && sens.rpm_valid) rpm_.update(sens.motor_hz);
  Vec3 gyro = gyro_lpf_.apply(imu.gyro);
  if (c_.rpm_filter) gyro = rpm_.apply(gyro);
  Vec3 acc = acc_lpf_.apply(imu.accel);
  if (!finite(gyro)) gyro = {};
  mahony_.update(gyro, acc, dt);
  state_.q = mahony_.q();
  state_.omega = mahony_.rates(gyro);
  kf_.predict(acc, state_.q, dt);
  kf_.correct(sens.flow, sens.range, state_.q, state_.omega);
  state_.pos = kf_.pos();
  state_.vel = kf_.vel();
  state_.height_valid = kf_.height_valid();
  state_.pos_valid = kf_.height_valid() && kf_.flow_valid();
  const bool est_finite = state_.q.finite() && finite(state_.omega) && finite(state_.pos) && finite(state_.vel);

  // ---- 2. arm / disarm (kill has priority)
  const bool arm_edge = rc.arm && !prev_arm_sw_;
  prev_arm_sw_ = rc.arm;
  if (armed_ && !rc.arm) {
    armed_ = false;
    sup_.disarm();
  }
  if (!armed_ && arm_edge && !rc.kill) {
    ArmingInput ai;
    ai.imu_b_valid = imu.b_valid;
    ai.imu_ab_ready = imu_diff_n_ >= imu_diff_min_n_;
    for (int i = 0; i < 3; ++i) { ai.gyro_ab_diff[i] = gdiff_[i].value(); ai.accel_ab_diff[i] = adiff_[i].value(); }
    ai.gyro_cal_done = cal_.done();
    ai.gyro_motion = cal_.motion();
    ai.est_converged = mahony_.converged(c_.arming.est_max_innov, c_.arming.est_min_time_s) && est_finite;
    ai.tilt_rad = tilt_angle(state_.q);
    ai.vbat = sens.vbat;
    ai.vbat_valid = sens.vbat_valid;
    ai.want_policy = rc.policy_enable;
    ai.policy_ok = policy_crc_ok_;
    ai.throttle = rc.throttle;
    ai.rc_link_ok = rc.link_ok;
    ai.kill = rc.kill;
    arm_fail_ = arming_checks(ai, c_.arming);
    if (arm_fail_ == 0) do_arm(now, rc.policy_enable);
  }

  // ---- 3. policy (control-rate tick, only while flying the policy)
  const bool tick = (step_count_ % static_cast<uint64_t>(ctrl_div_)) == 1 % ctrl_div_;
  SupervisorInput si;
  si.now_us = now;
  si.armed = armed_;
  si.kill = rc.kill;
  si.rc_ok = rc.link_ok;
  si.q = state_.q;
  si.omega = state_.omega;
  si.pos = state_.pos;
  si.vel = state_.vel;
  si.pos_valid = state_.pos_valid;
  si.height_valid = state_.height_valid;
  si.accel_innov = mahony_.innovation();
  si.kf_nis = kf_.nis();
  si.est_finite = est_finite;
  si.gyro_a = imu.gyro;
  si.acc_a = imu.accel;
  si.gyro_b = imu.gyro_b;
  si.acc_b = imu.accel_b;
  si.imu_b_valid = imu.b_valid;
  if (armed_ && sup_.mode() == FlightMode::POLICY && tick) {
    uint32_t t0 = c_.clock_us ? c_.clock_us() : 0;
    bool ok = policy_crc_ok_ && policy_.run(state_, target_, action_);
    uint32_t t1 = c_.clock_us ? c_.clock_us() : 0;
    inference_us_ = t1 - t0;
    si.policy_ran = ok;
    si.policy_error = !ok;
    si.inference_us = inference_us_;
    for (int i = 0; i < kActDim; ++i) si.action[i] = action_[i];
  }

  // ---- 4. supervisor
  FlightMode mode = sup_.update(si);
  if (rc.reengage && !prev_reengage_ && sup_.request_reengage(now)) {
    mode = sup_.mode();
    policy_.reset();  // INTERFACES.md section 10: previous action = zeros on engage
  }
  prev_reengage_ = rc.reengage;
  const bool fallback = mode == FlightMode::FALLBACK_HOLD || mode == FlightMode::CONTROLLED_DESCENT;
  const bool entered_fallback = fallback && !(prev_mode_ == FlightMode::FALLBACK_HOLD ||
                                               prev_mode_ == FlightMode::CONTROLLED_DESCENT);
  if (entered_fallback && prev_mode_ == FlightMode::POLICY) latch_hold_point();
  prev_mode_ = mode;

  // ---- 5. outer loop
  if (mode == FlightMode::POLICY && si.policy_ran) {
    ctbr_ = decode_ctbr(action_);
  } else if (fallback && (tick || entered_fallback)) {
    float dtc = static_cast<float>(ctrl_div_) / static_cast<float>(c_.rate_hz);
    CascadeMode cm = mode == FlightMode::CONTROLLED_DESCENT ? CascadeMode::DESCENT : CascadeMode::HOLD;
    ctbr_ = cascade_.update(state_, hold_sp_, hold_yaw_, cm, dtc);
  }

  // ---- 6. rate loop + mixer
  const bool flying = armed_ && (mode == FlightMode::POLICY || fallback);
  if (flying) {
    if (ctbr_.thrust_n < c_.min_thrust_for_i * c_.cascade.mass * c_.cascade.gravity) pid_.reset();
    Vec3 tau = pid_.update(ctbr_.rate_sp, state_.omega);
    mix_ = mixer_.mix(ctbr_.thrust_n, tau);
    pid_.integrate(mix_.any_saturated());
    for (int i = 0; i < 4; ++i) out.cmd[i] = mix_.cmd[i];
    // Hover-thrust estimate: thrust that gives 1 g of body-z specific force.
    const float t_alloc = mix_.thrust[0] + mix_.thrust[1] + mix_.thrust[2] + mix_.thrust[3];
    const float w = c_.cascade.mass * c_.cascade.gravity;
    const float g = c_.cascade.gravity;
    if (state_.height_valid && state_.pos.z > c_.hover_est_min_height_m && est_finite &&
        tilt_angle(state_.q) < c_.hover_est_max_tilt && t_alloc > c_.min_thrust_for_i * w &&
        acc.z > 0.5f * g && acc.z < 1.5f * g)
      hover_thrust_.apply(clampf(g * t_alloc / acc.z, 0.5f * w, 2.0f * w));
  } else {
    mix_ = MixerResult{};
    pid_.reset();
    ctbr_ = CtbrCommand{};
  }
  if (rc.kill) for (float& cmd : out.cmd) cmd = 0.0f;  // belt and braces

  out.armed = armed_;
  out.mode = mode;
  out.active_faults = sup_.active_faults();
  out.fault_history = sup_.fault_history();
  out.first_fault = sup_.first_fault();
  out.arming_failures = arm_fail_;
  return out;
}

}  // namespace nf
