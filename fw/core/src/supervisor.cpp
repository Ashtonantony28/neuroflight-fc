#include "nf/supervisor.hpp"

namespace nf {

namespace {
int popcount64(uint64_t v) {
  int n = 0;
  while (v) { v &= v - 1; ++n; }
  return n;
}
uint64_t window_mask(int n) {
  if (n >= 64) return ~0ull;
  if (n <= 0) return 0;
  return (1ull << n) - 1ull;
}
constexpr uint32_t kSevereFaults = FAULT_ESTIMATOR | FAULT_RC_LOSS;  // cannot hold -> descend
constexpr uint32_t kEnvelopeFaults = FAULT_TILT | FAULT_RATE;
}  // namespace

const char* to_string(FlightMode m) {
  switch (m) {
    case FlightMode::DISARMED: return "DISARMED";
    case FlightMode::POLICY: return "POLICY";
    case FlightMode::FALLBACK_HOLD: return "FALLBACK_HOLD";
    case FlightMode::CONTROLLED_DESCENT: return "CONTROLLED_DESCENT";
    case FlightMode::MOTOR_STOP: return "MOTOR_STOP";
  }
  return "?";
}

const char* fault_name(uint32_t b) {
  switch (b) {
    case FAULT_NONE: return "NONE";
    case FAULT_TILT: return "TILT";
    case FAULT_RATE: return "RATE";
    case FAULT_GEOFENCE: return "GEOFENCE";
    case FAULT_HEIGHT_FLOOR: return "HEIGHT_FLOOR";
    case FAULT_POLICY_NONFINITE: return "POLICY_NONFINITE";
    case FAULT_POLICY_RANGE: return "POLICY_RANGE";
    case FAULT_POLICY_SATURATION: return "POLICY_SATURATION";
    case FAULT_POLICY_OSCILLATION: return "POLICY_OSCILLATION";
    case FAULT_DEADLINE: return "DEADLINE";
    case FAULT_LOOP_JITTER: return "LOOP_JITTER";
    case FAULT_IMU_DISAGREE: return "IMU_DISAGREE";
    case FAULT_ESTIMATOR: return "ESTIMATOR";
    case FAULT_RC_LOSS: return "RC_LOSS";
    case FAULT_POLICY_MISSING: return "POLICY_MISSING";
    case FAULT_KILL: return "KILL";
    case FAULT_DESCENT_ENVELOPE: return "DESCENT_ENVELOPE";
    case FAULT_TILT_PREDICTED: return "TILT_PREDICTED";
    default: return "MULTIPLE/UNKNOWN";
  }
}

float tilt_rate(const Quat& q, const Vec3& w) {
  const Mat3 R = q.to_mat();
  const float r20 = R(2, 0), r21 = R(2, 1);
  const float s = std::sqrt(r20 * r20 + r21 * r21);  // sin(tilt)
  if (s < 1e-3f) return std::sqrt(w.x * w.x + w.y * w.y);
  return -(r20 * w.y - r21 * w.x) / s;
}

void Supervisor::init(const SupervisorConfig& c) {
  c_ = c;
  disarm();
}

void Supervisor::disarm() {
  mode_ = FlightMode::DISARMED;
}

void Supervisor::arm(uint64_t now, bool use_policy) {
  mode_ = use_policy ? FlightMode::POLICY : FlightMode::FALLBACK_HOLD;
  active_ = history_ = first_ = 0;
  first_t_ = 0;
  mode_since_ = now;
  last_rc_us_ = now;
  have_last_ = false;
  floor_armed_ = false;
  tilt_rate_ = pred_tilt_ = 0.0f;
  tilt_ = pred_ = rate_ = fence_ = floor_ = nonfinite_ = range_ = sat_ = imu_ = est_ = envelope_ = landed_ = Debounce{};
  deadline_misses_ = 0;
  for (int i = 0; i < 4; ++i) { osc_bits_[i] = 0; osc_prev_[i] = 0; osc_prev_delta_[i] = 0; }
  osc_have_prev_ = false;
  jitter_bits_ = 0;
}

void Supervisor::raise(uint32_t bits, uint64_t now) {
  if (!bits) return;
  history_ |= bits;
  if (!first_) {
    first_ = bits & (~bits + 1u);  // lowest set bit = highest-priority reason
    first_t_ = now;
  }
}

void Supervisor::escalate(FlightMode m, uint64_t now) {
  if (static_cast<uint8_t>(m) > static_cast<uint8_t>(mode_)) {
    mode_ = m;
    mode_since_ = now;
    envelope_ = Debounce{};
    landed_ = Debounce{};
  }
}

uint32_t Supervisor::evaluate(const SupervisorInput& in) {
  const uint64_t now = in.now_us;
  uint32_t f = 0;
  const bool in_policy = mode_ == FlightMode::POLICY;

  // --- envelope
  const float tilt = tilt_angle(in.q);
  if (tilt_.update(!in.q.finite() || tilt > c_.max_tilt_rad, now, c_.tilt_hold_us)) f |= FAULT_TILT;
  // Predicted tilt (policy only: in fallback the fallback IS the recovery).
  tilt_rate_ = in.q.finite() && finite(in.omega) ? tilt_rate(in.q, in.omega) : 0.0f;
  pred_tilt_ = tilt + (tilt_rate_ > 0.0f ? tilt_rate_ * c_.pred_horizon_s : 0.0f);
  const bool diverging = tilt_rate_ >= c_.pred_min_tilt_rate && pred_tilt_ > c_.pred_max_tilt_rad;
  if (pred_.update(in_policy && diverging, now, c_.pred_hold_us)) f |= FAULT_TILT_PREDICTED;
  float rmax = std::fmax(std::fabs(in.omega.x), std::fmax(std::fabs(in.omega.y), std::fabs(in.omega.z)));
  if (rate_.update(!finite(in.omega) || rmax > c_.max_rate, now, c_.rate_hold_us)) f |= FAULT_RATE;

  // --- geofence (horizontal needs pos_valid, ceiling needs height_valid)
  bool out_xy = in.pos_valid && std::sqrt(sq(in.pos.x) + sq(in.pos.y)) > c_.fence_radius_m;
  bool out_z = in.height_valid && in.pos.z > c_.fence_ceiling_m;
  if (fence_.update(out_xy || out_z, now, c_.fence_hold_us)) f |= FAULT_GEOFENCE;

  // --- IMU A/B
  bool imu_bad = in.imu_b_valid &&
                 (!(norm(in.gyro_a - in.gyro_b) <= c_.imu_gyro_diff) || !(norm(in.acc_a - in.acc_b) <= c_.imu_accel_diff));
  if (imu_.update(imu_bad, now, c_.imu_hold_us)) f |= FAULT_IMU_DISAGREE;

  // --- estimator health
  if (!in.est_finite) {
    f |= FAULT_ESTIMATOR;
  } else {
    bool bad = !(in.accel_innov <= c_.est_max_accel_innov) || !(in.kf_nis <= c_.est_max_nis);
    if (est_.update(bad, now, c_.est_hold_us)) f |= FAULT_ESTIMATOR;
  }

  // --- RC link
  if (in.rc_ok) last_rc_us_ = now;
  if (now - last_rc_us_ > c_.rc_timeout_us) f |= FAULT_RC_LOSS;

  // --- loop jitter (period between supervisor updates)
  if (have_last_) {
    uint64_t period = now - last_now_;
    uint64_t nom = c_.nominal_period_us;
    uint64_t dev = period > nom ? period - nom : nom - period;
    jitter_bits_ = ((jitter_bits_ << 1) | (dev > c_.max_jitter_us ? 1u : 0u)) & window_mask(c_.jitter_window);
  }
  if (in_policy && popcount64(jitter_bits_) >= c_.jitter_events_to_trip) f |= FAULT_LOOP_JITTER;

  // --- height floor (policy only: fallback descends through it on purpose)
  if (in.height_valid && in.pos.z > c_.height_floor_m + c_.height_floor_arm_margin_m) floor_armed_ = true;
  if (floor_.update(in_policy && floor_armed_ && in.height_valid && in.pos.z < c_.height_floor_m, now,
                    c_.height_floor_hold_us))
    f |= FAULT_HEIGHT_FLOOR;

  // --- policy output monitors
  if (in_policy && in.policy_error) f |= FAULT_POLICY_MISSING;
  if (in_policy && in.policy_ran) {
    bool nonfinite = false, out_of_range = false, saturated = false;
    for (int i = 0; i < 4; ++i) {
      float a = in.action[i];
      if (!is_finite(a)) { nonfinite = true; continue; }
      if (std::fabs(a) > c_.action_abs_max) out_of_range = true;
      if (std::fabs(a) >= c_.saturation_level) saturated = true;
    }
    if (nonfinite_.update(nonfinite, now, c_.nonfinite_hold_us)) f |= FAULT_POLICY_NONFINITE;
    if (range_.update(out_of_range, now, c_.range_hold_us)) f |= FAULT_POLICY_RANGE;
    if (sat_.update(saturated, now, c_.saturation_hold_us)) f |= FAULT_POLICY_SATURATION;

    // Oscillation: count sign reversals of significant output steps.
    const uint64_t mask = window_mask(c_.osc_window);
    for (int i = 0; i < 4; ++i) {
      float a = is_finite(in.action[i]) ? in.action[i] : 0.0f;
      uint64_t ev = 0;
      if (osc_have_prev_) {
        float d = a - osc_prev_[i];
        if (std::fabs(d) >= c_.osc_min_step) {
          if (osc_prev_delta_[i] * d < 0.0f) ev = 1;
          osc_prev_delta_[i] = d;
        }
      }
      osc_prev_[i] = a;
      osc_bits_[i] = ((osc_bits_[i] << 1) | ev) & mask;
      if (popcount64(osc_bits_[i]) > c_.osc_max_reversals) f |= FAULT_POLICY_OSCILLATION;
    }
    osc_have_prev_ = true;

    deadline_misses_ = in.inference_us > c_.inference_deadline_us ? deadline_misses_ + 1 : 0;
    if (deadline_misses_ >= c_.deadline_misses_to_trip) f |= FAULT_DEADLINE;
  }
  return f;
}

FlightMode Supervisor::update(const SupervisorInput& in) {
  const uint64_t now = in.now_us;
  // Kill and disarm override everything, in every mode.
  if (in.kill) {
    raise(FAULT_KILL, now);
    active_ |= FAULT_KILL;
    mode_ = FlightMode::MOTOR_STOP;
    return mode_;
  }
  if (!in.armed) {
    mode_ = FlightMode::DISARMED;
    return mode_;
  }
  if (mode_ == FlightMode::DISARMED || mode_ == FlightMode::MOTOR_STOP) return mode_;

  active_ = evaluate(in);
  last_now_ = now;
  have_last_ = true;
  raise(active_, now);

  switch (mode_) {
    case FlightMode::POLICY:
      if (active_ & kSevereFaults) escalate(FlightMode::CONTROLLED_DESCENT, now);
      else if (active_) escalate(FlightMode::FALLBACK_HOLD, now);
      break;
    case FlightMode::FALLBACK_HOLD:
      if (active_ & kSevereFaults) {
        escalate(FlightMode::CONTROLLED_DESCENT, now);
      } else if (envelope_.update((active_ & kEnvelopeFaults) != 0, now, c_.hold_escalate_us)) {
        raise(FAULT_DESCENT_ENVELOPE, now);
        escalate(FlightMode::CONTROLLED_DESCENT, now);
      }
      break;
    case FlightMode::CONTROLLED_DESCENT: {
      bool landed = in.height_valid && in.pos.z < c_.landed_height_m && std::fabs(in.vel.z) < c_.landed_vz;
      if (landed_.update(landed, now, c_.landed_hold_us)) {
        escalate(FlightMode::MOTOR_STOP, now);
      } else if (envelope_.update((active_ & kEnvelopeFaults) != 0, now, c_.descent_abort_us)) {
        raise(FAULT_DESCENT_ENVELOPE, now);
        escalate(FlightMode::MOTOR_STOP, now);
      } else if (now - mode_since_ > c_.descent_timeout_us) {
        escalate(FlightMode::MOTOR_STOP, now);
      }
      break;
    }
    default:
      break;
  }
  return mode_;
}

bool Supervisor::request_reengage(uint64_t now) {
  if (mode_ != FlightMode::FALLBACK_HOLD || active_ != 0) return false;
  mode_ = FlightMode::POLICY;
  mode_since_ = now;
  nonfinite_ = range_ = sat_ = floor_ = envelope_ = pred_ = Debounce{};
  deadline_misses_ = 0;
  for (int i = 0; i < 4; ++i) { osc_bits_[i] = 0; osc_prev_delta_[i] = 0; }
  osc_have_prev_ = false;
  jitter_bits_ = 0;
  return true;
}

}  // namespace nf
