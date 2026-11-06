// Simplex runtime supervisor (PLAN.md section 5.1 items 2, 4, 5).
//
// Ladder (monotonic, never steps back on its own):
//   POLICY -> FALLBACK_HOLD -> CONTROLLED_DESCENT -> MOTOR_STOP
// Any monitor trip while in POLICY -> FALLBACK_HOLD (or straight to DESCENT
// for faults that make holding impossible: estimator / RC loss). Returning to
// POLICY needs an explicit request_reengage(), only accepted from
// FALLBACK_HOLD with all monitors currently clear. Kill or disarm overrides
// everything and forces MOTOR_STOP immediately.
#pragma once

#include <cstdint>

#include "nf/math.hpp"

namespace nf {

enum class FlightMode : uint8_t { DISARMED = 0, POLICY, FALLBACK_HOLD, CONTROLLED_DESCENT, MOTOR_STOP };
const char* to_string(FlightMode m);

enum Fault : uint32_t {
  FAULT_NONE = 0,
  FAULT_TILT = 1u << 0,
  FAULT_RATE = 1u << 1,
  FAULT_GEOFENCE = 1u << 2,
  FAULT_HEIGHT_FLOOR = 1u << 3,
  FAULT_POLICY_NONFINITE = 1u << 4,
  FAULT_POLICY_RANGE = 1u << 5,
  FAULT_POLICY_SATURATION = 1u << 6,
  FAULT_POLICY_OSCILLATION = 1u << 7,
  FAULT_DEADLINE = 1u << 8,
  FAULT_LOOP_JITTER = 1u << 9,
  FAULT_IMU_DISAGREE = 1u << 10,
  FAULT_ESTIMATOR = 1u << 11,
  FAULT_RC_LOSS = 1u << 12,
  FAULT_POLICY_MISSING = 1u << 13,
  FAULT_KILL = 1u << 14,
  FAULT_DESCENT_ENVELOPE = 1u << 15,  // envelope still violated in fallback -> escalate
  FAULT_COUNT_ = 16
};
const char* fault_name(uint32_t single_bit);

// Persistence of a boolean condition over time.
struct Debounce {
  uint64_t since_us = 0;
  bool on = false;
  // Returns true once cond has held continuously for >= hold_us.
  bool update(bool cond, uint64_t now_us, uint32_t hold_us) {
    if (!cond) { on = false; return false; }
    if (!on) { on = true; since_us = now_us; }
    return now_us - since_us >= hold_us;
  }
};

struct SupervisorConfig {
  // Attitude / rate envelope.
  float max_tilt_rad = 1.05f;          // 60 deg
  uint32_t tilt_hold_us = 50000;
  float max_rate = 12.0f;              // rad/s, any axis (above CTBR limit 10)
  uint32_t rate_hold_us = 30000;
  // Geofence (relative to arming origin) and height floor.
  float fence_radius_m = 3.0f;
  float fence_ceiling_m = 2.5f;
  uint32_t fence_hold_us = 20000;
  float height_floor_m = 0.25f;
  float height_floor_arm_margin_m = 0.15f;  // floor active once above floor+margin
  uint32_t height_floor_hold_us = 100000;
  // Policy output.
  float action_abs_max = 1.0f + 1e-4f;  // tanh output range
  uint32_t nonfinite_hold_us = 0;       // trip on first NaN/Inf
  uint32_t range_hold_us = 0;
  float saturation_level = 0.98f;       // |a_i| >= this counts as saturated
  uint32_t saturation_hold_us = 200000;
  int osc_window = 50;                  // samples (<= 64) of policy output history
  float osc_min_step = 0.15f;           // |delta a| that counts for a reversal
  int osc_max_reversals = 20;           // trip when reversals in window exceed this, any channel
  // Timing.
  uint32_t inference_deadline_us = 400;
  int deadline_misses_to_trip = 3;      // consecutive overruns
  uint32_t nominal_period_us = 1000;
  uint32_t max_jitter_us = 500;         // |period - nominal|
  int jitter_events_to_trip = 5;        // within jitter_window
  int jitter_window = 64;               // samples (<= 64)
  // IMU A/B.
  float imu_gyro_diff = 0.5f;           // rad/s, norm of difference
  float imu_accel_diff = 3.0f;          // m/s^2
  uint32_t imu_hold_us = 50000;
  // Estimator health.
  float est_max_accel_innov = 0.35f;    // rad (Mahony low-passed)
  float est_max_nis = 12.0f;            // KF low-passed NIS
  uint32_t est_hold_us = 200000;
  // RC link.
  uint32_t rc_timeout_us = 300000;
  // Escalation.
  uint32_t hold_escalate_us = 1500000;   // envelope (tilt/rate) violated this long in HOLD -> DESCENT
  uint32_t descent_abort_us = 1000000;   // envelope violated this long in DESCENT -> MOTOR_STOP
  float landed_height_m = 0.08f;
  float landed_vz = 0.15f;
  uint32_t landed_hold_us = 500000;
  uint32_t descent_timeout_us = 15000000;
};

struct SupervisorInput {
  uint64_t now_us = 0;
  bool armed = false;
  bool kill = false;
  bool rc_ok = true;              // a valid RC frame arrived this step
  // State estimate.
  Quat q{};
  Vec3 omega{};
  Vec3 pos{};                     // x,y relative to arming origin; z = height above ground
  Vec3 vel{};
  bool pos_valid = false;
  bool height_valid = false;
  float accel_innov = 0.0f;       // Mahony innovation (rad)
  float kf_nis = 0.0f;
  bool est_finite = true;
  // IMU A/B (same body frame).
  Vec3 gyro_a{}, gyro_b{}, acc_a{}, acc_b{};
  bool imu_b_valid = false;
  // Policy (only meaningful when policy_ran).
  bool policy_ran = false;
  float action[4] = {0, 0, 0, 0};
  uint32_t inference_us = 0;
  bool policy_error = false;      // model missing/invalid or inference failed
};

class Supervisor {
 public:
  void init(const SupervisorConfig& c);
  // Arms into POLICY (or FALLBACK_HOLD if !use_policy). Clears faults.
  void arm(uint64_t now_us, bool use_policy);
  void disarm();
  FlightMode update(const SupervisorInput& in);
  // Explicit operator action. Returns true if switched back to POLICY.
  bool request_reengage(uint64_t now_us);

  FlightMode mode() const { return mode_; }
  uint32_t active_faults() const { return active_; }   // monitors tripped in the last update
  uint32_t fault_history() const { return history_; }  // sticky since arm
  uint32_t first_fault() const { return first_; }      // single bit, 0 if none
  uint64_t first_fault_time_us() const { return first_t_; }
  const SupervisorConfig& config() const { return c_; }

 private:
  void raise(uint32_t bits, uint64_t now);
  void escalate(FlightMode m, uint64_t now);
  uint32_t evaluate(const SupervisorInput& in);
  SupervisorConfig c_{};
  FlightMode mode_ = FlightMode::DISARMED;
  uint32_t active_ = 0, history_ = 0, first_ = 0;
  uint64_t first_t_ = 0, mode_since_ = 0, last_now_ = 0, last_rc_us_ = 0;
  bool have_last_ = false, floor_armed_ = false;
  Debounce tilt_, rate_, fence_, floor_, nonfinite_, range_, sat_, imu_, est_, envelope_, landed_;
  int deadline_misses_ = 0;
  // Oscillation: per-channel bit history of reversal events.
  uint64_t osc_bits_[4] = {0, 0, 0, 0};
  float osc_prev_[4] = {0, 0, 0, 0};
  float osc_prev_delta_[4] = {0, 0, 0, 0};
  bool osc_have_prev_ = false;
  // Jitter: bit history of jitter events.
  uint64_t jitter_bits_ = 0;
};

}  // namespace nf
