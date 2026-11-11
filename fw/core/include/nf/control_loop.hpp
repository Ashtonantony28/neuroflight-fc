// Top-level flight core: filters -> estimator -> {policy | fallback} ->
// supervisor -> rate PID -> mixer. This is what the SITL/HIL bridge calls.
#pragma once

#include <cstdint>

#include "nf/control.hpp"
#include "nf/estimator.hpp"
#include "nf/filters.hpp"
#include "nf/policy.hpp"
#include "nf/supervisor.hpp"
#include "nf/vehicle_params.hpp"

namespace nf {

// ------------------------------------------------------------------ inputs
struct ImuSample {
  Vec3 gyro{};    // IMU A, rad/s, body FLU
  Vec3 accel{};   // IMU A, specific force m/s^2, body FLU (level at rest: +g z)
  Vec3 gyro_b{};  // IMU B (already rotated into the body frame)
  Vec3 accel_b{};
  bool b_valid = false;
};

struct SensorSample {
  FlowSample flow{};    // valid only on steps where a new sample arrived
  RangeSample range{};  // idem
  float motor_hz[4] = {0, 0, 0, 0};  // motor rotation frequency (bidir DShot)
  bool rpm_valid = false;
  float vbat = 0.0f;
  bool vbat_valid = false;
};

struct RcInput {
  bool link_ok = true;        // a fresh, valid frame this step
  bool arm = false;           // arm switch (level); rising edge requests arming
  bool kill = false;          // hardware-priority kill, overrides everything
  bool policy_enable = true;  // arm into POLICY (true) or FALLBACK_HOLD (false)
  bool reengage = false;      // momentary; rising edge requests FALLBACK_HOLD -> POLICY
  float throttle = 0.0f;      // [0,1], only used by the arming check
};

struct MotorOutput {
  float cmd[4] = {0, 0, 0, 0};  // INTERFACES.md section 2 order, [0,1]
  bool armed = false;
  FlightMode mode = FlightMode::DISARMED;
  uint32_t active_faults = 0;
  uint32_t fault_history = 0;
  uint32_t first_fault = 0;
  uint32_t arming_failures = 0;  // last arming attempt, ArmFail bits
};

// ------------------------------------------------------------------ arming
enum ArmFail : uint32_t {
  ARM_OK = 0,
  ARM_FAIL_IMU_DISAGREE = 1u << 0,
  ARM_FAIL_IMU_B_MISSING = 1u << 1,
  ARM_FAIL_ESTIMATOR = 1u << 2,
  ARM_FAIL_VBAT = 1u << 3,
  ARM_FAIL_POLICY = 1u << 4,  // missing / invalid / CRC mismatch
  ARM_FAIL_THROTTLE = 1u << 5,
  ARM_FAIL_FAULTS = 1u << 6,
  ARM_FAIL_RC_LINK = 1u << 7,
  ARM_FAIL_KILL = 1u << 8,
  ARM_FAIL_TILT = 1u << 9,
  ARM_FAIL_GYRO_CAL = 1u << 10,  // at-rest gyro calibration not finished
  ARM_FAIL_MOTION = 1u << 11,    // motion detected while disarmed; calibration restarted
};

struct ArmingConfig {
  float min_vbat = 13.2f;        // 4S, 3.3 V/cell
  bool require_vbat = false;     // SITL has no battery model by default
  bool require_imu_b = false;
  // IMU A/B agreement (FOLLOWUPS #4) is judged on the low-passed difference
  // (gyro_a - calibrated bias_a) - gyro_b and accel_a - accel_b, not on single
  // raw samples. The filter must have run imu_diff_min_s before it counts.
  float max_gyro_diff = 0.1f;    // rad/s at rest (norm of filtered difference)
  float max_accel_diff = 1.0f;   // m/s^2
  float imu_diff_lpf_hz = 1.0f;
  float imu_diff_min_s = 0.5f;
  bool require_gyro_cal = true;  // FOLLOWUPS #3
  float max_throttle = 0.05f;
  float max_tilt_rad = 0.5f;
  float est_max_innov = 0.05f;
  float est_min_time_s = 0.5f;
};

struct ArmingInput {
  bool imu_b_valid = false;
  Vec3 gyro_ab_diff{};       // filtered (see ArmingConfig)
  Vec3 accel_ab_diff{};
  bool imu_ab_ready = false; // filter warmed up
  bool gyro_cal_done = false;
  bool gyro_motion = false;
  bool est_converged = false;
  float tilt_rad = 0.0f;
  float vbat = 0.0f;
  bool vbat_valid = false;
  bool want_policy = false;
  bool policy_ok = false;  // model present, validates, CRC matches
  float throttle = 0.0f;
  bool rc_link_ok = false;
  bool kill = false;
  uint32_t fault_flags = 0;  // faults that must be cleared before arming
};

uint32_t arming_checks(const ArmingInput& in, const ArmingConfig& c);

// ------------------------------------------------------------------ loop
struct ControlLoopConfig {
  int rate_hz = vehicle::kRateLoopHz;        // step() call rate
  int control_hz = vehicle::kControlRateHz;  // policy / outer-loop rate
  float gyro_lpf_hz = 150.0f;
  float accel_lpf_hz = 40.0f;
  bool rpm_filter = true;
  float hold_height_m = 1.0f;                // default target z
  float min_thrust_for_i = 0.3f;             // * weight; below -> rate I reset (on ground)
  // Hover-thrust estimate (bumpless fallback handover, FOLLOWUPS #6): low-pass
  // of g * T_alloc / f_bz while flying near level above hover_est_min_height_m.
  bool bumpless_handover = true;  // false: old behaviour (fallback integrators start at 0)
  float hover_est_lpf_hz = 0.5f;
  float hover_est_min_height_m = 0.15f;
  float hover_est_max_tilt = 0.35f;
  // Mahony specific-force reference (FOLLOWUPS 11): while armed with a valid
  // flow/ToF estimate, the accel is compared with KF acceleration + g
  // (low-passed derivative of the KF velocity, horizontal part clamped)
  // instead of gravity alone, so hard braking does not pull the attitude off.
  // false: plain gravity reference (behaviour up to hover_ctbr_v2).
  bool mahony_kf_ref = true;
  float mahony_ref_lpf_hz = 5.0f;
  float mahony_ref_max_acc = 6.0f;  // m/s^2, horizontal clamp (~31 deg of steady tilt)
  // Soft deadband (m/s^2) on the low-passed estimate (horizontal magnitude and z
  // separately): hover noise in the KF velocity derivative stays out of the
  // attitude (it cost fallback hover precision), braking accelerations pass.
  float mahony_ref_deadband = 0.3f;
  GyroCalibrator::Config gyro_cal{};
  Mahony::Config mahony{};
  TranslationKf::Config kf{};
  RatePidConfig rate{};
  CascadeConfig cascade{};
  MixerParams mixer{};
  SupervisorConfig sup{};
  ArmingConfig arming{};
  // Optional monotonic microsecond clock used to time policy inference
  // (DWT cycle counter on target). nullptr -> inference time reported as 0.
  uint32_t (*clock_us)() = nullptr;
};

// Diagnostic input override (SITL / HIL sim-to-real gap analysis only; leave
// unset in flight). Bits in `mask` replace what the POLICY observes (attitude,
// rates, position, velocity) and/or the rate measurement fed to the rate PID,
// after the estimator has run. The supervisor, fallback and logs keep using
// the real estimate. Used by fw/sitl/diagnose_gap.py.
struct InputOverride {
  enum : uint32_t { POLICY_Q = 1u, POLICY_OMEGA = 2u, POLICY_POS = 4u, POLICY_VEL = 8u, PID_OMEGA = 16u };
  uint32_t mask = 0;
  Quat q{};
  Vec3 omega_policy{}, omega_pid{}, pos{}, vel{};
};

class ControlLoop {
 public:
  // No heap: all state lives in this object. Call once.
  void init(const ControlLoopConfig& c);
  void set_policy(const FloatModel* m);
  void set_policy(const Int8Model* m);
  // Target for the policy and for FALLBACK_HOLD (until a fault latches the hold point).
  void set_target(const Vec3& pos, float yaw);
  // Diagnostics only (see InputOverride). nullptr disables (default).
  void set_input_override(const InputOverride* o) { ovr_ = o; }
  MotorOutput step(const ImuSample& imu, const SensorSample& sensors, const RcInput& rc, uint64_t now_us);

  // Introspection (logging / tests).
  const VehicleState& state() const { return state_; }
  const Supervisor& supervisor() const { return sup_; }
  const Mahony& attitude() const { return mahony_; }
  const TranslationKf& translation() const { return kf_; }
  const CtbrCommand& last_ctbr() const { return ctbr_; }
  const float* last_action() const { return action_; }
  uint32_t last_inference_us() const { return inference_us_; }
  bool policy_ok() const;
  bool armed() const { return armed_; }
  const GyroCalibrator& gyro_cal() const { return cal_; }
  float hover_thrust_estimate() const { return hover_thrust_.value(); }
  // Low-passed KF acceleration estimate (W) used as the Mahony reference.
  Vec3 accel_ref() const;
  Vec3 imu_gyro_diff() const { return {gdiff_[0].value(), gdiff_[1].value(), gdiff_[2].value()}; }
  const CascadeController& cascade() const { return cascade_; }

 private:
  void do_arm(uint64_t now, bool use_policy);
  void latch_hold_point();
  ControlLoopConfig c_{};
  BiquadLpfVec3 gyro_lpf_{};
  Pt1Vec3 acc_lpf_{};
  RpmNotchBank rpm_{};
  GyroCalibrator cal_{};
  Pt1 gdiff_[3], adiff_[3];
  int imu_diff_n_ = 0, imu_diff_min_n_ = 500;
  Pt1 hover_thrust_{};
  Pt1 acc_ref_[3];
  Vec3 prev_kf_vel_{};
  Mahony mahony_{};
  TranslationKf kf_{};
  RatePid pid_{};
  CascadeController cascade_{};
  Mixer mixer_{};
  Supervisor sup_{};
  PolicyRunner policy_{};
  VehicleState state_{};
  CtbrCommand ctbr_{};
  MixerResult mix_{};
  Vec3 target_{0, 0, 1};
  float target_yaw_ = 0.0f;
  Vec3 hold_sp_{};
  float hold_yaw_ = 0.0f;
  float action_[kActDim] = {0, 0, 0, 0};
  uint64_t last_us_ = 0, step_count_ = 0;
  uint32_t inference_us_ = 0, arm_fail_ = 0;
  int ctrl_div_ = 2;
  bool first_ = true, armed_ = false, prev_arm_sw_ = false, prev_reengage_ = false;
  bool policy_crc_ok_ = false;
  FlightMode prev_mode_ = FlightMode::DISARMED;
  const InputOverride* ovr_ = nullptr;
};

}  // namespace nf
