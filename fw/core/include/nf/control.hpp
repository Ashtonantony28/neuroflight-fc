// Classical control: rate PID, attitude P, CTBR mixer, cascaded fallback.
// Torques are in N*m, thrust in N, rates in rad/s (INTERFACES.md sections 1-2).
#pragma once

#include "nf/filters.hpp"
#include "nf/math.hpp"
#include "nf/vehicle_params.hpp"

namespace nf {

// Collective thrust + body-rate command (policy CTBR action and fallback output).
struct CtbrCommand {
  float thrust_n = 0.0f;  // total collective thrust (N)
  Vec3 rate_sp{};         // body-rate setpoint (rad/s)
};

// --------------------------------------------------------------------------
// Mixer: (F, tau) -> per-motor thrust -> motor command via inverse thrust curve.
struct MixerParams {
  float arm_xy = vehicle::kArmXy;
  float yaw_k = vehicle::kYawTorqueCoeff;
  float thrust_max = vehicle::kThrustMax;
  float curve_a = vehicle::kThrustCurveA;
  float idle_command = vehicle::kIdleCommand;
};

struct MixerResult {
  float cmd[4] = {0, 0, 0, 0};     // motor commands in [idle, 1]
  float thrust[4] = {0, 0, 0, 0};  // allocated per-motor thrust (N), after saturation handling
  bool rp_saturated = false;       // roll/pitch had to be scaled down
  bool yaw_saturated = false;      // yaw was reduced to protect roll/pitch
  bool thrust_saturated = false;   // collective moved away from request (> 1e-6 N)
  float rp_scale = 1.0f, yaw_scale = 1.0f;
  // Same definition as sim/ctbr.py: scale_rp < 1 | s_y < 1 | collective shifted.
  bool any_saturated() const { return rp_saturated || yaw_saturated || thrust_saturated; }
};

class Mixer {
 public:
  void init(const MixerParams& p);
  // INTERFACES.md section 9 / sim/ctbr.py::Mixer, in this order:
  // (1) roll/pitch differential thrusts scaled down uniformly to fit
  //     [T_idle, thrust_max]; (2) collective shifted to make room;
  // (3) yaw added with the largest s in [0,1] that stays within limits.
  MixerResult mix(float thrust_n, const Vec3& torque) const;
  float thrust_to_cmd(float t) const;  // steady-state inverse of the thrust curve
  float cmd_to_thrust(float c) const;
  float min_thrust() const { return t_min_; }
  float max_thrust() const { return p_.thrust_max; }
  // Per-motor body torque from per-motor thrusts (INTERFACES.md section 2).
  Vec3 torque_from_thrusts(const float t[4]) const;

 private:
  MixerParams p_{};
  float t_min_ = 0.0f;
};

// --------------------------------------------------------------------------
// Rate PID (per axis), INTERFACES.md section 9 / sim/ctbr.py::RateController:
//   e = w_sp - w
//   d_filt += alpha * (-(w - w_prev)/dt - d_filt),  alpha = 1 - exp(-2 pi f_d dt)
//   tau = J*(kp*e + kd*d_filt + ki*integ)     (integ from the previous step)
//   after mixing: if !mixer_saturated: integ = clamp(integ + e*dt, +-i_limit)
// dt is the fixed loop period (1/sample_hz), exactly as the sim does.
// Defaults equal sim/ctbr.py::RatePIDGains. I-term relax and output limit are
// firmware-only extras and are DISABLED by default to keep sim parity.
struct RatePidConfig {
  Vec3 kp{35.0f, 35.0f, 20.0f};  // 1/s      (angular-acceleration gains, x J)
  Vec3 ki{60.0f, 60.0f, 20.0f};  // 1/s^2
  Vec3 kd{0.7f, 0.7f, 0.0f};     // dimensionless
  float d_lpf_hz = 20.0f;
  Vec3 i_limit{8.0f, 8.0f, 8.0f};  // |integ| limit (integral of rate error, rad)
  float out_limit = 0.0f;          // rad/s^2 per axis, 0 = disabled
  float relax_lpf_hz = 15.0f;      // I-term relax setpoint high-pass cutoff
  float relax_threshold = 0.0f;    // rad/s; 0 = relax disabled
  Vec3 inertia{vehicle::kInertia[0], vehicle::kInertia[1], vehicle::kInertia[2]};
};

class RatePid {
 public:
  void init(const RatePidConfig& c, float sample_hz);
  void reset();
  // Step 1: torque demand (N*m) using the current integrator.
  Vec3 update(const Vec3& rate_sp, const Vec3& rate);
  // Step 2 (after the mixer): integrate unless the mixer saturated.
  void integrate(bool mixer_saturated);
  const Vec3& integrator() const { return i_; }
  float relax_factor(int axis) const { return relax_[axis]; }
  float d_alpha() const { return d_alpha_; }

 private:
  RatePidConfig c_{};
  Vec3 i_{}, prev_rate_{}, d_filt_{}, e_{};
  float relax_[3] = {1, 1, 1};
  float dt_ = 0.001f, d_alpha_ = 1.0f;
  bool first_ = true;
  Pt1 sp_lpf_[3];
};

// --------------------------------------------------------------------------
// Attitude P controller: quaternion error -> body-rate setpoint.
struct AttitudeConfig {
  Vec3 kp{8.0f, 8.0f, 4.0f};  // 1/s
  Vec3 rate_limit{vehicle::kMaxBodyRate[0], vehicle::kMaxBodyRate[1], vehicle::kMaxBodyRate[2]};
};
Vec3 attitude_p(const Quat& q, const Quat& q_sp, const AttitudeConfig& c);

// --------------------------------------------------------------------------
// Cascaded fallback: position P -> velocity PI -> thrust vector/attitude -> rate.
struct CascadeConfig {
  Vec3 pos_kp{1.2f, 1.2f, 1.5f};
  Vec3 vel_kp{2.5f, 2.5f, 4.0f};
  Vec3 vel_ki{0.4f, 0.4f, 1.5f};
  Vec3 vel_i_limit{1.0f, 1.0f, 3.0f};   // m/s^2
  Vec3 vel_limit{1.0f, 1.0f, 0.8f};     // m/s
  float max_tilt = 0.6f;                // rad
  float max_acc_z = 6.0f;               // m/s^2
  float min_acc_z = -6.0f;
  float descent_rate = 0.4f;            // m/s for controlled descent
  float mass = vehicle::kMass;
  float gravity = vehicle::kGravity;
  AttitudeConfig att{};
};

struct VehicleState {
  Quat q{};
  Vec3 omega{};   // body rates (bias-corrected, filtered)
  Vec3 pos{};     // W
  Vec3 vel{};     // W
  bool pos_valid = false;  // horizontal position/velocity usable
  bool height_valid = false;
};

enum class CascadeMode : uint8_t { HOLD, DESCENT, LEVEL };

class CascadeController {
 public:
  void init(const CascadeConfig& c);
  void reset();
  // HOLD: hold pos_sp/yaw_sp. DESCENT: hold xy, descend at descent_rate.
  // LEVEL: no position feedback (attitude level, thrust = hover - descent);
  //        used automatically when pos/height are invalid.
  CtbrCommand update(const VehicleState& s, const Vec3& pos_sp, float yaw_sp, CascadeMode mode, float dt);
  const Quat& q_sp() const { return q_sp_; }
  // Bumpless handover (FOLLOWUPS #6): reset, then preload the vertical
  // velocity integrator so that, level and at zero velocity error, the
  // commanded collective equals hover_thrust_n (the estimated thrust that
  // currently holds the vehicle up). Horizontal integrators start at 0.
  void seed_hover_thrust(float hover_thrust_n);
  const Vec3& vel_integrator() const { return vi_; }

 private:
  CascadeConfig c_{};
  Vec3 vi_{};
  Quat q_sp_{};
};

}  // namespace nf
