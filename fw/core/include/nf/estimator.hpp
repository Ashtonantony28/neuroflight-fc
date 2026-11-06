// State estimation: Mahony attitude filter and a flow+ToF velocity/height KF.
#pragma once

#include "nf/math.hpp"

namespace nf {

// Mahony complementary filter (gyro + accel). Accel is specific force in B
// (at rest, level: [0,0,+g]). Yaw is unobservable (no magnetometer) and
// integrates gyro only.
class Mahony {
 public:
  struct Config {
    float kp = 1.0f;           // accel correction gain (rad/s per rad)
    float ki = 0.05f;          // gyro bias integral gain
    float kp_init = 10.0f;     // boosted gain during init_time_s (fast convergence)
    float init_time_s = 1.0f;
    float accel_min_g = 0.5f;  // accel correction gated to |a| in [min,max]*g
    float accel_max_g = 1.5f;
    float max_rate_for_accel = 2.0f;  // rad/s; above this the accel is not trusted
                                      // (a quad's accel sees thrust, not gravity, in manoeuvres)
    float bias_limit = 0.2f;   // rad/s
    float gravity = 9.81f;
    float innov_lpf_hz = 1.0f; // low-pass on accel-vs-estimate angle (health)
  };
  void init(const Config& c);
  void reset(const Quat& q = Quat{});
  // Level the attitude from one accel sample (roll/pitch, yaw = 0).
  void init_from_accel(const Vec3& accel);
  void update(const Vec3& gyro, const Vec3& accel, float dt);

  const Quat& q() const { return q_; }
  const Vec3& bias() const { return bias_; }  // estimated gyro bias (rad/s)
  Vec3 rates(const Vec3& gyro) const { return gyro - bias_; }
  // Low-passed angle (rad) between measured and predicted gravity direction,
  // updated only on samples that pass the accel gates (|a| and |omega|).
  float innovation() const { return innov_; }
  float time_s() const { return t_; }
  bool converged(float max_innov_rad = 0.05f, float min_time_s = 0.5f) const {
    return t_ >= min_time_s && innov_ <= max_innov_rad && q_.finite();
  }

 private:
  Config c_{};
  Quat q_{};
  Vec3 bias_{};
  float innov_ = 0.0f, t_ = 0.0f;
};

// Inputs for the translational estimator.
struct FlowSample {
  // Body-frame ego-motion flow (rad/s). Convention the driver must produce:
  //   flow_x = v_Bx / h - omega_y,  flow_y = v_By / h + omega_x
  // (h = height above ground, level). quality in [0,1].
  float flow_x = 0.0f, flow_y = 0.0f, quality = 0.0f;
  bool valid = false;
};
struct RangeSample {
  float range_m = 0.0f;  // distance along -z_B to ground
  bool valid = false;
};

// Three decoupled 2-state [p, v] Kalman filters (world x, y, z), driven by
// the world-frame acceleration from IMU+attitude. z is corrected by ToF
// height, x/y by flow-derived velocity. x/y position is dead-reckoned
// (relative to reset point).
class TranslationKf {
 public:
  struct Config {
    float accel_noise = 0.5f;        // m/s^2 (process noise, continuous white accel)
    float flow_vel_noise = 0.1f;     // m/s
    float range_noise = 0.02f;       // m
    float min_quality = 0.3f;
    float min_height_flow = 0.08f;   // m, below this flow is not used
    float max_tilt_range = 0.7f;     // rad, ToF ignored beyond
    float max_range = 4.0f;          // m
    float nis_gate = 25.0f;          // reject measurements whose NIS exceeds this
    float nis_lpf_hz = 2.0f;
    float gravity = 9.81f;
  };
  void init(const Config& c);
  void reset(const Vec3& p = Vec3{}, const Vec3& v = Vec3{});
  // accel_b: specific force (m/s^2) in B; q: attitude; omega: body rates.
  void predict(const Vec3& accel_b, const Quat& q, float dt);
  void correct(const FlowSample& flow, const RangeSample& range, const Quat& q, const Vec3& omega);

  Vec3 pos() const { return {x_[0].p, x_[1].p, x_[2].p}; }
  Vec3 vel() const { return {x_[0].v, x_[1].v, x_[2].v}; }
  bool height_valid() const { return height_age_ < 0.2f; }
  bool flow_valid() const { return flow_age_ < 0.2f; }
  // Low-passed normalized innovation squared (mean over used measurements).
  float nis() const { return nis_lpf_; }
  uint32_t rejected() const { return rejected_; }
  bool healthy() const;

 private:
  struct Axis {
    float p = 0, v = 0;
    float P[2][2] = {{1, 0}, {0, 1}};
  };
  void predict_axis(Axis& a, float acc, float dt);
  bool update_axis(Axis& a, int idx, float z, float r2);  // idx 0=pos,1=vel
  Config c_{};
  Axis x_[3];
  float nis_lpf_ = 0.0f, dt_last_ = 0.001f;
  float height_age_ = 1e3f, flow_age_ = 1e3f;
  uint32_t rejected_ = 0;
};

}  // namespace nf
