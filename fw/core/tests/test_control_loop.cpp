// Closed-loop tests: ControlLoop driving the test quad simulator.
#include "model_fixture.hpp"
#include "nf/control_loop.hpp"
#include "nftest.hpp"
#include "quad_sim.hpp"

#include <cstdlib>

using namespace nf;

namespace {
constexpr uint64_t kDtUs = 1000;
constexpr float kDt = 0.001f;

// Constant-output "policy": all weights zero, last-layer bias = atanh(target).
struct ConstPolicy {
  float mean[kObsDim] = {}, stdv[kObsDim];
  float W0[16 * kObsDim] = {}, b0[16] = {};
  float W1[kActDim * 16] = {}, b1[kActDim] = {};
  FloatLayer L[2];
  FloatModel m{};
  explicit ConstPolicy(const float a[kActDim]) {
    for (float& s : stdv) s = 1.0f;
    for (int i = 0; i < kActDim; ++i) b1[i] = std::atanh(a[i]);
    L[0] = {kObsDim, 16, W0, b0, Act::Tanh};
    L[1] = {16, kActDim, W1, b1, Act::Tanh};
    m = {kObsDim, kActDim, 2, mean, stdv, L, 0};
    m.crc32 = policy_crc32(m);
  }
};
const float kHoverA0 = 2.0f * vehicle::kMass * vehicle::kGravity / (4.0f * vehicle::kThrustMax) - 1.0f;

struct Rig {
  nftest::QuadSim sim;
  ControlLoop loop;
  RcInput rc;
  uint64_t t = 0;
  MotorOutput out;
  bool imu_b_fault = false;
  Vec3 gyro_bias{};          // IMU A gyro bias (rad/s)
  Vec3 b_offset{};           // IMU B constant gyro offset
  float b_gyro_noise = 0.0f, b_acc_noise = 0.0f;  // independent IMU B noise
  uint32_t brng = 99;
  float bn() {
    float s = 0;
    for (int i = 0; i < 4; ++i) { brng = brng * 1664525u + 1013904223u; s += static_cast<float>(brng >> 8) / 16777216.0f - 0.5f; }
    return s * 1.732f;
  }
  explicit Rig(const ControlLoopConfig& c = ControlLoopConfig{}) { loop.init(c); }
  // Disarmed warm-up with the vehicle held still (sim not integrated).
  void warmup(int ms) {
    rc.arm = false;
    for (int i = 0; i < ms; ++i) tick(false);
  }
  void tick(bool integrate = true) {
    ImuSample imu = sim.imu();
    if (b_gyro_noise > 0.0f || b_acc_noise > 0.0f) {  // IMU B gets its own noise
      Vec3 f_w = sim.last_acc_w + Vec3{0, 0, vehicle::kGravity};
      imu.gyro_b = sim.omega + Vec3{bn(), bn(), bn()} * b_gyro_noise;
      imu.accel_b = sim.q.rotate_inv(f_w) + Vec3{bn(), bn(), bn()} * b_acc_noise;
    }
    imu.gyro += gyro_bias;
    imu.gyro_b += b_offset;
    if (imu_b_fault) imu.gyro_b.x += 1.5f;
    SensorSample s = sim.sensors((t / kDtUs) % 10 == 0);  // flow/ToF at 100 Hz
    out = loop.step(imu, s, rc, t);
    if (integrate) sim.step(out.cmd, kDt);
    t += kDtUs;
  }
  void run(int ms) { for (int i = 0; i < ms; ++i) tick(); }
  void arm(bool policy) {
    rc.policy_enable = policy;
    rc.arm = true;
    tick();
  }
};
}  // namespace

TEST(arming_checks_each_reason) {
  ArmingConfig c;
  ArmingInput ok;
  ok.est_converged = true;
  ok.rc_link_ok = true;
  ok.imu_b_valid = true;
  ok.imu_ab_ready = true;
  ok.gyro_cal_done = true;
  CHECK_EQ(arming_checks(ok, c), 0);
  struct { void (*f)(ArmingInput&); uint32_t bit; } cases[] = {
      {[](ArmingInput& a) { a.kill = true; }, ARM_FAIL_KILL},
      {[](ArmingInput& a) { a.rc_link_ok = false; }, ARM_FAIL_RC_LINK},
      {[](ArmingInput& a) { a.throttle = 0.3f; }, ARM_FAIL_THROTTLE},
      {[](ArmingInput& a) { a.gyro_ab_diff.x = 0.15f; }, ARM_FAIL_IMU_DISAGREE},
      {[](ArmingInput& a) { a.accel_ab_diff.z = 1.5f; }, ARM_FAIL_IMU_DISAGREE},
      {[](ArmingInput& a) { a.gyro_ab_diff.y = NAN; }, ARM_FAIL_IMU_DISAGREE},
      {[](ArmingInput& a) { a.imu_ab_ready = false; }, ARM_FAIL_IMU_DISAGREE},
      {[](ArmingInput& a) { a.gyro_cal_done = false; }, ARM_FAIL_GYRO_CAL},
      {[](ArmingInput& a) { a.gyro_cal_done = false; a.gyro_motion = true; }, ARM_FAIL_GYRO_CAL | ARM_FAIL_MOTION},
      {[](ArmingInput& a) { a.est_converged = false; }, ARM_FAIL_ESTIMATOR},
      {[](ArmingInput& a) { a.tilt_rad = 0.8f; }, ARM_FAIL_TILT},
      {[](ArmingInput& a) { a.vbat_valid = true; a.vbat = 12.0f; }, ARM_FAIL_VBAT},
      {[](ArmingInput& a) { a.want_policy = true; a.policy_ok = false; }, ARM_FAIL_POLICY},
      {[](ArmingInput& a) { a.fault_flags = 4; }, ARM_FAIL_FAULTS},
  };
  for (auto& k : cases) {
    ArmingInput a = ok;
    k.f(a);
    CHECK_EQ(arming_checks(a, c), k.bit);
  }
  ArmingConfig strict = c;
  strict.require_imu_b = true;
  strict.require_vbat = true;
  ArmingInput a = ok;
  a.imu_b_valid = false;
  a.imu_ab_ready = false;
  CHECK_EQ(arming_checks(a, strict), ARM_FAIL_IMU_B_MISSING | ARM_FAIL_VBAT);
  ArmingConfig nocal = c;
  nocal.require_gyro_cal = false;
  a = ok;
  a.gyro_cal_done = false;
  CHECK_EQ(arming_checks(a, nocal), 0);
}

TEST(refuses_to_arm_when_unconverged_or_bad_policy) {
  Rig r;
  r.sim.p = {0, 0, 0};  // on the ground: the arm tick integrates the sim (no free fall)
  r.sim.q = Quat::from_euler(0.3f, 0, 0);
  r.loop.set_policy(static_cast<const FloatModel*>(nullptr));
  r.tick(false);
  r.arm(false);  // estimator not converged after 1 step
  CHECK(!r.out.armed);
  CHECK(r.out.arming_failures & ARM_FAIL_ESTIMATOR);
  r.warmup(1500);
  r.arm(true);   // wants policy, none loaded
  CHECK(!r.out.armed);
  CHECK_EQ(r.out.arming_failures, ARM_FAIL_POLICY);
  // Corrupted weights -> CRC mismatch -> refuse.
  float a[4] = {kHoverA0, 0, 0, 0};
  ConstPolicy p(a);
  p.b1[2] = 0.1f;
  r.loop.set_policy(&p.m);
  CHECK(!r.loop.policy_ok());
  r.warmup(10);
  r.arm(true);
  CHECK(!r.out.armed);
}

TEST(fallback_takeoff_and_hover) {
  Rig r;
  r.sim.p = {0, 0, 0};
  r.warmup(1000);
  r.arm(false);
  CHECK(r.out.armed);
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);
  r.run(8000);
  std::printf("  hover: p=(%.3f %.3f %.3f) est=(%.3f %.3f %.3f) v=%.3f\n", static_cast<double>(r.sim.p.x),
              static_cast<double>(r.sim.p.y), static_cast<double>(r.sim.p.z), static_cast<double>(r.loop.state().pos.x),
              static_cast<double>(r.loop.state().pos.y), static_cast<double>(r.loop.state().pos.z),
              static_cast<double>(norm(r.sim.v)));
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);
  CHECK_NEAR(r.sim.p.z, 1.0f, 0.1);
  CHECK(std::sqrt(sq(r.sim.p.x) + sq(r.sim.p.y)) < 0.2f);
  CHECK(norm(r.sim.v) < 0.1f);
  CHECK(tilt_angle(r.sim.q) < 0.05f);
  CHECK_EQ(r.out.fault_history, 0);
}

TEST(fallback_recovers_from_disturbance_with_noise) {
  Rig r;
  r.sim.gyro_noise = 0.02f;
  r.sim.acc_noise = 0.2f;
  r.warmup(1000);
  r.arm(false);
  r.run(2000);
  r.sim.omega = {6.0f, -4.0f, 2.0f};  // kick
  r.run(4000);
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);
  CHECK(tilt_angle(r.sim.q) < 0.1f);
  CHECK_NEAR(r.sim.p.z, 1.0f, 0.2);
  CHECK(norm(r.sim.v) < 0.2f);
}

TEST(policy_flies_then_nan_policy_falls_back) {
  // Constant hover policy flies a few seconds in POLICY mode.
  float a[4] = {kHoverA0, 0, 0, 0};
  ConstPolicy p(a);
  Rig r;
  r.loop.set_policy(&p.m);
  CHECK(r.loop.policy_ok());
  r.warmup(1000);
  r.arm(true);
  CHECK(r.out.mode == FlightMode::POLICY);
  r.run(1000);
  CHECK(r.out.mode == FlightMode::POLICY);
  CHECK(tilt_angle(r.sim.q) < 0.1f);
  CHECK_NEAR(r.loop.last_action()[0], kHoverA0, 1e-5);

  // Now a policy whose output is NaN (bias NaN, CRC matches so it arms).
  float b[4] = {kHoverA0, 0, 0, 0};
  ConstPolicy bad(b);
  bad.b1[1] = NAN;
  bad.m.crc32 = policy_crc32(bad.m);
  Rig r2;
  r2.loop.set_policy(&bad.m);
  r2.warmup(1000);
  r2.arm(true);
  CHECK(r2.out.armed);
  r2.run(5);
  CHECK(r2.out.mode == FlightMode::FALLBACK_HOLD);
  CHECK_EQ(r2.out.first_fault, FAULT_POLICY_NONFINITE);
  r2.run(5000);
  CHECK(r2.out.mode == FlightMode::FALLBACK_HOLD);
  CHECK(tilt_angle(r2.sim.q) < 0.05f);
  CHECK_NEAR(r2.sim.p.z, 1.0f, 0.15);
  for (float c : r2.out.cmd) CHECK(is_finite(c));
}

TEST(simplex_rescues_flipping_policy) {
  // Policy commands a hard roll: the envelope monitors must trip and the
  // fallback must bring the vehicle back to level hover.
  float a[4] = {kHoverA0, 0.8f, 0, 0};
  ConstPolicy p(a);
  Rig r;
  r.loop.set_policy(&p.m);
  r.warmup(1000);
  r.arm(true);
  float max_tilt = 0;
  for (int i = 0; i < 4000; ++i) {
    r.tick();
    max_tilt = std::fmax(max_tilt, tilt_angle(r.sim.q));
    if (std::getenv("NF_TRACE") && i % 50 == 0)
      std::printf("    t=%d mode=%s tilt=%.2f z=%.2f vz=%.2f w=(%.1f %.1f %.1f) thr=%.2f est_tilt=%.2f hist=%x innov=%.2f nis=%.1f\n", i, to_string(r.out.mode),
                  static_cast<double>(tilt_angle(r.sim.q)), static_cast<double>(r.sim.p.z), static_cast<double>(r.sim.v.z),
                  static_cast<double>(r.sim.omega.x), static_cast<double>(r.sim.omega.y), static_cast<double>(r.sim.omega.z),
                  static_cast<double>(r.loop.last_ctbr().thrust_n), static_cast<double>(tilt_angle(r.loop.state().q)), r.out.fault_history,
                  static_cast<double>(r.loop.attitude().innovation()), static_cast<double>(r.loop.translation().nis()));
  }
  std::printf("  rescue: first=%s mode=%s max_tilt=%.2f z=%.2f\n", fault_name(r.out.first_fault), to_string(r.out.mode),
              static_cast<double>(max_tilt), static_cast<double>(r.sim.p.z));
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);
  CHECK(r.out.first_fault == FAULT_TILT_PREDICTED || r.out.first_fault == FAULT_TILT ||
        r.out.first_fault == FAULT_POLICY_SATURATION || r.out.first_fault == FAULT_RATE);
  CHECK(tilt_angle(r.sim.q) < 0.1f);
  CHECK(r.sim.p.z > 0.3f);
}

namespace {
uint32_t g_fake_us = 0;
uint32_t g_fake_step = 0;
uint32_t fake_clock() { g_fake_us += g_fake_step; return g_fake_us; }
}  // namespace

TEST(deadline_overrun_then_reengage) {
  ControlLoopConfig c;
  c.clock_us = fake_clock;
  g_fake_step = 20;
  float a[4] = {kHoverA0, 0, 0, 0};
  ConstPolicy p(a);
  Rig r(c);
  r.loop.set_policy(&p.m);
  r.warmup(1000);
  r.arm(true);
  r.run(500);
  CHECK(r.out.mode == FlightMode::POLICY);
  CHECK_EQ(r.loop.last_inference_us(), 20);
  g_fake_step = 600;  // > 400 us deadline
  r.run(10);
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);
  CHECK_EQ(r.out.first_fault, FAULT_DEADLINE);
  g_fake_step = 20;
  r.run(1000);
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);  // no auto-return
  r.rc.reengage = true;
  r.tick();
  r.rc.reengage = false;
  CHECK(r.out.mode == FlightMode::POLICY);
  r.run(500);
  CHECK(r.out.mode == FlightMode::POLICY);
}

TEST(imu_disagreement_in_loop) {
  Rig r;
  r.warmup(1000);
  float a[4] = {kHoverA0, 0, 0, 0};
  ConstPolicy p(a);
  r.loop.set_policy(&p.m);
  r.arm(true);
  r.run(300);
  r.imu_b_fault = true;
  r.run(200);
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);
  CHECK(r.out.fault_history & FAULT_IMU_DISAGREE);
}

TEST(kill_cuts_motors_same_step) {
  Rig r;
  r.warmup(1000);
  r.arm(false);
  r.run(3000);
  CHECK(r.out.cmd[0] > 0.1f);
  r.rc.kill = true;
  r.tick();
  for (float c : r.out.cmd) CHECK_NEAR(c, 0.0f, 0);
  CHECK(r.out.mode == FlightMode::MOTOR_STOP);
  r.rc.kill = false;
  r.run(100);
  for (float c : r.out.cmd) CHECK_NEAR(c, 0.0f, 0);  // stays stopped
  // Arm switch toggle re-arms (vehicle on the ground by now).
  r.rc.arm = false;
  r.run(1500);
  CHECK(!r.out.armed);
  r.arm(false);
  CHECK(r.out.armed);
  CHECK(r.out.mode == FlightMode::FALLBACK_HOLD);
}

TEST(rc_loss_descends_and_stops) {
  Rig r;
  r.warmup(1000);
  r.arm(false);
  r.run(4000);
  r.rc.link_ok = false;
  int t_stop = -1;
  for (int i = 0; i < 10000; ++i) {
    r.tick();
    if (r.out.mode == FlightMode::MOTOR_STOP) { t_stop = i; break; }
  }
  std::printf("  rc loss: stopped after %d ms, z=%.3f\n", t_stop, static_cast<double>(r.sim.p.z));
  CHECK(r.out.fault_history & FAULT_RC_LOSS);
  CHECK(t_stop > 1500);  // a real descent from 1 m, not an immediate drop
  CHECK(r.sim.p.z < 0.05f);
  for (float c : r.out.cmd) CHECK_NEAR(c, 0.0f, 0);
}

TEST(int8_policy_in_loop) {
  nftest::MlpFixture f;
  int hidden[2] = {64, 64};
  f.build(hidden, 2, 77);
  Rig r;
  r.loop.set_policy(&f.qm);
  CHECK(r.loop.policy_ok());
  r.warmup(1000);
  r.arm(true);
  CHECK(r.out.mode == FlightMode::POLICY);
  r.run(3000);  // random net: whatever it does, the supervisor keeps outputs sane
  for (float c : r.out.cmd) CHECK(is_finite(c) && c >= 0.0f && c <= 1.0f);
}

// ------------------------------------------------------- FOLLOWUPS #3: gyro calibration
TEST(arms_2s_after_power_on_with_calibrated_bias_and_holds_position) {
  Rig r;
  r.gyro_bias = {0.05f, -0.04f, 0.03f};
  r.sim.gyro_noise = 0.05f;
  r.sim.acc_noise = 0.3f;
  r.sim.p = {0, 0, 0};
  r.warmup(1999);
  CHECK(r.loop.gyro_cal().done());
  r.arm(false);  // t = 2.0 s after power-on, first attempt
  CHECK(r.out.armed);
  CHECK_EQ(r.out.arming_failures, 0);
  for (int i = 0; i < 3; ++i) CHECK_NEAR(r.loop.attitude().bias()[i], r.gyro_bias[i], 0.008);
  r.run(4000);
  float max_err = 0.0f;
  for (int i = 0; i < 6000; ++i) {
    r.tick();
    max_err = std::fmax(max_err, norm(r.sim.p - Vec3{0, 0, 1}));
  }
  std::printf("  power-on 2 s, bias (0.05,-0.04,0.03): max hover error %.3f m\n", static_cast<double>(max_err));
  CHECK(max_err < 0.05f);
  CHECK_EQ(r.out.fault_history, 0);
}

TEST(refuses_to_arm_until_calibrated_and_while_moving) {
  Rig r;
  r.sim.p = {0, 0, 0};
  r.warmup(300);  // < one 0.5 s calibration window
  r.arm(false);
  CHECK(!r.out.armed);
  CHECK(r.out.arming_failures & ARM_FAIL_GYRO_CAL);
  CHECK(!(r.out.arming_failures & ARM_FAIL_MOTION));
  r.warmup(700);
  CHECK(r.loop.gyro_cal().done());
  // Picked up / rotated while disarmed: calibration dropped, arming refused.
  r.sim.omega = {0.0f, 0.0f, 1.0f};
  r.warmup(200);
  r.arm(false);
  CHECK(!r.out.armed);
  CHECK(r.out.arming_failures & ARM_FAIL_MOTION);
  CHECK(r.out.arming_failures & ARM_FAIL_GYRO_CAL);
  // Still again: arming waits for a full new still window.
  r.sim.omega = {};
  r.warmup(400);
  r.arm(false);
  CHECK(!r.out.armed);
  CHECK(r.out.arming_failures & ARM_FAIL_MOTION);
  r.warmup(200);
  r.arm(false);
  CHECK(r.out.armed);
  // Opting out keeps the old behaviour (arm as soon as the estimator converged).
  ControlLoopConfig c;
  c.arming.require_gyro_cal = false;
  Rig r2(c);
  r2.sim.p = {0, 0, 0};
  r2.warmup(600);
  r2.sim.omega = {0.0f, 0.0f, 1.0f};
  r2.tick(false);
  r2.sim.omega = {};
  r2.arm(false);
  CHECK(r2.out.armed);
}

// ------------------------------------------------------- FOLLOWUPS #4: filtered A/B check
TEST(arming_imu_ab_check_uses_filtered_difference) {
  // Max-spec noise on both IMUs, independent: single raw samples disagree by
  // ~0.12 rad/s / ~0.7 m/s^2 (norm) on average, the filtered difference does not.
  int raw_fail = 0, armed_first_try = 0;
  const int kTrials = 20;
  for (int k = 0; k < kTrials; ++k) {
    Rig r;
    r.brng = 1000u + static_cast<uint32_t>(k);
    r.sim.rng = 5000u + static_cast<uint32_t>(k);
    r.sim.gyro_noise = 0.05f;
    r.sim.acc_noise = 0.3f;
    r.b_gyro_noise = 0.05f;
    r.b_acc_noise = 0.3f;
    r.gyro_bias = {0.05f, 0.05f, -0.05f};  // A's bias is calibrated out of the comparison
    r.b_offset = {0.01f, -0.01f, 0.01f};
    r.warmup(999);
    ImuSample s = r.sim.imu();  // what the old single-sample check would have compared
    Vec3 gb = r.sim.omega + Vec3{r.bn(), r.bn(), r.bn()} * 0.05f + r.b_offset;
    Vec3 ab = s.accel_b + Vec3{r.bn(), r.bn(), r.bn()} * 0.3f;
    if (!(norm(s.gyro + r.gyro_bias - gb) <= 0.1f) || !(norm(s.accel - ab) <= 1.0f)) ++raw_fail;
    r.arm(false);
    if (r.out.armed) ++armed_first_try;
    else std::printf("  trial %d refused: 0x%x diff=%.3f\n", k, r.out.arming_failures,
                     static_cast<double>(norm(r.loop.imu_gyro_diff())));
  }
  std::printf("  A/B arming: old raw-sample check would refuse %d/%d, filtered check armed %d/%d first try\n", raw_fail,
              kTrials, armed_first_try, kTrials);
  CHECK(raw_fail >= 5);
  CHECK_EQ(armed_first_try, kTrials);

  // A real disagreement (IMU B gyro off by 0.3 rad/s) is still refused...
  Rig bad;
  bad.b_offset = {0.0f, 0.3f, 0.0f};
  bad.warmup(1000);
  bad.arm(false);
  CHECK(!bad.out.armed);
  CHECK_EQ(bad.out.arming_failures, ARM_FAIL_IMU_DISAGREE);
  // ...and right after IMU B (re)appears the filter is not trusted yet.
  Rig fresh;
  fresh.warmup(1000);
  fresh.rc.arm = false;
  for (int i = 0; i < 100; ++i) {
    ImuSample imu = fresh.sim.imu();
    imu.b_valid = i > 90;
    fresh.out = fresh.loop.step(imu, fresh.sim.sensors(), fresh.rc, fresh.t);
    fresh.t += kDtUs;
  }
  fresh.arm(false);
  CHECK(!fresh.out.armed);
  CHECK(fresh.out.arming_failures & ARM_FAIL_IMU_DISAGREE);
}

// ------------------------------------------------------- FOLLOWUPS #6: bumpless handover
namespace {
// Weak motors (thrust 85% of nominal): the fallback needs ~18% more collective
// than nominal weight. Takes off in fallback, hovers in POLICY with a constant
// policy at the true hover thrust, then the policy output goes NaN.
float handover_alt_lost(bool bumpless, float* hover_est = nullptr) {
  const float scale = 0.85f;
  float hover_t = vehicle::kMass * vehicle::kGravity / scale;
  float a[4] = {2.0f * hover_t / (4.0f * vehicle::kThrustMax) - 1.0f, 0, 0, 0};
  ConstPolicy p(a);
  ControlLoopConfig c;
  c.bumpless_handover = bumpless;
  Rig r(c);
  r.sim.thrust_scale = scale;
  r.sim.p = {0, 0, 0};
  r.loop.set_policy(&p.m);
  r.warmup(1000);
  r.arm(false);
  r.run(6000);
  r.rc.reengage = true;
  r.tick();
  r.rc.reengage = false;
  r.run(2000);
  if (r.out.mode != FlightMode::POLICY) return 1e3f;
  if (hover_est) *hover_est = r.loop.hover_thrust_estimate();
  p.b1[1] = NAN;  // in-flight corruption (CRC is not re-checked in flight)
  float z0 = r.sim.p.z, zmin = z0;
  for (int i = 0; i < 3000; ++i) {
    r.tick();
    zmin = std::fmin(zmin, r.sim.p.z);
  }
  if (r.out.mode != FlightMode::FALLBACK_HOLD || r.out.first_fault != FAULT_POLICY_NONFINITE) return 2e3f;
  return z0 - zmin;
}
}  // namespace

TEST(bumpless_fallback_handover_seeds_vertical_integrator) {
  float est = 0.0f;
  float lost_new = handover_alt_lost(true, &est);
  float lost_old = handover_alt_lost(false);
  const float want = vehicle::kMass * vehicle::kGravity / 0.85f;
  std::printf("  handover (thrust x0.85): hover-thrust est %.3f N (true %.3f), alt lost %.3f m bumpless vs %.3f m reset\n",
              static_cast<double>(est), static_cast<double>(want), static_cast<double>(lost_new),
              static_cast<double>(lost_old));
  CHECK_NEAR(est, want, 0.03f * want);
  CHECK(lost_new < 0.03f);
  CHECK(lost_old > 3.0f * lost_new);
}

namespace {
// FOLLOWUPS 11: hovering in FALLBACK_HOLD, the vehicle is pushed to 3 m/s
// horizontally (20 ms, felt by the accelerometer as in SITL) and the fallback brakes hard. Returns the peak attitude-estimate
// tilt error (rad) and peak Mahony innovation during the 3 s after the push.
void braking_estimate_error(bool kf_ref, float* peak_err, float* peak_innov, uint32_t* faults) {
  ControlLoopConfig c;
  c.mahony_kf_ref = kf_ref;
  Rig r(c);
  r.sim.gyro_noise = 0.01f;
  r.sim.acc_noise = 0.1f;
  r.warmup(1000);
  r.arm(false);
  r.run(4000);
  *peak_err = 0.0f;
  *peak_innov = 0.0f;
  for (int i = 0; i < 3000; ++i) {
    r.sim.ext_acc_w = i < 20 ? Vec3{150.0f, 0.0f, 0.0f} : Vec3{};  // 20 ms push to 3 m/s
    r.tick();
    const Vec3 zt = r.sim.q.rotate({0, 0, 1}), ze = r.loop.state().q.rotate({0, 0, 1});
    *peak_err = std::fmax(*peak_err, std::acos(clampf(dot(zt, ze), -1.0f, 1.0f)));
    *peak_innov = std::fmax(*peak_innov, r.loop.attitude().innovation());
  }
  *faults = r.out.fault_history;
}
}  // namespace

TEST(mahony_kf_acceleration_reference_while_braking) {
  float e_new, i_new, e_old, i_old;
  uint32_t f_new, f_old;
  braking_estimate_error(true, &e_new, &i_new, &f_new);
  braking_estimate_error(false, &e_old, &i_old, &f_old);
  std::printf("  braking from 3 m/s: peak attitude error %.2f deg (KF-accel ref) vs %.2f deg (gravity ref), "
              "peak innovation %.3f vs %.3f rad\n", static_cast<double>(e_new * 57.2958f),
              static_cast<double>(e_old * 57.2958f), static_cast<double>(i_new), static_cast<double>(i_old));
  CHECK(e_new < 0.05f);         // < 3 deg
  CHECK(e_old > 2.0f * e_new);  // the gravity reference is pulled off while braking
  CHECK(i_new < 0.5f * i_old);
  CHECK_EQ(f_new, 0);
  CHECK_EQ(f_old & ~static_cast<uint32_t>(FAULT_ESTIMATOR), 0);
  std::printf("  faults: %u (KF-accel ref) vs %u (gravity ref)\n", f_new, f_old);
}

TEST(cascade_seed_hover_thrust_sets_level_collective) {
  CascadeController cc;
  CascadeConfig cfg;
  cc.init(cfg);
  VehicleState s;
  s.pos = {0, 0, 1};
  s.pos_valid = s.height_valid = true;
  const float t_h = 1.15f * cfg.mass * cfg.gravity;
  cc.seed_hover_thrust(t_h);
  CHECK_NEAR(cc.vel_integrator().z, t_h / cfg.mass - cfg.gravity, 1e-5);
  CtbrCommand cmd = cc.update(s, {0, 0, 1}, 0.0f, CascadeMode::HOLD, 0.0f);
  CHECK_NEAR(cmd.thrust_n, t_h, 1e-3);
  cc.seed_hover_thrust(10.0f * cfg.mass * cfg.gravity);  // clamped to the integrator limit
  CHECK_NEAR(cc.vel_integrator().z, cfg.vel_i_limit.z, 1e-6);
  cc.seed_hover_thrust(NAN);
  CHECK_NEAR(cc.vel_integrator().z, 0.0f, 0);
}

NFTEST_MAIN()
