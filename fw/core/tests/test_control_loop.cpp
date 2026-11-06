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
  explicit Rig(const ControlLoopConfig& c = ControlLoopConfig{}) { loop.init(c); }
  // Disarmed warm-up with the vehicle held still (sim not integrated).
  void warmup(int ms) {
    rc.arm = false;
    for (int i = 0; i < ms; ++i) tick(false);
  }
  void tick(bool integrate = true) {
    ImuSample imu = sim.imu();
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
  ok.imu.b_valid = true;
  CHECK_EQ(arming_checks(ok, c), 0);
  struct { void (*f)(ArmingInput&); uint32_t bit; } cases[] = {
      {[](ArmingInput& a) { a.kill = true; }, ARM_FAIL_KILL},
      {[](ArmingInput& a) { a.rc_link_ok = false; }, ARM_FAIL_RC_LINK},
      {[](ArmingInput& a) { a.throttle = 0.3f; }, ARM_FAIL_THROTTLE},
      {[](ArmingInput& a) { a.imu.gyro_b.x = 0.5f; }, ARM_FAIL_IMU_DISAGREE},
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
  a.imu.b_valid = false;
  CHECK_EQ(arming_checks(a, strict), ARM_FAIL_IMU_B_MISSING | ARM_FAIL_VBAT);
}

TEST(refuses_to_arm_when_unconverged_or_bad_policy) {
  Rig r;
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
  CHECK(r.out.first_fault == FAULT_TILT || r.out.first_fault == FAULT_POLICY_SATURATION ||
        r.out.first_fault == FAULT_RATE);
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

NFTEST_MAIN()
