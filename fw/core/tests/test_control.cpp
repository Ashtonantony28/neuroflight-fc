#include "nf/control.hpp"
#include "nftest.hpp"
#include "quad_sim.hpp"

using namespace nf;

namespace {
Mixer make_mixer() {
  Mixer m;
  m.init(MixerParams{});
  return m;
}
const float kHover = vehicle::kMass * vehicle::kGravity;
}  // namespace

TEST(mixer_torque_signs_match_interfaces) {
  Mixer m = make_mixer();
  // Positive roll torque (tau_x > 0, right side down in FLU means left side up):
  // left motors (0, 3) must spin up relative to right motors (1, 2).
  MixerResult r = m.mix(kHover, {0.02f, 0, 0});
  CHECK(r.thrust[0] > r.thrust[1]); CHECK(r.thrust[3] > r.thrust[2]);
  CHECK_NEAR(r.thrust[0], r.thrust[3], 1e-5); CHECK_NEAR(r.thrust[1], r.thrust[2], 1e-5);
  // Positive pitch (tau_y > 0 about +y = nose down in FLU): rear motors (2, 3) up.
  r = m.mix(kHover, {0, 0.02f, 0});
  CHECK(r.thrust[2] > r.thrust[1]); CHECK(r.thrust[3] > r.thrust[0]);
  // Positive yaw (CCW from above): CW props 0 and 2 (reaction +z) spin up.
  r = m.mix(kHover, {0, 0, 0.005f});
  CHECK(r.thrust[0] > r.thrust[1]); CHECK(r.thrust[2] > r.thrust[3]);
  CHECK_NEAR(r.thrust[0], r.thrust[2], 1e-5);
  // Exact allocation round-trip (unsaturated): torque_from_thrusts(mix(tau)) == tau.
  Vec3 tau{0.013f, -0.021f, 0.004f};
  r = m.mix(kHover, tau);
  CHECK(!r.any_saturated());
  Vec3 back = m.torque_from_thrusts(r.thrust);
  CHECK_NEAR(back.x, tau.x, 1e-5); CHECK_NEAR(back.y, tau.y, 1e-5); CHECK_NEAR(back.z, tau.z, 1e-5);
  float F = r.thrust[0] + r.thrust[1] + r.thrust[2] + r.thrust[3];
  CHECK_NEAR(F, kHover, 1e-4);
}

TEST(mixer_physical_sign_check_in_sim) {
  // Apply a positive-roll mix to the sim: omega_x must become positive.
  for (int ax = 0; ax < 3; ++ax) {
    nftest::QuadSim sim;
    sim.ground = false;
    Mixer m = make_mixer();
    Vec3 tau{};
    tau[ax] = ax == 2 ? 0.004f : 0.01f;
    MixerResult r = m.mix(kHover, tau);
    for (int i = 0; i < 4; ++i) sim.w[i] = r.cmd[i];
    for (int i = 0; i < 20; ++i) sim.step(r.cmd, 0.001f);
    CHECK(sim.omega[ax] > 0.05f);
  }
}

TEST(mixer_thrust_curve_inverse) {
  Mixer m = make_mixer();
  for (float c = vehicle::kIdleCommand; c <= 1.0f; c += 0.05f) CHECK_NEAR(m.thrust_to_cmd(m.cmd_to_thrust(c)), c, 1e-5);
  MixerResult r = m.mix(0.0f, {});
  for (float c : r.cmd) CHECK_NEAR(c, vehicle::kIdleCommand, 1e-6);  // floor at idle
  r = m.mix(100.0f, {});
  for (float c : r.cmd) CHECK_NEAR(c, 1.0f, 1e-6);
  CHECK(r.thrust_saturated);
  r = m.mix(NAN, {NAN, 0, 0});
  for (float c : r.cmd) CHECK(is_finite(c));
}

TEST(mixer_saturation_keeps_roll_pitch_authority) {
  Mixer m = make_mixer();
  // Big yaw + moderate roll/pitch near full thrust: roll/pitch delivered exactly,
  // yaw reduced, all commands in range.
  Vec3 tau{0.08f, -0.05f, 0.05f};
  MixerResult r = m.mix(4.0f * vehicle::kThrustMax * 0.85f, tau);
  CHECK(r.yaw_saturated);
  CHECK(!r.rp_saturated);
  Vec3 got = m.torque_from_thrusts(r.thrust);
  CHECK_NEAR(got.x, tau.x, 1e-4);
  CHECK_NEAR(got.y, tau.y, 1e-4);
  CHECK(std::fabs(got.z) < tau.z);
  CHECK(got.z >= 0.0f);
  for (int i = 0; i < 4; ++i) { CHECK(r.cmd[i] <= 1.0f); CHECK(r.cmd[i] >= vehicle::kIdleCommand); }
  // At low thrust the collective is raised to keep the roll authority.
  r = m.mix(0.2f, {0.05f, 0, 0});
  got = m.torque_from_thrusts(r.thrust);
  CHECK_NEAR(got.x, 0.05f, 1e-4);
  CHECK(r.thrust_saturated);
  // Impossible roll/pitch demand: scaled uniformly, direction preserved.
  r = m.mix(kHover, {2.0f, 1.0f, 0.0f});
  CHECK(r.rp_saturated);
  got = m.torque_from_thrusts(r.thrust);
  CHECK_NEAR(got.x / got.y, 2.0f, 1e-3);
}

// Hand-computed cases of the INTERFACES.md section 9 rule (sim/ctbr.py::Mixer).
// a=0.046, k=0.006, Tmax=4.5, A=0.7, idle=0.05 -> T_idle = 4.5*(0.7*0.05^2+0.3*0.05) = 0.075375.
TEST(mixer_section9_hand_computed) {
  Mixer m = make_mixer();
  const float lo = 0.075375f;
  CHECK_NEAR(m.min_thrust(), lo, 1e-6);
  struct Case {
    float F; Vec3 tau; float T[4]; bool sat; float yaw_scale;
  } cases[] = {
      // A: unsaturated roll. t_rp = 0.0368/(4*0.046) = 0.2*[+,-,-,+], base 2.5.
      {10.0f, {0.0368f, 0, 0}, {2.7f, 2.3f, 2.3f, 2.7f}, false, 1.0f},
      // B: base 4.0, roll 0.2, yaw t_y = 0.012/(4*0.006) = 0.5*[+,-,+,-]. Motor 0 room 0.3 -> s_y=0.6.
      {16.0f, {0.0368f, 0, 0.012f}, {4.5f, 3.5f, 4.1f, 3.9f}, true, 0.6f},
      // B': same with small yaw (0.2/motor) fits: s_y = 1.
      {16.0f, {0.0368f, 0, 0.0048f}, {4.4f, 3.6f, 4.0f, 4.0f}, false, 1.0f},
      // C: collective shifted down: base 4.5 -> min(.., 4.5-0.2) = 4.3.
      {18.0f, {0.0368f, 0, 0}, {4.5f, 4.1f, 4.1f, 4.5f}, true, 1.0f},
      // D: pitch demand too big: rp scaled to the full range, collective = lo + range/2.
      {10.0f, {0, -1.0f, 0}, {4.5f, 4.5f, lo, lo}, true, -1.0f},
      // E: zero thrust request: collective raised to T_idle.
      {0.0f, {0, 0, 0}, {lo, lo, lo, lo}, true, -1.0f},  // s_y = 0 (room 0 at T_idle), as in sim
      // F: pure yaw at hover: t_y = 0.0024/0.024 = 0.1*[+,-,+,-] around 0.613125.
      {2.4525f, {0, 0, 0.0024f}, {0.713125f, 0.513125f, 0.713125f, 0.513125f}, false, 1.0f},
  };
  for (auto& c : cases) {
    MixerResult r = m.mix(c.F, c.tau);
    for (int i = 0; i < 4; ++i) CHECK_NEAR(r.thrust[i], c.T[i], 2e-5);
    CHECK(r.any_saturated() == c.sat);
    if (c.yaw_scale >= 0.0f) CHECK_NEAR(r.yaw_scale, c.yaw_scale, 1e-5);  // -1: yaw-free edge case
  }
  // Case A through the inverse thrust curve: T=2.7 -> x=0.6,
  // w = (-0.3 + sqrt(0.09 + 2.8*0.6)) / 1.4 = 0.7360093.
  MixerResult r = m.mix(10.0f, {0.0368f, 0, 0});
  CHECK_NEAR(r.cmd[0], 0.7360093f, 2e-6);
  CHECK_NEAR(r.cmd[3], 0.7360093f, 2e-6);
}

TEST(rate_pid_section9_gains_and_order) {
  RatePidConfig c;
  const Vec3 J{vehicle::kInertia[0], vehicle::kInertia[1], vehicle::kInertia[2]};
  CHECK_NEAR(c.kp.x, 35, 0); CHECK_NEAR(c.kp.z, 20, 0);
  CHECK_NEAR(c.ki.x, 60, 0); CHECK_NEAR(c.ki.z, 20, 0);
  CHECK_NEAR(c.kd.x, 0.7, 1e-7); CHECK_NEAR(c.kd.z, 0, 0);
  CHECK_NEAR(c.i_limit.y, 8, 0);
  RatePid pid;
  pid.init(c, 1000.0f);
  CHECK_NEAR(pid.d_alpha(), 1.0 - std::exp(-2.0 * 3.14159265358979 * 20.0 * 0.001), 1e-6);
  // Step 1: integrator still zero; D on measurement -> no kick from the setpoint.
  Vec3 tau = pid.update({1, 0, 0}, {0, 0, 0});
  CHECK_NEAR(tau.x, J.x * 35.0f, 1e-9);
  pid.integrate(false);
  CHECK_NEAR(pid.integrator().x, 0.001f, 1e-9);
  // Step 2: rate moves 0 -> 0.1: d = -100 rad/s^2, filtered by alpha.
  tau = pid.update({1, 0, 0}, {0.1f, 0, 0});
  float dfilt = pid.d_alpha() * -100.0f;
  CHECK_NEAR(tau.x, J.x * (35.0f * 0.9f + 0.7f * dfilt + 60.0f * 0.001f), 1e-8);
}

TEST(rate_pid_anti_windup) {
  RatePidConfig c;
  RatePid pid; pid.init(c, 1000.0f);
  for (int i = 0; i < 2000; ++i) { pid.update({2, 0, 0}, {0, 0, 0}); pid.integrate(true); }
  CHECK_NEAR(pid.integrator().x, 0.0f, 0);  // frozen while saturated
  for (int i = 0; i < 5000; ++i) { pid.update({2, 0, 0}, {0, 0, 0}); pid.integrate(false); }
  CHECK_NEAR(pid.integrator().x, 8.0f, 1e-5);  // clamped at i_limit
  for (int i = 0; i < 10; ++i) { pid.update({-2, 0, 0}, {0, 0, 0}); pid.integrate(true); }
  CHECK_NEAR(pid.integrator().x, 8.0f, 1e-5);  // frozen in both directions (section 9)
}

TEST(rate_pid_iterm_relax_optional) {
  RatePidConfig c;
  c.relax_threshold = 5.0f;  // firmware-only extra, off by default
  RatePid pid; pid.init(c, 1000.0f);
  RatePid def; def.init(RatePidConfig{}, 1000.0f);
  pid.update({0, 0, 0}, {0, 0, 0});
  def.update({0, 0, 0}, {0, 0, 0});
  for (int i = 0; i < 3; ++i) {
    pid.update({8, 0, 0}, {0, 0, 0}); pid.integrate(false);
    def.update({8, 0, 0}, {0, 0, 0}); def.integrate(false);
  }
  CHECK(pid.relax_factor(0) < 0.2f);       // fast setpoint change -> I accumulation suppressed
  CHECK_NEAR(def.relax_factor(0), 1.0f, 0);  // disabled by default (sim parity)
  CHECK(pid.integrator().x < 0.3f * def.integrator().x);
  for (int i = 0; i < 500; ++i) { pid.update({8, 0, 0}, {7, 0, 0}); pid.integrate(false); }
  CHECK_NEAR(pid.relax_factor(0), 1.0f, 0.02);
}

TEST(attitude_p_signs) {
  AttitudeConfig c;
  // Want roll +0.2 from level -> positive roll rate.
  Vec3 r = attitude_p(Quat{}, Quat::from_euler(0.2f, 0, 0), c);
  CHECK_NEAR(r.x, c.kp.x * 0.2f, 0.01);
  CHECK_NEAR(r.y, 0, 1e-6);
  r = attitude_p(Quat{}, Quat::from_euler(0, -0.1f, 0.3f), c);
  CHECK(r.y < 0); CHECK(r.z > 0);
  // Large error is rate-limited; the shortest path is taken.
  r = attitude_p(Quat{}, Quat::from_euler(0, 0, 3.0f), c);
  CHECK_NEAR(r.z, c.rate_limit.z, 1e-5);
}

TEST(rate_loop_closed_in_sim) {
  // Rate PID + mixer track a roll-rate step on the simulated vehicle.
  nftest::QuadSim sim; sim.ground = false;
  RatePid pid; pid.init(RatePidConfig{}, 1000.0f);
  Mixer m = make_mixer();
  MixerResult r;
  float hover_c = m.thrust_to_cmd(kHover / 4);
  for (float& w : sim.w) w = hover_c;
  float err_sum = 0;
  for (int i = 0; i < 1500; ++i) {
    Vec3 sp = i > 100 ? Vec3{3, -2, 1} : Vec3{};
    Vec3 tau = pid.update(sp, sim.omega);
    r = m.mix(kHover, tau);
    pid.integrate(r.any_saturated());
    sim.step(r.cmd, 0.001f);
    if (i > 1300) err_sum += norm(sim.omega - sp);
  }
  std::printf("  rate tracking mean |err| %.3f rad/s\n", static_cast<double>(err_sum / 200.0f));
  CHECK(err_sum / 200.0f < 0.15f);
}

TEST(cascade_hover_hold_in_sim) {
  // Ideal-state closed loop: cascade + rate PID + mixer + sim, start off-target.
  nftest::QuadSim sim; sim.ground = false;
  sim.p = {0.5f, -0.4f, 0.6f};
  sim.q = Quat::from_euler(0.3f, -0.2f, 0.5f);
  CascadeController cc; cc.init(CascadeConfig{});
  RatePid pid; pid.init(RatePidConfig{}, 1000.0f);
  Mixer m = make_mixer();
  MixerResult r;
  CtbrCommand cmd;
  for (int i = 0; i < 6000; ++i) {
    VehicleState s;
    s.q = sim.q; s.omega = sim.omega; s.pos = sim.p; s.vel = sim.v; s.pos_valid = s.height_valid = true;
    if (i % 2 == 0) cmd = cc.update(s, {0, 0, 1}, 0.0f, CascadeMode::HOLD, 0.002f);
    Vec3 tau = pid.update(cmd.rate_sp, sim.omega);
    r = m.mix(cmd.thrust_n, tau);
    pid.integrate(r.any_saturated());
    sim.step(r.cmd, 0.001f);
  }
  CHECK(norm(sim.p - Vec3{0, 0, 1}) < 0.05f);
  CHECK(norm(sim.v) < 0.05f);
  CHECK(std::fabs(wrap_pi(sim.q.yaw())) < 0.05f);
  // Descent mode descends at ~descent_rate.
  for (int i = 0; i < 1000; ++i) {
    VehicleState s;
    s.q = sim.q; s.omega = sim.omega; s.pos = sim.p; s.vel = sim.v; s.pos_valid = s.height_valid = true;
    if (i % 2 == 0) cmd = cc.update(s, {0, 0, 1}, 0.0f, CascadeMode::DESCENT, 0.002f);
    Vec3 tau = pid.update(cmd.rate_sp, sim.omega);
    r = m.mix(cmd.thrust_n, tau);
    pid.integrate(r.any_saturated());
    sim.step(r.cmd, 0.001f);
  }
  CHECK_NEAR(sim.v.z, -CascadeConfig{}.descent_rate, 0.08);
}

NFTEST_MAIN()
