#include <cstring>

#include "nf/supervisor.hpp"
#include "nftest.hpp"

using namespace nf;

namespace {
constexpr uint64_t kDt = 1000;  // 1 kHz

struct Harness {
  Supervisor sup;
  SupervisorInput in;
  uint64_t t = 1000000;
  int step_n = 0;
  uint32_t seed = 1;
  explicit Harness(const SupervisorConfig& c = SupervisorConfig{}, bool policy = true) {
    sup.init(c);
    sup.arm(t, policy);
  }
  float noise(float a) {
    seed = seed * 1664525u + 1013904223u;
    return (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 2.0f * a;
  }
  // Realistic clean flight: manoeuvring within the envelope.
  void nominal() {
    float ts = static_cast<float>(t) * 1e-6f;
    in = SupervisorInput{};
    in.now_us = t;
    in.armed = true;
    in.rc_ok = true;
    in.q = Quat::from_euler(0.35f * std::sin(1.3f * ts), 0.3f * std::cos(0.9f * ts), 0.5f * ts);
    in.omega = {2.0f * std::sin(3.0f * ts) + noise(0.05f), 1.5f * std::cos(2.0f * ts) + noise(0.05f), 0.5f};
    in.pos = {1.5f * std::sin(0.3f * ts), 1.5f * std::cos(0.4f * ts), 1.0f + 0.4f * std::sin(0.5f * ts)};
    in.vel = {0.5f, -0.3f, 0.2f};
    in.pos_valid = in.height_valid = true;
    in.accel_innov = 0.05f + noise(0.02f);
    in.kf_nis = 1.0f + noise(0.5f);
    in.gyro_a = in.omega;
    in.gyro_b = in.omega + Vec3{noise(0.05f), noise(0.05f), noise(0.05f)};
    in.acc_a = {0.3f, -0.2f, 9.81f};
    in.acc_b = in.acc_a + Vec3{noise(0.3f), noise(0.3f), noise(0.3f)};
    in.imu_b_valid = true;
    in.policy_ran = (step_n % 2) == 0;
    // Smooth action with a sharp-but-single step every 2 s (not oscillation).
    in.action[0] = -0.7f + 0.2f * std::sin(2.0f * ts) + noise(0.01f);
    in.action[1] = 0.6f * std::sin(6.0f * ts) + noise(0.02f);
    in.action[2] = (static_cast<int>(ts / 2.0f) % 2) ? 0.5f : -0.5f;
    in.action[3] = 0.99f * std::sin(3.0f * ts);  // touches saturation for ~95 ms at a time
    in.inference_us = 40 + static_cast<uint32_t>(noise(10.0f) + 10.0f);
    // A little jitter on the period.
  }
  FlightMode step() {
    FlightMode m = sup.update(in);
    t += kDt + static_cast<uint64_t>(noise(50.0f) + 50.0f);
    ++step_n;
    return m;
  }
  // Run with an injector for `ms` milliseconds.
  template <class F>
  FlightMode run(int ms, F inject) {
    FlightMode m = sup.mode();
    uint64_t end = t + static_cast<uint64_t>(ms) * 1000u;
    while (t < end) {
      nominal();
      inject(*this);
      m = step();
    }
    return m;
  }
};
void none(Harness&) {}
}  // namespace

TEST(nominal_flight_never_trips) {
  Harness h;
  FlightMode m = h.run(20000, none);
  CHECK(m == FlightMode::POLICY);
  CHECK_EQ(h.sup.fault_history(), 0);
  CHECK_EQ(h.sup.first_fault(), 0);
}

struct MonitorCase {
  const char* name;
  uint32_t bit;
  void (*inject)(Harness&);
  int min_ms;  // must NOT have tripped after this long (debounce), -1 = skip
};

TEST(every_monitor_trips_on_its_fault) {
  MonitorCase cases[] = {
      {"tilt", FAULT_TILT, [](Harness& h) { h.in.q = Quat::from_euler(1.3f, 0, 0); }, 40},
      {"rate", FAULT_RATE, [](Harness& h) { h.in.omega.x = 15.0f; }, 20},
      {"geofence_xy", FAULT_GEOFENCE, [](Harness& h) { h.in.pos.x = 3.5f; }, 10},
      {"geofence_ceiling", FAULT_GEOFENCE, [](Harness& h) { h.in.pos.z = 2.8f; }, 10},
      {"height_floor", FAULT_HEIGHT_FLOOR, [](Harness& h) { h.in.pos.z = 0.15f; }, 80},
      {"policy_nan", FAULT_POLICY_NONFINITE, [](Harness& h) { h.in.action[2] = NAN; }, -1},
      {"policy_inf", FAULT_POLICY_NONFINITE, [](Harness& h) { h.in.action[0] = -INFINITY; }, -1},
      {"policy_range", FAULT_POLICY_RANGE, [](Harness& h) { h.in.action[1] = 1.3f; }, -1},
      {"policy_saturation", FAULT_POLICY_SATURATION, [](Harness& h) { h.in.action[0] = 1.0f; }, 150},
      {"policy_oscillation", FAULT_POLICY_OSCILLATION,
       [](Harness& h) { h.in.action[1] = (h.step_n / 2) % 2 ? 0.4f : -0.4f; }, 20},
      {"deadline", FAULT_DEADLINE, [](Harness& h) { h.in.inference_us = 900; }, 3},
      {"loop_jitter", FAULT_LOOP_JITTER, [](Harness& h) { h.t += 2000; }, 2},
      {"imu_gyro", FAULT_IMU_DISAGREE, [](Harness& h) { h.in.gyro_b.z += 1.0f; }, 40},
      {"imu_accel", FAULT_IMU_DISAGREE, [](Harness& h) { h.in.acc_b.x += 5.0f; }, 40},
      {"estimator_innov", FAULT_ESTIMATOR, [](Harness& h) { h.in.accel_innov = 0.6f; }, 150},
      {"estimator_nis", FAULT_ESTIMATOR, [](Harness& h) { h.in.kf_nis = 50.0f; }, 150},
      {"estimator_nan", FAULT_ESTIMATOR, [](Harness& h) { h.in.est_finite = false; }, -1},
      {"rc_loss", FAULT_RC_LOSS, [](Harness& h) { h.in.rc_ok = false; }, 280},
  };
  for (auto& c : cases) {
    Harness h;
    h.run(1000, none);
    bool early = false;
    if (c.min_ms > 0) {
      h.run(c.min_ms, c.inject);
      early = h.sup.mode() != FlightMode::POLICY;
    }
    h.run(1000, c.inject);
    bool tripped = (h.sup.fault_history() & c.bit) != 0;
    bool first_ok = h.sup.first_fault() == c.bit;
    bool left_policy = h.sup.mode() != FlightMode::POLICY;
    if (!tripped || early || !first_ok || !left_policy)
      std::printf("  monitor %s: tripped=%d early=%d first=%s mode=%s\n", c.name, tripped, early,
                  fault_name(h.sup.first_fault()), to_string(h.sup.mode()));
    CHECK(tripped);
    CHECK(!early);
    CHECK(first_ok);
    CHECK(left_policy);
  }
}

TEST(short_glitches_are_debounced) {
  Harness h;
  h.run(500, none);
  h.run(20, [](Harness& x) { x.in.q = Quat::from_euler(1.3f, 0, 0); });   // < 50 ms
  h.run(200, none);
  h.run(15, [](Harness& x) { x.in.omega.y = -20.0f; });                  // < 30 ms
  h.run(200, none);
  h.run(100, [](Harness& x) { x.in.accel_innov = 1.0f; });               // < 200 ms
  h.run(200, none);
  h.run(2, [](Harness& x) { x.in.inference_us = 2000; });                // 1 miss (policy every other step)
  h.run(500, none);
  CHECK(h.sup.mode() == FlightMode::POLICY);
  CHECK_EQ(h.sup.fault_history(), 0);
}

TEST(ladder_order_hold_descent_stop) {
  Harness h;
  h.run(500, none);
  auto tilted = [](Harness& x) { x.in.q = Quat::from_euler(1.4f, 0, 0); };
  h.run(100, tilted);
  CHECK(h.sup.mode() == FlightMode::FALLBACK_HOLD);
  h.run(1300, tilted);  // < hold_escalate (1.5 s since HOLD entry)
  CHECK(h.sup.mode() == FlightMode::FALLBACK_HOLD);
  h.run(400, tilted);
  CHECK(h.sup.mode() == FlightMode::CONTROLLED_DESCENT);
  CHECK(h.sup.fault_history() & FAULT_DESCENT_ENVELOPE);
  h.run(1100, tilted);
  CHECK(h.sup.mode() == FlightMode::MOTOR_STOP);
  CHECK_EQ(h.sup.first_fault(), FAULT_TILT);
}

TEST(descent_lands_then_motor_stop) {
  Harness h;
  h.run(500, none);
  h.run(50, [](Harness& x) { x.in.est_finite = false; });  // severe -> straight to DESCENT
  CHECK(h.sup.mode() == FlightMode::CONTROLLED_DESCENT);
  h.run(300, [](Harness& x) { x.in.pos.z = 0.5f; x.in.vel.z = -0.4f; });
  CHECK(h.sup.mode() == FlightMode::CONTROLLED_DESCENT);
  h.run(700, [](Harness& x) { x.in.pos.z = 0.03f; x.in.vel.z = 0.0f; });
  CHECK(h.sup.mode() == FlightMode::MOTOR_STOP);
}

TEST(descent_timeout_stops_motors) {
  SupervisorConfig c;
  c.descent_timeout_us = 2000000;
  Harness h(c);
  h.run(500, none);
  h.run(50, [](Harness& x) { x.in.rc_ok = false; x.t += 0; });
  h.run(400, [](Harness& x) { x.in.rc_ok = false; });
  CHECK(h.sup.mode() == FlightMode::CONTROLLED_DESCENT);
  h.run(2100, [](Harness& x) { x.in.rc_ok = false; x.in.pos.z = 0.8f; });
  CHECK(h.sup.mode() == FlightMode::MOTOR_STOP);
}

TEST(no_auto_return_and_explicit_reengage) {
  Harness h;
  h.run(500, none);
  h.run(5, [](Harness& x) { x.in.action[0] = NAN; });
  CHECK(h.sup.mode() == FlightMode::FALLBACK_HOLD);
  // Fault gone for 10 s: still in fallback.
  h.run(10000, none);
  CHECK(h.sup.mode() == FlightMode::FALLBACK_HOLD);
  // Re-engage refused while a monitor is active.
  h.run(100, [](Harness& x) { x.in.gyro_b.x += 2.0f; });
  CHECK(!h.sup.request_reengage(h.t));
  CHECK(h.sup.mode() == FlightMode::FALLBACK_HOLD);
  h.run(100, none);
  CHECK(h.sup.request_reengage(h.t));
  CHECK(h.sup.mode() == FlightMode::POLICY);
  // History and first fault are kept after re-engage.
  CHECK_EQ(h.sup.first_fault(), FAULT_POLICY_NONFINITE);
  CHECK(h.sup.fault_history() & FAULT_IMU_DISAGREE);
  h.run(2000, none);
  CHECK(h.sup.mode() == FlightMode::POLICY);
  // Never from DESCENT.
  h.run(10, [](Harness& x) { x.in.est_finite = false; });
  CHECK(h.sup.mode() == FlightMode::CONTROLLED_DESCENT);
  h.run(100, none);
  CHECK(!h.sup.request_reengage(h.t));
  CHECK(h.sup.mode() == FlightMode::CONTROLLED_DESCENT);
}

TEST(kill_overrides_everything) {
  FlightMode modes[] = {FlightMode::POLICY, FlightMode::FALLBACK_HOLD, FlightMode::CONTROLLED_DESCENT};
  for (FlightMode target : modes) {
    Harness h;
    h.run(200, none);
    if (target != FlightMode::POLICY) h.run(5, [](Harness& x) { x.in.action[0] = NAN; });
    if (target == FlightMode::CONTROLLED_DESCENT) h.run(5, [](Harness& x) { x.in.est_finite = false; });
    CHECK(h.sup.mode() == target);
    // Kill with every other fault active at once: one step -> MOTOR_STOP.
    h.nominal();
    h.in.kill = true;
    h.in.action[0] = NAN;
    h.in.q = Quat::from_euler(2.0f, 0, 0);
    CHECK(h.step() == FlightMode::MOTOR_STOP);
    CHECK(h.sup.fault_history() & FAULT_KILL);
    // Releasing kill does not bring the motors back.
    h.run(1000, none);
    CHECK(h.sup.mode() == FlightMode::MOTOR_STOP);
    CHECK(!h.sup.request_reengage(h.t));
  }
  // Kill also wins in POLICY with no faults at all, and over disarm state.
  Harness h;
  h.nominal();
  h.in.kill = true;
  CHECK(h.step() == FlightMode::MOTOR_STOP);
}

TEST(first_fault_is_kept) {
  Harness h;
  h.run(200, none);
  h.run(5, [](Harness& x) { x.in.action[3] = 3.0f; });  // POLICY_RANGE
  h.run(200, [](Harness& x) { x.in.q = Quat::from_euler(0, 1.4f, 0); });
  CHECK_EQ(h.sup.first_fault(), FAULT_POLICY_RANGE);
  CHECK(h.sup.fault_history() & FAULT_TILT);
  CHECK(h.sup.first_fault_time_us() > 0u);
}

TEST(disarm_and_fallback_only_arm) {
  Harness h(SupervisorConfig{}, false);
  CHECK(h.sup.mode() == FlightMode::FALLBACK_HOLD);
  h.run(100, none);
  CHECK(h.sup.mode() == FlightMode::FALLBACK_HOLD);  // policy monitors inactive
  h.nominal();
  h.in.armed = false;
  CHECK(h.step() == FlightMode::DISARMED);
  CHECK_NEAR(static_cast<float>(std::strlen(to_string(FlightMode::MOTOR_STOP))), 10.0f, 0);
}

NFTEST_MAIN()
