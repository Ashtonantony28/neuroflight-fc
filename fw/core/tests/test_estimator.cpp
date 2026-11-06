#include "nf/estimator.hpp"
#include "nftest.hpp"

using namespace nf;

namespace {
float quat_angle(const Quat& a, const Quat& b) {
  float d = std::fabs(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z);
  return 2.0f * std::acos(clampf(d, -1.0f, 1.0f));
}
Vec3 accel_for(const Quat& q_true) { return q_true.rotate_inv({0, 0, 9.81f}); }
}  // namespace

TEST(mahony_converges_from_wrong_attitude) {
  Mahony m;
  Mahony::Config c;
  m.init(c);
  Quat truth = Quat::from_euler(0.6f, -0.4f, 0.0f);
  m.reset(Quat::from_euler(-0.5f, 0.5f, 0.0f));  // ~1.4 rad tilt error
  const float dt = 0.001f;
  for (int i = 0; i < 3000; ++i) m.update({0, 0, 0}, accel_for(truth), dt);
  CHECK_NEAR(m.q().roll(), 0.6f, 0.01);
  CHECK_NEAR(m.q().pitch(), -0.4f, 0.01);
  CHECK(m.converged());
}

TEST(mahony_converges_upside_down_start) {
  Mahony m; m.init({});
  m.reset(Quat::from_euler(3.0f, 0.0f, 0.0f));
  for (int i = 0; i < 5000; ++i) m.update({0, 0, 0}, {0, 0, 9.81f}, 0.001f);
  CHECK(tilt_angle(m.q()) < 0.02f);
}

TEST(mahony_tracks_rotation_and_bias) {
  Mahony m;
  Mahony::Config c;
  c.ki = 0.3f;
  m.init(c);
  const Vec3 bias{0.03f, -0.02f, 0.0f};
  Quat truth;
  m.reset(truth);
  const float dt = 0.001f;
  for (int i = 0; i < 30000; ++i) {
    float t = static_cast<float>(i) * dt;
    Vec3 w{0.5f * std::sin(t), 0.4f * std::cos(0.7f * t), 0.2f};
    truth = truth.integrated(w, dt);
    m.update(w + bias, accel_for(truth), dt);
  }
  // Tilt tracked (yaw is unobservable without a magnetometer).
  CHECK(std::fabs(wrap_pi(m.q().roll() - truth.roll())) < 0.02f);
  CHECK(std::fabs(m.q().pitch() - truth.pitch()) < 0.02f);
  CHECK_NEAR(m.bias().x, bias.x, 0.01);
  CHECK_NEAR(m.bias().y, bias.y, 0.01);
}

TEST(mahony_init_from_accel_and_gating) {
  Mahony m; m.init({});
  Quat truth = Quat::from_euler(-0.3f, 0.2f, 0.0f);
  m.init_from_accel(accel_for(truth));
  CHECK(quat_angle(m.q(), truth) < 1e-3f);
  // Free-fall accel (|a| ~ 0) must not corrupt the attitude.
  for (int i = 0; i < 1000; ++i) m.update({0, 0, 0}, {0.1f, 0.0f, 0.0f}, 0.001f);
  CHECK(quat_angle(m.q(), truth) < 1e-3f);
  CHECK(m.q().finite());
  m.update({NAN, 0, 0}, {0, 0, 9.81f}, 0.001f);
  CHECK(m.q().finite());
}

TEST(kf_height_and_velocity) {
  TranslationKf kf;
  TranslationKf::Config c;
  kf.init(c);
  kf.reset({0, 0, 0.5f}, {});
  Quat q;
  const float dt = 0.001f;
  // Truth: climbing at 0.3 m/s from 0.5 m, moving +x at 0.5 m/s (level, so flow
  // = v/h). The KF starts with zero velocity and must pick both up.
  float z = 0.5f;
  for (int i = 0; i < 3000; ++i) {
    z += 0.3f * dt;

    kf.predict({0, 0, 9.81f}, q, dt);
    FlowSample f;
    RangeSample r;
    if (i % 10 == 0) {
      r.valid = true; r.range_m = z;
      f.valid = true; f.quality = 1.0f; f.flow_x = 0.5f / z; f.flow_y = 0.0f;
    }
    kf.correct(f, r, q, {0, 0, 0});
  }
  CHECK_NEAR(kf.pos().z, z, 0.02);
  CHECK_NEAR(kf.vel().z, 0.3f, 0.05);
  CHECK_NEAR(kf.vel().x, 0.5f, 0.05);
  CHECK_NEAR(kf.vel().y, 0.0f, 0.05);
  CHECK(kf.height_valid());
  CHECK(kf.flow_valid());
  CHECK(kf.healthy());
}

TEST(kf_flow_rotation_compensation) {
  // Hovering still at 1 m while pitching: flow contains only rotation, which
  // must be compensated -> velocity stays ~0.
  TranslationKf kf; kf.init({});
  kf.reset({0, 0, 1.0f}, {});
  Quat q;
  for (int i = 0; i < 2000; ++i) {
    float wy = 1.5f * std::sin(0.01f * static_cast<float>(i));
    kf.predict({0, 0, 9.81f}, q, 0.001f);
    FlowSample f; f.valid = true; f.quality = 1; f.flow_x = -wy; f.flow_y = 0;
    RangeSample r; r.valid = true; r.range_m = 1.0f;
    kf.correct(f, r, q, {0, wy, 0});
  }
  CHECK(std::fabs(kf.vel().x) < 0.02f);
}

TEST(kf_rejects_outlier_and_reports_nis) {
  TranslationKf kf; kf.init({});
  kf.reset({0, 0, 1.0f}, {});
  Quat q;
  for (int i = 0; i < 500; ++i) {
    kf.predict({0, 0, 9.81f}, q, 0.001f);
    RangeSample r; r.valid = true; r.range_m = 1.0f;
    kf.correct({}, r, q, {});
  }
  float nis_ok = kf.nis();
  RangeSample bad; bad.valid = true; bad.range_m = 3.0f;
  kf.correct({}, bad, q, {});
  CHECK_NEAR(kf.pos().z, 1.0f, 0.01);  // gated
  CHECK(kf.rejected() >= 1u);
  CHECK(nis_ok < 2.0f);
}

// ------------------------------------------------------------ GyroCalibrator
namespace {
struct Lcg {
  uint32_t s = 7;
  float randn() {  // approx N(0,1)
    float a = 0;
    for (int i = 0; i < 4; ++i) { s = s * 1664525u + 1013904223u; a += static_cast<float>(s >> 8) / 16777216.0f - 0.5f; }
    return a * 1.732f;
  }
  Vec3 randn3(float sd) { return Vec3{randn(), randn(), randn()} * sd; }
};
}  // namespace

TEST(gyro_cal_estimates_bias_from_one_still_window) {
  GyroCalibrator cal;
  cal.init({}, 1000.0f);
  Lcg r;
  const Vec3 bias{0.05f, -0.04f, 0.03f};          // edge of the randomization spec
  const Vec3 up = Quat::from_euler(0.1f, -0.05f, 0).rotate_inv({0, 0, 9.81f});
  int done_at = -1;
  for (int i = 0; i < 2000; ++i) {
    bool ok = cal.update(bias + r.randn3(0.05f), up + r.randn3(0.3f));  // max noise spec
    if (ok && done_at < 0) done_at = i;
  }
  CHECK_EQ(done_at, 499);  // first 0.5 s window
  CHECK(cal.done());
  CHECK(!cal.motion());
  CHECK_EQ(cal.windows(), 4);
  for (int i = 0; i < 3; ++i) CHECK_NEAR(cal.bias()[i], bias[i], 0.003);  // 4 windows: 0.05/sqrt(2000) * 3 sigma
  CHECK_EQ(cal.rejections(), 0);
}

TEST(gyro_cal_rejects_motion_and_restarts) {
  GyroCalibrator cal;
  cal.init({}, 1000.0f);
  Lcg r;
  const Vec3 up{0, 0, 9.81f};
  // Gross rotation (picked up): never done, motion flagged.
  for (int i = 0; i < 1000; ++i) cal.update({0.8f, 0, 0}, up);
  CHECK(!cal.done());
  CHECK(cal.motion());
  // Slow wobble under the gross thresholds: window statistics reject it.
  for (int i = 0; i < 1000; ++i) {
    float t = static_cast<float>(i) * 0.001f;
    cal.update({0.3f * std::sin(20.0f * t), 0, 0}, up + Vec3{1.0f * std::sin(15.0f * t), 0, 0});
  }
  CHECK(!cal.done());
  CHECK(cal.motion());
  CHECK(cal.rejections() >= 2u);
  // Constant rotation beyond the bias spec (turntable): rejected too.
  for (int i = 0; i < 600; ++i) cal.update({0, 0, 0.3f}, up);
  CHECK(!cal.done());
  // Free fall (accel ~ 0): gross motion.
  for (int i = 0; i < 100; ++i) cal.update({}, {0, 0, 0.2f});
  CHECK(cal.motion());
  // Still for less than a window: not yet.
  for (int i = 0; i < 400; ++i) cal.update(Vec3{0.01f, 0, 0} + r.randn3(0.02f), up + r.randn3(0.2f));
  CHECK(!cal.done());
  for (int i = 0; i < 100; ++i) cal.update(Vec3{0.01f, 0, 0} + r.randn3(0.02f), up + r.randn3(0.2f));
  CHECK(cal.done());
  CHECK(!cal.motion());
  CHECK_NEAR(cal.bias().x, 0.01f, 0.004);
  // A bump after calibration drops it again.
  cal.update({0, 2.0f, 0}, up);
  CHECK(!cal.done());
  CHECK(cal.motion());
  // NaN sample counts as motion.
  for (int i = 0; i < 500; ++i) cal.update({}, up);
  CHECK(cal.done());
  cal.update({NAN, 0, 0}, up);
  CHECK(!cal.done());
}

TEST(mahony_set_bias_seeds_and_clamps) {
  Mahony m;
  Mahony::Config c;
  m.init(c);
  m.set_bias({0.03f, -0.02f, 0.01f});
  CHECK_NEAR(m.bias().x, 0.03f, 1e-7);
  CHECK_NEAR(m.rates({0.03f, -0.02f, 0.01f}).y, 0.0f, 1e-7);
  m.set_bias({1.0f, -1.0f, 0});
  CHECK_NEAR(m.bias().x, c.bias_limit, 1e-7);
  CHECK_NEAR(m.bias().y, -c.bias_limit, 1e-7);
  m.set_bias({NAN, 0, 0});  // ignored
  CHECK_NEAR(m.bias().x, c.bias_limit, 1e-7);
  // Seeded with the true bias, the attitude holds with no drift even with ki = 0.
  Mahony::Config c0;
  c0.ki = 0.0f;
  c0.kp = 0.0f;
  c0.init_time_s = 0.0f;
  Mahony m0;
  m0.init(c0);
  const Vec3 bias{0.04f, -0.03f, 0.02f};
  m0.set_bias(bias);
  for (int i = 0; i < 10000; ++i) m0.update(bias, {0, 0, 9.81f}, 0.001f);
  CHECK(tilt_angle(m0.q()) < 1e-4f);
}

NFTEST_MAIN()
