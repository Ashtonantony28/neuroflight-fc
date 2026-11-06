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

NFTEST_MAIN()
