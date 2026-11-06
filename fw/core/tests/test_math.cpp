#include "nf/math.hpp"
#include "nf/vehicle_params.hpp"
#include "nftest.hpp"

using namespace nf;

TEST(quat_rotation_matches_axis_angle) {
  // 90 deg about +z maps body x to world y (right-handed, B->W).
  Quat q = Quat::from_axis_angle({0, 0, 1}, kPi / 2);
  Vec3 v = q.rotate({1, 0, 0});
  CHECK_NEAR(v.x, 0, 1e-6); CHECK_NEAR(v.y, 1, 1e-6); CHECK_NEAR(v.z, 0, 1e-6);
  Vec3 b = q.rotate_inv(v);
  CHECK_NEAR(b.x, 1, 1e-6); CHECK_NEAR(b.y, 0, 1e-6);
}

TEST(quat_mat_roundtrip_and_euler) {
  for (float r = -1.2f; r < 1.2f; r += 0.4f)
    for (float p = -1.2f; p < 1.2f; p += 0.4f)
      for (float y = -3.0f; y < 3.0f; y += 0.9f) {
        Quat q = Quat::from_euler(r, p, y);
        Quat q2 = Quat::from_mat(q.to_mat());
        float d = std::fabs(q.w * q2.w + q.x * q2.x + q.y * q2.y + q.z * q2.z);
        CHECK_NEAR(d, 1.0f, 1e-5);
        CHECK_NEAR(q.roll(), r, 1e-4);
        CHECK_NEAR(q.pitch(), p, 1e-4);
        CHECK_NEAR(q.yaw(), y, 1e-4);
      }
}

TEST(quat_integration_constant_rate) {
  Quat q;
  Vec3 w{0.3f, -0.2f, 0.5f};
  for (int i = 0; i < 1000; ++i) q = q.integrated(w, 0.001f);
  Quat ref = Quat::from_axis_angle(w, norm(w) * 1.0f);
  float d = std::fabs(q.w * ref.w + q.x * ref.x + q.y * ref.y + q.z * ref.z);
  CHECK_NEAR(d, 1.0f, 1e-5);
}

TEST(tilt_and_mat_ops) {
  CHECK_NEAR(tilt_angle(Quat::from_euler(0.5f, 0, 1.0f)), 0.5f, 1e-5);
  Mat3 R = Quat::from_euler(0.1f, 0.2f, 0.3f).to_mat();
  Mat3 I = R * R.transposed();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) CHECK_NEAR(I(i, j), i == j ? 1.0f : 0.0f, 1e-6);
  Vec3 c = cross({1, 0, 0}, {0, 1, 0});
  CHECK_NEAR(c.z, 1, 0);
  CHECK_NEAR(wrap_pi(3 * kPi), kPi, 1e-5);
}

TEST(vehicle_params_present) {
  CHECK(vehicle::kMass > 0.0f);
  CHECK_EQ(vehicle::kControlRateHz, 500);
  CHECK_EQ(vehicle::kRateLoopHz, 1000);
  CHECK_NEAR(vehicle::kMaxBodyRate[2], 6.0f, 0);
}

NFTEST_MAIN()
