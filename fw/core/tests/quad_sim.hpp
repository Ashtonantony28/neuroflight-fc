// Tiny rigid-body quad simulator for closed-loop tests (test code only).
// Follows INTERFACES.md: ENU/FLU, motor order/spin, thrust curve, motor lag.
#pragma once
#include "nf/control_loop.hpp"
#include "nf/math.hpp"
#include "nf/vehicle_params.hpp"

namespace nftest {
using namespace nf;

struct QuadSim {
  Vec3 p{0, 0, 1}, v{}, omega{};
  Quat q{};
  float w[4] = {0, 0, 0, 0};
  Vec3 last_acc_w{};
  float t = 0.0f;
  bool ground = true;  // ground plane at z = 0
  Vec3 ext_torque{};   // disturbance (N m)
  Vec3 ext_acc_w{};    // disturbance acceleration (m/s^2, world; seen by the accelerometer)
  uint32_t rng = 12345;
  float gyro_noise = 0.0f, acc_noise = 0.0f;
  float thrust_scale = 1.0f;  // actual / nominal thrust (weak or strong motors)

  float randn() {  // cheap approx normal
    float s = 0;
    for (int i = 0; i < 4; ++i) {
      rng = rng * 1664525u + 1013904223u;
      s += static_cast<float>(rng >> 8) / 16777216.0f - 0.5f;
    }
    return s * 1.732f;
  }

  float thrust_of(float wi) const {
    const float a = vehicle::kThrustCurveA;
    return vehicle::kThrustMax * (a * wi * wi + (1 - a) * wi);
  }

  void step(const float cmd[4], float dt) {
    const int sub = 4;
    const float h = dt / sub;
    for (int s = 0; s < sub; ++s) {
      float T[4];
      for (int i = 0; i < 4; ++i) {
        float c = clampf(cmd[i], 0.0f, 1.0f);
        float tau = c > w[i] ? vehicle::kMotorTauUp : vehicle::kMotorTauDown;
        w[i] += (c - w[i]) * (h / tau);
        T[i] = thrust_scale * thrust_of(w[i]);
      }
      const float a = vehicle::kArmXy, k = vehicle::kYawTorqueCoeff;
      Vec3 tau{a * (T[0] - T[1] - T[2] + T[3]), a * (-T[0] - T[1] + T[2] + T[3]), k * (T[0] - T[1] + T[2] - T[3])};
      tau += ext_torque;
      Vec3 J{vehicle::kInertia[0], vehicle::kInertia[1], vehicle::kInertia[2]};
      Vec3 Jw = hadamard(J, omega);
      Vec3 rd{vehicle::kRotDrag[0], vehicle::kRotDrag[1], vehicle::kRotDrag[2]};
      Vec3 wdot = tau - cross(omega, Jw) - hadamard(rd, omega);
      wdot = {wdot.x / J.x, wdot.y / J.y, wdot.z / J.z};
      omega += wdot * h;
      q = q.integrated(omega, h);
      float F = T[0] + T[1] + T[2] + T[3];
      Vec3 vb = q.rotate_inv(v);
      Vec3 ld{vehicle::kLinDrag[0], vehicle::kLinDrag[1], vehicle::kLinDrag[2]};
      Vec3 fb = Vec3{0, 0, F} - hadamard(ld, vb);
      Vec3 acc = q.rotate(fb) / vehicle::kMass + Vec3{0, 0, -vehicle::kGravity} + ext_acc_w;
      v += acc * h;
      p += v * h;
      if (ground && p.z <= 0.0f) {
        p.z = 0.0f;
        if (v.z < 0) v.z = 0;
        v.x *= 0.9f; v.y *= 0.9f;
        acc = {0, 0, 0};
      }
      last_acc_w = acc;
    }
    t += dt;
  }

  ImuSample imu() {
    ImuSample s;
    Vec3 f_w = last_acc_w + Vec3{0, 0, vehicle::kGravity};  // specific force (W)
    s.gyro = omega + Vec3{randn(), randn(), randn()} * gyro_noise;
    s.accel = q.rotate_inv(f_w) + Vec3{randn(), randn(), randn()} * acc_noise;
    s.gyro_b = s.gyro;
    s.accel_b = s.accel;
    s.b_valid = true;
    return s;
  }

  SensorSample sensors(bool new_flow = true) const {
    SensorSample s;
    Mat3 R = q.to_mat();
    float hz = p.z;
    if (new_flow && R(2, 2) > 0.3f) {
      s.range.valid = true;
      s.range.range_m = hz / R(2, 2);
      float hb = s.range.range_m;
      Vec3 vb = q.rotate_inv(v);
      if (hb > 0.05f) {
        s.flow.valid = true;
        s.flow.quality = 1.0f;
        s.flow.flow_x = vb.x / hb - omega.y;
        s.flow.flow_y = vb.y / hb + omega.x;
      }
    }
    for (int i = 0; i < 4; ++i) s.motor_hz[i] = 400.0f * w[i];
    s.rpm_valid = true;
    return s;
  }
};
}  // namespace nftest
