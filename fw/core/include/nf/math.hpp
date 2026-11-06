// Small float32 linear algebra for the flight core. Header-only, no heap.
// Conventions (INTERFACES.md section 1): quaternion q = [w,x,y,z] rotates B->W.
#pragma once

#include <cmath>
#include <cstdint>

namespace nf {

constexpr float kPi = 3.14159265358979f;

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float sq(float v) { return v * v; }
inline bool is_finite(float v) { return std::isfinite(v); }
inline float signf(float v) { return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); }
inline float wrap_pi(float a) {
  while (a > kPi) a -= 2.0f * kPi;
  while (a < -kPi) a += 2.0f * kPi;
  return a;
}

struct Vec3 {
  float x = 0.0f, y = 0.0f, z = 0.0f;
  constexpr Vec3() = default;
  constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator-() const { return {-x, -y, -z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
  Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
  Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
  Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};
inline Vec3 operator*(float s, const Vec3& v) { return v * s; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 hadamard(const Vec3& a, const Vec3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline float norm(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalized(const Vec3& v) {
  float n = norm(v);
  return n > 1e-12f ? v / n : Vec3{};
}
inline bool finite(const Vec3& v) { return is_finite(v.x) && is_finite(v.y) && is_finite(v.z); }
inline Vec3 clamp(const Vec3& v, const Vec3& lim) {
  return {clampf(v.x, -lim.x, lim.x), clampf(v.y, -lim.y, lim.y), clampf(v.z, -lim.z, lim.z)};
}

struct Mat3 {
  float m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  static Mat3 identity() { return Mat3{}; }
  float operator()(int r, int c) const { return m[r][c]; }
  float& operator()(int r, int c) { return m[r][c]; }
  Vec3 col(int c) const { return {m[0][c], m[1][c], m[2][c]}; }
  Vec3 row(int r) const { return {m[r][0], m[r][1], m[r][2]}; }
  Vec3 operator*(const Vec3& v) const {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
  }
  Mat3 operator*(const Mat3& o) const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
    return r;
  }
  Mat3 transposed() const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
    return r;
  }
  // R^T v without forming the transpose.
  Vec3 tmul(const Vec3& v) const { return {dot(col(0), v), dot(col(1), v), dot(col(2), v)}; }
};

struct Quat {
  float w = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f;
  constexpr Quat() = default;
  constexpr Quat(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}
  static Quat identity() { return Quat{}; }
  static Quat from_axis_angle(const Vec3& axis, float angle) {
    Vec3 a = nf::normalized(axis);
    float s = std::sin(0.5f * angle);
    return {std::cos(0.5f * angle), a.x * s, a.y * s, a.z * s};
  }
  // Intrinsic Z-Y-X (yaw, pitch, roll) Euler angles, angles about FLU axes.
  static Quat from_euler(float roll, float pitch, float yaw) {
    return from_axis_angle({0, 0, 1}, yaw) * from_axis_angle({0, 1, 0}, pitch) * from_axis_angle({1, 0, 0}, roll);
  }
  // From an orthonormal rotation matrix (Shepperd's method).
  static Quat from_mat(const Mat3& r) {
    float tr = r.m[0][0] + r.m[1][1] + r.m[2][2];
    Quat q;
    if (tr > 0.0f) {
      float s = std::sqrt(tr + 1.0f) * 2.0f;
      q = {0.25f * s, (r.m[2][1] - r.m[1][2]) / s, (r.m[0][2] - r.m[2][0]) / s, (r.m[1][0] - r.m[0][1]) / s};
    } else if (r.m[0][0] > r.m[1][1] && r.m[0][0] > r.m[2][2]) {
      float s = std::sqrt(1.0f + r.m[0][0] - r.m[1][1] - r.m[2][2]) * 2.0f;
      q = {(r.m[2][1] - r.m[1][2]) / s, 0.25f * s, (r.m[0][1] + r.m[1][0]) / s, (r.m[0][2] + r.m[2][0]) / s};
    } else if (r.m[1][1] > r.m[2][2]) {
      float s = std::sqrt(1.0f + r.m[1][1] - r.m[0][0] - r.m[2][2]) * 2.0f;
      q = {(r.m[0][2] - r.m[2][0]) / s, (r.m[0][1] + r.m[1][0]) / s, 0.25f * s, (r.m[1][2] + r.m[2][1]) / s};
    } else {
      float s = std::sqrt(1.0f + r.m[2][2] - r.m[0][0] - r.m[1][1]) * 2.0f;
      q = {(r.m[1][0] - r.m[0][1]) / s, (r.m[0][2] + r.m[2][0]) / s, (r.m[1][2] + r.m[2][1]) / s, 0.25f * s};
    }
    return q.normalized();
  }
  Vec3 vec() const { return {x, y, z}; }
  Quat operator*(const Quat& b) const {
    return {w * b.w - x * b.x - y * b.y - z * b.z, w * b.x + x * b.w + y * b.z - z * b.y,
            w * b.y - x * b.z + y * b.w + z * b.x, w * b.z + x * b.y - y * b.x + z * b.w};
  }
  Quat conj() const { return {w, -x, -y, -z}; }
  float norm() const { return std::sqrt(w * w + x * x + y * y + z * z); }
  Quat normalized() const {
    float n = norm();
    return n > 1e-12f ? Quat{w / n, x / n, y / n, z / n} : Quat{};
  }
  bool finite() const { return is_finite(w) && is_finite(x) && is_finite(y) && is_finite(z); }
  // Rotation matrix B->W.
  Mat3 to_mat() const {
    Mat3 r;
    float ww = w * w, xx = x * x, yy = y * y, zz = z * z;
    r.m[0][0] = ww + xx - yy - zz; r.m[0][1] = 2 * (x * y - w * z); r.m[0][2] = 2 * (x * z + w * y);
    r.m[1][0] = 2 * (x * y + w * z); r.m[1][1] = ww - xx + yy - zz; r.m[1][2] = 2 * (y * z - w * x);
    r.m[2][0] = 2 * (x * z - w * y); r.m[2][1] = 2 * (y * z + w * x); r.m[2][2] = ww - xx - yy + zz;
    return r;
  }
  Vec3 rotate(const Vec3& v) const { return to_mat() * v; }       // B -> W
  Vec3 rotate_inv(const Vec3& v) const { return to_mat().tmul(v); }  // W -> B
  float roll() const { return std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)); }
  float pitch() const { return std::asin(clampf(2 * (w * y - z * x), -1.0f, 1.0f)); }
  float yaw() const { return std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)); }
  // Integrate body rates omega (rad/s) over dt: q <- q * exp(0.5 omega dt).
  Quat integrated(const Vec3& omega, float dt) const {
    float ang = nf::norm(omega) * dt;
    Quat dq = ang > 1e-9f ? from_axis_angle(omega, ang) : Quat{1, 0.5f * omega.x * dt, 0.5f * omega.y * dt, 0.5f * omega.z * dt};
    return ((*this) * dq).normalized();
  }
};

// Angle (rad) between body z axis and world z axis.
inline float tilt_angle(const Quat& q) {
  float c = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;  // R[2][2]
  return std::acos(clampf(c, -1.0f, 1.0f));
}

}  // namespace nf
