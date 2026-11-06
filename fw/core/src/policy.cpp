#include "nf/policy.hpp"

#include "nf/vehicle_params.hpp"

namespace nf {

namespace {
bool layers_chain_ok(uint16_t obs_dim, uint16_t act_dim, uint16_t n, uint16_t (*in)(const void*, int),
                     uint16_t (*out)(const void*, int), const void* layers) {
  if (n == 0 || layers == nullptr) return false;
  uint16_t prev = obs_dim;
  for (int l = 0; l < n; ++l) {
    if (in(layers, l) != prev || out(layers, l) == 0) return false;
    if (in(layers, l) > kMaxLayerWidth || out(layers, l) > kMaxLayerWidth) return false;
    prev = out(layers, l);
  }
  return prev == act_dim;
}
uint16_t f_in(const void* p, int l) { return static_cast<const FloatLayer*>(p)[l].in; }
uint16_t f_out(const void* p, int l) { return static_cast<const FloatLayer*>(p)[l].out; }
uint16_t q_in(const void* p, int l) { return static_cast<const Int8Layer*>(p)[l].in; }
uint16_t q_out(const void* p, int l) { return static_cast<const Int8Layer*>(p)[l].out; }
}  // namespace

bool validate(const FloatModel& m) {
  if (m.obs_dim != kObsDim || m.act_dim != kActDim) return false;
  if (!m.obs_mean || !m.obs_std) return false;
  if (!layers_chain_ok(m.obs_dim, m.act_dim, m.n_layers, f_in, f_out, m.layers)) return false;
  for (int i = 0; i < m.obs_dim; ++i)
    if (!(m.obs_std[i] > 0.0f) || !is_finite(m.obs_mean[i]) || !is_finite(m.obs_std[i])) return false;
  for (int l = 0; l < m.n_layers; ++l)
    if (!m.layers[l].W || !m.layers[l].b) return false;
  return true;
}

bool validate(const Int8Model& m) {
  if (m.obs_dim != kObsDim || m.act_dim != kActDim) return false;
  if (!m.obs_mean || !m.obs_std) return false;
  if (!(m.s_in > 0.0f) || !(m.s_x_last > 0.0f)) return false;
  if (!layers_chain_ok(m.obs_dim, m.act_dim, m.n_layers, q_in, q_out, m.layers)) return false;
  for (int i = 0; i < m.obs_dim; ++i)
    if (!(m.obs_std[i] > 0.0f) || !is_finite(m.obs_mean[i]) || !is_finite(m.obs_std[i])) return false;
  for (int l = 0; l < m.n_layers; ++l) {
    const Int8Layer& L = m.layers[l];
    if (!L.W || !L.b) return false;
    if (l + 1 < m.n_layers) {
      if (!L.M || !L.S || !L.lut) return false;
      for (int j = 0; j < L.out; ++j)
        if (L.S[j] < 1 || L.S[j] > 62) return false;
    } else if (!L.s_w) {
      return false;
    }
  }
  return true;
}

uint32_t crc32_update(uint32_t crc, const void* data, uint32_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  crc = ~crc;
  for (uint32_t i = 0; i < len; ++i) {
    crc ^= p[i];
    for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

// NOTE: hashes raw in-memory bytes; correct on little-endian targets (Cortex-M7, x86).
uint32_t policy_crc32(const FloatModel& m) {
  uint32_t c = 0;
  c = crc32_update(c, m.obs_mean, 4u * m.obs_dim);
  c = crc32_update(c, m.obs_std, 4u * m.obs_dim);
  for (int l = 0; l < m.n_layers; ++l) {
    const FloatLayer& L = m.layers[l];
    c = crc32_update(c, L.W, 4u * L.in * L.out);
    c = crc32_update(c, L.b, 4u * L.out);
  }
  return c;
}

uint32_t policy_crc32(const Int8Model& m) {
  uint32_t c = 0;
  c = crc32_update(c, m.obs_mean, 4u * m.obs_dim);
  c = crc32_update(c, m.obs_std, 4u * m.obs_dim);
  for (int l = 0; l < m.n_layers; ++l) {
    const Int8Layer& L = m.layers[l];
    c = crc32_update(c, L.W, 1u * L.in * L.out);
    c = crc32_update(c, L.b, 4u * L.out);
    if (L.M) c = crc32_update(c, L.M, 4u * L.out);
    if (L.S) c = crc32_update(c, L.S, 1u * L.out);
    if (L.lut) c = crc32_update(c, L.lut, 256u);
    if (L.s_w) c = crc32_update(c, L.s_w, 4u * L.out);
  }
  return c;
}

int64_t rshift_round(int64_t v, int s) {
  int64_t t = v + (static_cast<int64_t>(1) << (s - 1));
  // floor(t / 2^s) without implementation-defined right shift of negatives:
  // for t < 0, ~t = -t-1 >= 0 and floor(t/2^s) = ~((~t) >> s).
  return t >= 0 ? (t >> s) : ~((~t) >> s);
}

int8_t quantize_input(float x) {
  if (!(x == x)) return 0;  // NaN
  float r = std::round(x);   // half away from zero
  r = clampf(r, -127.0f, 127.0f);
  return static_cast<int8_t>(r);
}

bool mlp_forward(const FloatModel& m, const float* obs, float* out) {
  float a[kMaxLayerWidth], b[kMaxLayerWidth];
  for (int i = 0; i < m.obs_dim; ++i) a[i] = (obs[i] - m.obs_mean[i]) / m.obs_std[i];
  float* x = a;
  float* y = b;
  for (int l = 0; l < m.n_layers; ++l) {
    const FloatLayer& L = m.layers[l];
    const float* w = L.W;
    for (int j = 0; j < L.out; ++j, w += L.in) {
      float acc = L.b[j];
      for (int i = 0; i < L.in; ++i) acc += w[i] * x[i];
      y[j] = L.act == Act::Tanh ? std::tanh(acc) : acc;
    }
    float* t = x; x = y; y = t;
  }
  for (int j = 0; j < m.act_dim; ++j) out[j] = x[j];
  return true;
}

bool mlp_forward_int8(const Int8Model& m, const float* obs, float* out, int32_t* last_acc, int8_t* x_q_out) {
  int8_t a[kMaxLayerWidth], b[kMaxLayerWidth];
  for (int i = 0; i < m.obs_dim; ++i) {
    float xn = (obs[i] - m.obs_mean[i]) / m.obs_std[i];
    a[i] = quantize_input(xn / m.s_in);
    if (x_q_out) x_q_out[i] = a[i];
  }
  int8_t* x = a;
  int8_t* y = b;
  for (int l = 0; l < m.n_layers; ++l) {
    const Int8Layer& L = m.layers[l];
    const bool last = (l + 1 == m.n_layers);
    const int8_t* w = L.W;
    for (int j = 0; j < L.out; ++j, w += L.in) {
      // Products are <= 127*128 in magnitude; int32 is the contract. Sum in
      // int32 over at most kMaxLayerWidth terms cannot overflow from the
      // products; the bias add is done in int64 and saturated (a valid
      // quantizer never produces a bias that saturates).
      int32_t dotp = 0;
      for (int i = 0; i < L.in; ++i) dotp += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
      int64_t acc64 = static_cast<int64_t>(dotp) + L.b[j];
      if (acc64 > INT32_MAX) acc64 = INT32_MAX;
      if (acc64 < INT32_MIN) acc64 = INT32_MIN;
      int32_t acc = static_cast<int32_t>(acc64);
      if (last) {
        if (last_acc) last_acc[j] = acc;
        out[j] = std::tanh(static_cast<float>(acc) * m.s_x_last * L.s_w[j]);
      } else {
        int64_t v = rshift_round(static_cast<int64_t>(acc) * L.M[j], L.S[j]);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        y[j] = L.lut[v + 128];
      }
    }
    int8_t* t = x; x = y; y = t;
  }
  return true;
}

void build_obs(const VehicleState& s, const Vec3& tgt, const float prev[kActDim], float obs[kObsDim]) {
  Vec3 pe = clamp(s.pos - tgt, {2.0f, 2.0f, 2.0f});
  Vec3 v = clamp(s.vel, {5.0f, 5.0f, 5.0f});
  Mat3 R = s.q.to_mat();
  obs[0] = pe.x; obs[1] = pe.y; obs[2] = pe.z;
  obs[3] = v.x; obs[4] = v.y; obs[5] = v.z;
  obs[6] = R(0, 0); obs[7] = R(1, 0); obs[8] = R(2, 0);
  obs[9] = R(0, 1); obs[10] = R(1, 1); obs[11] = R(2, 1);
  obs[12] = s.omega.x; obs[13] = s.omega.y; obs[14] = s.omega.z;
  for (int i = 0; i < kActDim; ++i) obs[15 + i] = prev[i];
}

namespace {
float sane(float a) { return is_finite(a) ? clampf(a, -1.0f, 1.0f) : 0.0f; }
}  // namespace

CtbrCommand decode_ctbr(const float a[kActDim]) {
  CtbrCommand c;
  c.thrust_n = 0.5f * (sane(a[0]) + 1.0f) * 4.0f * vehicle::kThrustMax;
  c.rate_sp = {sane(a[1]) * vehicle::kMaxBodyRate[0], sane(a[2]) * vehicle::kMaxBodyRate[1],
               sane(a[3]) * vehicle::kMaxBodyRate[2]};
  return c;
}

void encode_ctbr(const CtbrCommand& c, float a[kActDim]) {
  a[0] = sane(c.thrust_n / (2.0f * vehicle::kThrustMax) - 1.0f);
  a[1] = sane(c.rate_sp.x / vehicle::kMaxBodyRate[0]);
  a[2] = sane(c.rate_sp.y / vehicle::kMaxBodyRate[1]);
  a[3] = sane(c.rate_sp.z / vehicle::kMaxBodyRate[2]);
}

bool PolicyRunner::valid() const {
  if (f32_) return validate(*f32_);
  if (i8_) return validate(*i8_);
  return false;
}

uint32_t PolicyRunner::computed_crc() const {
  if (f32_) return policy_crc32(*f32_);
  if (i8_) return policy_crc32(*i8_);
  return 0;
}

uint32_t PolicyRunner::expected_crc() const {
  if (f32_) return f32_->crc32;
  if (i8_) return i8_->crc32;
  return 0;
}

void PolicyRunner::reset() {
  for (float& p : prev_) p = 0.0f;
  for (float& o : obs_) o = 0.0f;
}

void PolicyRunner::set_prev_action(const float a[kActDim]) {
  for (int i = 0; i < kActDim; ++i) prev_[i] = sane(a[i]);
}

bool PolicyRunner::run(const VehicleState& s, const Vec3& tgt, float action[kActDim]) {
  if (!has_model()) return false;
  build_obs(s, tgt, prev_, obs_);
  bool ok = f32_ ? mlp_forward(*f32_, obs_, action) : mlp_forward_int8(*i8_, obs_, action);
  if (!ok) return false;
  set_prev_action(action);
  return true;
}

}  // namespace nf
