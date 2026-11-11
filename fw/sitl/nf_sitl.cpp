// SITL bridge implementation: see nf_sitl.h. Host-only (uses the heap); the
// firmware core underneath is the unmodified fw/core library.
#include "nf_sitl.h"

#include <cstring>
#include <memory>
#include <vector>

#include "nf/control_loop.hpp"

#define NF_API extern "C" __attribute__((visibility("default")))

namespace {
using namespace nf;

struct OwnedModel {
  std::vector<float> mean, stdv;
  std::vector<std::vector<float>> W, b;
  std::vector<FloatLayer> layers;
  FloatModel m{};
};

struct OwnedI8Model {
  std::vector<float> mean, stdv;
  std::vector<std::vector<int8_t>> W, lut;
  std::vector<std::vector<int32_t>> b, M;
  std::vector<std::vector<uint8_t>> S;
  std::vector<std::vector<float>> s_w;
  std::vector<Int8Layer> layers;
  Int8Model m{};
};

struct Env {
  ControlLoop loop;
  std::unique_ptr<OwnedModel> model;
  std::unique_ptr<OwnedI8Model> model_i8;
  InputOverride ovr{};
  uint32_t clk = 0;        // simulated microsecond clock for inference timing
  uint32_t inject_us = 0;  // simulated inference duration
};

// ControlLoopConfig::clock_us is a plain function pointer; route it to the env
// currently being stepped. Each inference calls it twice (t0, t1), so t1 - t0
// equals the injected duration.
thread_local Env* g_env = nullptr;
uint32_t sim_clock_us() {
  Env* e = g_env;
  if (!e) return 0;
  uint32_t r = e->clk;
  e->clk += e->inject_us;
  return r;
}

struct Bank {
  std::vector<std::unique_ptr<Env>> envs;
};

Bank* bank(void* h) { return static_cast<Bank*>(h); }

template <class F>
void for_envs(void* h, int32_t env, F f) {
  Bank* b = bank(h);
  if (!b) return;
  const int32_t n = static_cast<int32_t>(b->envs.size());
  if (env < 0) {
    for (auto& e : b->envs) f(*e);
  } else if (env < n) {
    f(*b->envs[static_cast<size_t>(env)]);
  }
}

Vec3 v3(const float* p) { return {p[0], p[1], p[2]}; }

struct Ctbr {
  RatePid pid;
  Mixer mixer;
};
struct CtbrBank {
  std::vector<Ctbr> v;
};

struct GyroChain {
  BiquadLpfVec3 lpf;
  RpmNotchBank rpm;
};
struct GyroChainBank {
  std::vector<GyroChain> v;
};
}  // namespace

NF_API int nf_sitl_abi_version(void) { return NF_SITL_ABI_VERSION; }

NF_API void* nf_sitl_create(int32_t n, const nf_sitl_config* cfg) {
  if (n <= 0) return nullptr;
  ControlLoopConfig c;
  if (cfg) {
    if (cfg->rate_hz > 0) c.rate_hz = cfg->rate_hz;
    if (cfg->control_hz > 0) c.control_hz = cfg->control_hz;
    if (cfg->hold_height_m > 0.0f) c.hold_height_m = cfg->hold_height_m;
    c.rpm_filter = cfg->rpm_filter != 0;
    if (cfg->gyro_lpf_hz > 0.0f) c.gyro_lpf_hz = cfg->gyro_lpf_hz;
    if (cfg->accel_lpf_hz > 0.0f) c.accel_lpf_hz = cfg->accel_lpf_hz;
    if (cfg->mahony_kf_ref >= 0) c.mahony_kf_ref = cfg->mahony_kf_ref != 0;
    if (cfg->mahony_ref_deadband >= 0.0f) c.mahony_ref_deadband = cfg->mahony_ref_deadband;
  }
  c.clock_us = &sim_clock_us;
  Bank* b = new Bank;
  b->envs.reserve(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    b->envs.emplace_back(new Env);
    b->envs.back()->loop.init(c);
  }
  return b;
}

NF_API void nf_sitl_destroy(void* h) { delete bank(h); }

NF_API int32_t nf_sitl_n_envs(void* h) { return h ? static_cast<int32_t>(bank(h)->envs.size()) : 0; }

NF_API int32_t nf_sitl_set_policy_f32(void* h, int32_t env, int32_t n_layers, const int32_t* dims,
                                      const float* mean, const float* stdv, const float* W, const float* b,
                                      const int32_t* acts, uint32_t* crc_out) {
  if (!h || n_layers <= 0 || !dims || !mean || !stdv || !W || !b || !acts) return 0;
  int32_t all_ok = 1;
  for_envs(h, env, [&](Env& e) {
    auto m = std::make_unique<OwnedModel>();
    m->mean.assign(mean, mean + dims[0]);
    m->stdv.assign(stdv, stdv + dims[0]);
    const float* wp = W;
    const float* bp = b;
    m->W.resize(static_cast<size_t>(n_layers));
    m->b.resize(static_cast<size_t>(n_layers));
    for (int32_t l = 0; l < n_layers; ++l) {
      const size_t in = static_cast<size_t>(dims[l]), out = static_cast<size_t>(dims[l + 1]);
      m->W[static_cast<size_t>(l)].assign(wp, wp + in * out);
      m->b[static_cast<size_t>(l)].assign(bp, bp + out);
      wp += in * out;
      bp += out;
    }
    for (int32_t l = 0; l < n_layers; ++l) {
      FloatLayer L{static_cast<uint16_t>(dims[l]), static_cast<uint16_t>(dims[l + 1]),
                   m->W[static_cast<size_t>(l)].data(), m->b[static_cast<size_t>(l)].data(),
                   acts[l] ? Act::Tanh : Act::None};
      m->layers.push_back(L);
    }
    m->m = FloatModel{static_cast<uint16_t>(dims[0]), static_cast<uint16_t>(dims[n_layers]),
                      static_cast<uint16_t>(n_layers), m->mean.data(), m->stdv.data(), m->layers.data(), 0};
    m->m.crc32 = policy_crc32(m->m);
    if (crc_out) *crc_out = m->m.crc32;
    e.model = std::move(m);
    e.model_i8.reset();
    e.loop.set_policy(&e.model->m);
    if (!e.loop.policy_ok()) all_ok = 0;
  });
  return all_ok;
}

NF_API void nf_sitl_clear_policy(void* h, int32_t env) {
  for_envs(h, env, [](Env& e) {
    e.loop.set_policy(static_cast<const FloatModel*>(nullptr));
    e.model.reset();
    e.model_i8.reset();
  });
}

NF_API int32_t nf_sitl_set_policy_i8(void* h, int32_t env, int32_t n_layers, const int32_t* dims, const float* mean,
                                     const float* stdv, float s_in, const int8_t* Wq, const int32_t* bq,
                                     const float* s_w, const int32_t* M, const int32_t* S, const int8_t* lut,
                                     const float* s_x, const float* s_pre, uint32_t* crc_out) {
  (void)s_pre;  // folded into M/S/lut by the quantizer
  if (!h || n_layers <= 0 || !dims || !mean || !stdv || !Wq || !bq || !s_w || !s_x) return 0;
  int32_t all_ok = 1;
  for_envs(h, env, [&](Env& e) {
    auto m = std::make_unique<OwnedI8Model>();
    m->mean.assign(mean, mean + dims[0]);
    m->stdv.assign(stdv, stdv + dims[0]);
    const size_t L = static_cast<size_t>(n_layers);
    m->W.resize(L); m->b.resize(L); m->M.resize(L); m->S.resize(L); m->lut.resize(L); m->s_w.resize(L);
    const int8_t* wp = Wq;
    const int32_t* bp = bq;
    const float* sp = s_w;
    const int32_t* mp = M;
    const int32_t* shp = S;
    const int8_t* lp = lut;
    for (size_t l = 0; l < L; ++l) {
      const size_t in = static_cast<size_t>(dims[l]), out = static_cast<size_t>(dims[l + 1]);
      m->W[l].assign(wp, wp + in * out);
      m->b[l].assign(bp, bp + out);
      wp += in * out;
      bp += out;
      if (l + 1 < L) {
        m->M[l].assign(mp, mp + out);
        for (size_t j = 0; j < out; ++j) m->S[l].push_back(static_cast<uint8_t>(shp[j]));
        m->lut[l].assign(lp, lp + 256);
        mp += out;
        shp += out;
        lp += 256;
      } else {
        m->s_w[l].assign(sp, sp + out);
      }
      sp += out;
    }
    for (size_t l = 0; l < L; ++l) {
      const bool last = l + 1 == L;
      Int8Layer IL{static_cast<uint16_t>(dims[l]), static_cast<uint16_t>(dims[l + 1]), m->W[l].data(),
                   m->b[l].data(), last ? nullptr : m->M[l].data(), last ? nullptr : m->S[l].data(),
                   last ? nullptr : m->lut[l].data(), last ? m->s_w[l].data() : nullptr};
      m->layers.push_back(IL);
    }
    m->m = Int8Model{static_cast<uint16_t>(dims[0]), static_cast<uint16_t>(dims[n_layers]),
                     static_cast<uint16_t>(n_layers), m->mean.data(), m->stdv.data(), m->layers.data(),
                     s_in, s_x[n_layers - 1], 0};
    m->m.crc32 = policy_crc32(m->m);
    if (crc_out) *crc_out = m->m.crc32;
    e.model_i8 = std::move(m);
    e.model.reset();
    e.loop.set_policy(&e.model_i8->m);
    if (!e.loop.policy_ok()) all_ok = 0;
  });
  return all_ok;
}

NF_API void nf_sitl_set_overrides(void* h, const uint32_t* mask, const float* vals) {
  Bank* b = bank(h);
  if (!b) return;
  for (size_t k = 0; k < b->envs.size(); ++k) {
    Env& e = *b->envs[k];
    const float* v = vals + k * NF_OVR_F;
    e.ovr.mask = mask[k];
    e.ovr.q = Quat{v[0], v[1], v[2], v[3]};
    e.ovr.omega_policy = v3(v + 4);
    e.ovr.omega_pid = v3(v + 7);
    e.ovr.pos = v3(v + 10);
    e.ovr.vel = v3(v + 13);
    e.loop.set_input_override(e.ovr.mask ? &e.ovr : nullptr);
  }
}

NF_API int32_t nf_sitl_poke_policy(void* h, int32_t env, int32_t layer, int32_t which, int32_t index, float value) {
  int32_t ok = 0;
  for_envs(h, env, [&](Env& e) {
    if (!e.model || layer < 0 || layer >= static_cast<int32_t>(e.model->W.size()) || index < 0) return;
    auto& vec = which == 0 ? e.model->W[static_cast<size_t>(layer)] : e.model->b[static_cast<size_t>(layer)];
    if (static_cast<size_t>(index) >= vec.size()) return;
    vec[static_cast<size_t>(index)] = value;  // CRC deliberately not updated
    ok = 1;
  });
  return ok;
}

NF_API int32_t nf_sitl_policy_forward(void* h, int32_t env, const float* obs, float* out) {
  int32_t ok = 0;
  for_envs(h, env, [&](Env& e) {
    if (e.model) ok = mlp_forward(e.model->m, obs, out) ? 1 : 0;
    else if (e.model_i8) ok = mlp_forward_int8(e.model_i8->m, obs, out) ? 1 : 0;
  });
  return ok;
}

NF_API void nf_sitl_set_target(void* h, int32_t env, float x, float y, float z, float yaw) {
  for_envs(h, env, [&](Env& e) { e.loop.set_target({x, y, z}, yaw); });
}

NF_API void nf_sitl_set_inference_us(void* h, int32_t env, uint32_t us) {
  for_envs(h, env, [&](Env& e) { e.inject_us = us; });
}

NF_API void nf_sitl_step(void* h, uint64_t now_us, const float* in_f, const int32_t* in_i, float* out_f,
                         uint32_t* out_u) {
  Bank* b = bank(h);
  if (!b) return;
  const size_t n = b->envs.size();
  for (size_t k = 0; k < n; ++k) {
    Env& e = *b->envs[k];
    const float* f = in_f + k * NF_IN_F;
    const int32_t* iv = in_i + k * NF_IN_I;
    ImuSample imu;
    imu.gyro = v3(f + NF_IN_GYRO);
    imu.accel = v3(f + NF_IN_ACCEL);
    imu.gyro_b = v3(f + NF_IN_GYRO_B);
    imu.accel_b = v3(f + NF_IN_ACCEL_B);
    imu.b_valid = iv[NF_IN_B_VALID] != 0;
    SensorSample s;
    s.flow.flow_x = f[NF_IN_FLOW_X];
    s.flow.flow_y = f[NF_IN_FLOW_Y];
    s.flow.quality = f[NF_IN_FLOW_Q];
    s.flow.valid = iv[NF_IN_FLOW_VALID] != 0;
    s.range.range_m = f[NF_IN_RANGE];
    s.range.valid = iv[NF_IN_RANGE_VALID] != 0;
    for (int i = 0; i < 4; ++i) s.motor_hz[i] = f[NF_IN_MOTOR_HZ + i];
    s.rpm_valid = iv[NF_IN_RPM_VALID] != 0;
    s.vbat = f[NF_IN_VBAT];
    s.vbat_valid = iv[NF_IN_VBAT_VALID] != 0;
    RcInput rc;
    rc.link_ok = iv[NF_IN_RC_LINK_OK] != 0;
    rc.arm = iv[NF_IN_RC_ARM] != 0;
    rc.kill = iv[NF_IN_RC_KILL] != 0;
    rc.policy_enable = iv[NF_IN_RC_POLICY_ENABLE] != 0;
    rc.reengage = iv[NF_IN_RC_REENGAGE] != 0;
    rc.throttle = f[NF_IN_THROTTLE];

    g_env = &e;
    MotorOutput o = e.loop.step(imu, s, rc, now_us);
    g_env = nullptr;

    float* of = out_f + k * NF_OUT_F;
    uint32_t* ou = out_u + k * NF_OUT_U;
    for (int i = 0; i < 4; ++i) of[NF_OUT_CMD + i] = o.cmd[i];
    const VehicleState& st = e.loop.state();
    of[NF_OUT_Q + 0] = st.q.w;
    of[NF_OUT_Q + 1] = st.q.x;
    of[NF_OUT_Q + 2] = st.q.y;
    of[NF_OUT_Q + 3] = st.q.z;
    for (int i = 0; i < 3; ++i) {
      of[NF_OUT_OMEGA + i] = st.omega[i];
      of[NF_OUT_POS + i] = st.pos[i];
      of[NF_OUT_VEL + i] = st.vel[i];
      of[NF_OUT_RATE_SP + i] = e.loop.last_ctbr().rate_sp[i];
    }
    of[NF_OUT_THRUST] = e.loop.last_ctbr().thrust_n;
    for (int i = 0; i < 4; ++i) of[NF_OUT_ACTION + i] = e.loop.last_action()[i];
    of[NF_OUT_ACC_INNOV] = e.loop.attitude().innovation();
    of[NF_OUT_KF_NIS] = e.loop.translation().nis();
    {
      const Vec3 ar = e.loop.accel_ref();
      for (int i = 0; i < 3; ++i) of[NF_OUT_ACC_REF + i] = ar[i];
    }
    ou[NF_OUT_ARMED] = o.armed ? 1u : 0u;
    ou[NF_OUT_MODE] = static_cast<uint32_t>(o.mode);
    ou[NF_OUT_ACTIVE_FAULTS] = o.active_faults;
    ou[NF_OUT_FAULT_HISTORY] = o.fault_history;
    ou[NF_OUT_FIRST_FAULT] = o.first_fault;
    ou[NF_OUT_ARM_FAIL] = o.arming_failures;
    ou[NF_OUT_INFER_US] = e.loop.last_inference_us();
    ou[NF_OUT_FLAGS] = (st.pos_valid ? 1u : 0u) | (st.height_valid ? 2u : 0u) | (e.loop.policy_ok() ? 4u : 0u);
  }
}

NF_API const char* nf_sitl_mode_name(int32_t m) { return to_string(static_cast<FlightMode>(m)); }
NF_API const char* nf_sitl_fault_name(uint32_t bit) { return fault_name(bit); }

// ------------------------------------------------------------------ CTBR parity
NF_API void* nf_ctbr_create(int32_t n, int32_t rate_hz) {
  if (n <= 0) return nullptr;
  CtbrBank* b = new CtbrBank;
  b->v.resize(static_cast<size_t>(n));
  const float fs = static_cast<float>(rate_hz > 0 ? rate_hz : vehicle::kRateLoopHz);
  for (auto& c : b->v) {
    c.pid.init(RatePidConfig{}, fs);
    c.mixer.init(MixerParams{});
  }
  return b;
}

NF_API void nf_ctbr_destroy(void* h) { delete static_cast<CtbrBank*>(h); }

NF_API void nf_ctbr_reset(void* h) {
  if (!h) return;
  for (auto& c : static_cast<CtbrBank*>(h)->v) c.pid.reset();
}

NF_API void nf_ctbr_step(void* h, const float* action, const float* gyro, float* cmd, int32_t* sat, float* torque) {
  if (!h) return;
  auto& v = static_cast<CtbrBank*>(h)->v;
  for (size_t k = 0; k < v.size(); ++k) {
    CtbrCommand c = decode_ctbr(action + 4 * k);
    Vec3 tau = v[k].pid.update(c.rate_sp, v3(gyro + 3 * k));
    MixerResult r = v[k].mixer.mix(c.thrust_n, tau);
    v[k].pid.integrate(r.any_saturated());
    for (int i = 0; i < 4; ++i) cmd[4 * k + i] = r.cmd[i];
    if (sat) sat[k] = r.any_saturated() ? 1 : 0;
    if (torque)
      for (int i = 0; i < 3; ++i) torque[3 * k + i] = tau[i];
  }
}

// ------------------------------------------------------------------ gyro filter parity
NF_API void* nf_gfilt_create(int32_t n, int32_t rate_hz) {
  if (n <= 0) return nullptr;
  GyroChainBank* b = new GyroChainBank;
  b->v.resize(static_cast<size_t>(n));
  const ControlLoopConfig c{};
  const float fs = static_cast<float>(rate_hz > 0 ? rate_hz : c.rate_hz);
  for (auto& g : b->v) {
    g.lpf.init(c.gyro_lpf_hz, fs);
    g.rpm.init(fs);
  }
  return b;
}

NF_API void nf_gfilt_destroy(void* h) { delete static_cast<GyroChainBank*>(h); }

NF_API void nf_gfilt_step(void* h, const float* gyro, const float* motor_hz, float* out) {
  if (!h) return;
  auto& v = static_cast<GyroChainBank*>(h)->v;
  for (size_t k = 0; k < v.size(); ++k) {
    v[k].rpm.update(motor_hz + 4 * k);  // ControlLoop::step order: rpm update, LPF, notches
    Vec3 g = v[k].lpf.apply(v3(gyro + 3 * k));
    g = v[k].rpm.apply(g);
    for (int i = 0; i < 3; ++i) out[3 * k + i] = g[i];
  }
}

// ---------------------------------------------------------------- Mahony in isolation
namespace {
struct MahonyBank {
  std::vector<Mahony> v;
};
}  // namespace

NF_API void* nf_mahony_create(int32_t n) {
  if (n <= 0) return nullptr;
  MahonyBank* b = new MahonyBank;
  b->v.resize(static_cast<size_t>(n));
  const ControlLoopConfig c{};
  for (auto& m : b->v) m.init(c.mahony);
  return b;
}

NF_API void nf_mahony_destroy(void* h) { delete static_cast<MahonyBank*>(h); }

NF_API void nf_mahony_reset(void* h, const float* q, const float* bias, float t_s) {
  if (!h) return;
  auto& v = static_cast<MahonyBank*>(h)->v;
  for (size_t k = 0; k < v.size(); ++k) {
    v[k].reset(q ? Quat{q[4 * k], q[4 * k + 1], q[4 * k + 2], q[4 * k + 3]} : Quat{}, t_s);
    if (bias) v[k].set_bias(v3(bias + 3 * k));
  }
}

NF_API void nf_mahony_step(void* h, const float* gyro, const float* accel, const float* sf_ref, float dt,
                           float* q_out, float* bias_out, float* innov_out) {
  if (!h) return;
  auto& v = static_cast<MahonyBank*>(h)->v;
  for (size_t k = 0; k < v.size(); ++k) {
    if (sf_ref) v[k].update(v3(gyro + 3 * k), v3(accel + 3 * k), dt, v3(sf_ref + 3 * k));
    else v[k].update(v3(gyro + 3 * k), v3(accel + 3 * k), dt);
    const Quat& q = v[k].q();
    if (q_out) { q_out[4 * k] = q.w; q_out[4 * k + 1] = q.x; q_out[4 * k + 2] = q.y; q_out[4 * k + 3] = q.z; }
    if (bias_out) for (int i = 0; i < 3; ++i) bias_out[3 * k + i] = v[k].bias()[i];
    if (innov_out) innov_out[k] = v[k].innovation();
  }
}
