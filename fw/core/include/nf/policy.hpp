// Policy runtime: float32 MLP and int8 MLP (INTERFACES.md sections 4-7).
//
// ============================ CODEGEN CONTRACT ==============================
// The codegen (tools/) emits plain `const` arrays plus one model struct. No
// constructors, no heap; everything is aggregate-initialisable at namespace
// scope so it lands in flash. Matrices are row-major [out][in] (same as the
// `W: [out][in]` in *.nfpol.json), flattened: W[j*in + i].
//
//   // float32 model
//   static const float kMean[19] = {...}, kStd[19] = {...};
//   static const float kW0[64*19] = {...}, kB0[64] = {...};
//   static const float kW1[64*64] = {...}, kB1[64] = {...};
//   static const float kW2[4*64]  = {...}, kB2[4]  = {...};
//   static const nf::FloatLayer kLayers[3] = {
//     {19, 64, kW0, kB0, nf::Act::Tanh}, {64, 64, kW1, kB1, nf::Act::Tanh},
//     {64, 4, kW2, kB2, nf::Act::Tanh}};
//   const nf::FloatModel kPolicyF32 = {19, 4, 3, kMean, kStd, kLayers, 0xCRC32u};
//
//   // int8 model (section 7). Hidden layers: W, b, M, S, lut (s_w unused ->
//   // nullptr). Last layer: W, b, s_w (M, S, lut = nullptr).
//   static const nf::Int8Layer kQLayers[3] = {
//     {19, 64, kQW0, kQB0, kM0, kS0, kLut0, nullptr},
//     {64, 64, kQW1, kQB1, kM1, kS1, kLut1, nullptr},
//     {64, 4,  kQW2, kQB2, nullptr, nullptr, nullptr, kSw2}};
//   const nf::Int8Model kPolicyI8 = {19, 4, 3, kMean, kStd, kQLayers,
//                                    /*s_in=*/..f, /*s_x_last=*/..f, 0xCRC32u};
//
// s_x_last is the activation scale feeding the last layer: 1/127 for a
// network with hidden layers (emit the float32 literal the golden model
// uses), s_in for a single-layer net.
//
// CRC32 (zlib/IEEE, `zlib.crc32`) covers the little-endian bytes of, in
// order: obs_mean, obs_std, then for each layer: W, b, and for int8 also
// M, S, lut (hidden) / s_w (last), skipping null pointers. Layer sizes
// are not hashed. `nf::policy_crc32()` recomputes it for the arming check.
// ============================================================================
#pragma once

#include <cstdint>

#include "nf/control.hpp"
#include "nf/math.hpp"

namespace nf {

constexpr int kObsDim = 19;
constexpr int kActDim = 4;
constexpr int kMaxLayerWidth = 128;  // scratch buffer size (per layer in/out)

enum class Act : uint8_t { None = 0, Tanh = 1 };

struct FloatLayer {
  uint16_t in;
  uint16_t out;
  const float* W;  // [out*in], row-major
  const float* b;  // [out]
  Act act;
};

struct FloatModel {
  uint16_t obs_dim;
  uint16_t act_dim;
  uint16_t n_layers;
  const float* obs_mean;  // [obs_dim]
  const float* obs_std;   // [obs_dim]
  const FloatLayer* layers;
  uint32_t crc32;
};

struct Int8Layer {
  uint16_t in;
  uint16_t out;
  const int8_t* W;    // [out*in] row-major, symmetric, zero point 0
  const int32_t* b;   // [out] b_q
  const int32_t* M;   // [out] requant multiplier (hidden layers)
  const uint8_t* S;   // [out] requant shift in [1,62] (hidden layers)
  const int8_t* lut;  // [256] tanh LUT indexed by y+128 (hidden layers)
  const float* s_w;   // [out] per-channel weight scale (last layer)
};

struct Int8Model {
  uint16_t obs_dim;
  uint16_t act_dim;
  uint16_t n_layers;
  const float* obs_mean;
  const float* obs_std;
  const Int8Layer* layers;
  float s_in;      // input quantization scale
  float s_x_last;  // activation scale feeding the last layer
  uint32_t crc32;
};

// Structural validation (sizes chain, widths <= kMaxLayerWidth, non-null
// pointers, S in [1,62]). Call once at init.
bool validate(const FloatModel& m);
bool validate(const Int8Model& m);
uint32_t crc32_update(uint32_t crc, const void* data, uint32_t len);
uint32_t policy_crc32(const FloatModel& m);
uint32_t policy_crc32(const Int8Model& m);

// ---- int8 primitives (exposed for bit-exact tests) -------------------------
// (v + 2^(s-1)) >> s with arithmetic (floor) shift, s >= 1. Implemented
// without relying on implementation-defined signed right shift.
int64_t rshift_round(int64_t v, int s);
// round-half-away-from-zero then clamp to [-127,127]; NaN -> 0.
int8_t quantize_input(float xn_over_s);

// ---- inference -------------------------------------------------------------
// out[act_dim] = model(obs). Returns false if the model is invalid.
bool mlp_forward(const FloatModel& m, const float* obs, float* out);
// int8 path. last_acc (optional, [act_dim]) receives the last-layer int32
// accumulators (bias included) for bit-exact comparison with the golden model.
// x_q_out (optional, [obs_dim]) receives the quantized input.
bool mlp_forward_int8(const Int8Model& m, const float* obs, float* out, int32_t* last_acc = nullptr,
                      int8_t* x_q_out = nullptr);

// ---- observation / action glue (sections 4-5) ------------------------------
void build_obs(const VehicleState& s, const Vec3& pos_target, const float prev_action[kActDim],
               float obs[kObsDim]);
// Non-finite entries decode as 0; entries are clamped to [-1,1].
CtbrCommand decode_ctbr(const float action[kActDim]);
// Inverse of decode_ctbr (clamped) = sim/ctbr.py::ctbr_to_action. Note: on
// (re)engage the previous action is ZEROS (INTERFACES.md section 10), not this.
void encode_ctbr(const CtbrCommand& c, float action[kActDim]);

// Wraps either model, keeps the previous action, checks output sanity.
class PolicyRunner {
 public:
  void set_model(const FloatModel* f) { f32_ = f; i8_ = nullptr; reset(); }
  void set_model(const Int8Model* q) { i8_ = q; f32_ = nullptr; reset(); }
  bool has_model() const { return f32_ != nullptr || i8_ != nullptr; }
  bool valid() const;
  uint32_t computed_crc() const;
  uint32_t expected_crc() const;
  void reset();
  // Runs one inference. Returns false if no/invalid model. The raw output is
  // reported as-is (may be NaN) so the supervisor can judge it.
  bool run(const VehicleState& s, const Vec3& pos_target, float action[kActDim]);
  const float* prev_action() const { return prev_; }
  void set_prev_action(const float a[kActDim]);
  const float* last_obs() const { return obs_; }

 private:
  const FloatModel* f32_ = nullptr;
  const Int8Model* i8_ = nullptr;
  float prev_[kActDim] = {0, 0, 0, 0};
  float obs_[kObsDim] = {};
};

}  // namespace nf
