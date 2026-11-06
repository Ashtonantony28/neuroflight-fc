/* C ABI of the SITL bridge (fw/sitl). Wraps nf::ControlLoop (fw/core) so the
 * Python simulator can fly the real firmware core. Host-only code.
 *
 * One handle holds N independent firmware instances ("envs"), all stepped
 * together by nf_sitl_step() with flat float/int arrays (row-major, env-major).
 * The layouts below are mirrored in fw/sitl/sitl.py; bump NF_SITL_ABI_VERSION
 * on any change.
 */
#ifndef NF_SITL_H_
#define NF_SITL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NF_SITL_ABI_VERSION 1

/* ---- per-env float inputs (NF_IN_F floats) */
enum {
  NF_IN_GYRO = 0,      /* [3] IMU A gyro, rad/s, body FLU */
  NF_IN_ACCEL = 3,     /* [3] IMU A specific force, m/s^2 */
  NF_IN_GYRO_B = 6,    /* [3] IMU B gyro (body frame) */
  NF_IN_ACCEL_B = 9,   /* [3] IMU B specific force */
  NF_IN_FLOW_X = 12,   /* flow, rad/s (FlowSample convention, estimator.hpp) */
  NF_IN_FLOW_Y = 13,
  NF_IN_FLOW_Q = 14,   /* quality [0,1] */
  NF_IN_RANGE = 15,    /* ToF range along -z_B, m */
  NF_IN_MOTOR_HZ = 16, /* [4] motor rotation frequency, Hz */
  NF_IN_VBAT = 20,
  NF_IN_THROTTLE = 21, /* RC throttle [0,1] (arming check only) */
  NF_IN_F = 22
};
/* ---- per-env int inputs (NF_IN_I int32, 0/1) */
enum {
  NF_IN_B_VALID = 0,
  NF_IN_FLOW_VALID = 1,
  NF_IN_RANGE_VALID = 2,
  NF_IN_RPM_VALID = 3,
  NF_IN_VBAT_VALID = 4,
  NF_IN_RC_LINK_OK = 5,
  NF_IN_RC_ARM = 6,
  NF_IN_RC_KILL = 7,
  NF_IN_RC_POLICY_ENABLE = 8,
  NF_IN_RC_REENGAGE = 9,
  NF_IN_I = 10
};
/* ---- per-env float outputs (NF_OUT_F floats) */
enum {
  NF_OUT_CMD = 0,       /* [4] motor commands [0,1], INTERFACES.md section 2 order */
  NF_OUT_Q = 4,         /* [4] estimated attitude w,x,y,z */
  NF_OUT_OMEGA = 8,     /* [3] estimated body rates */
  NF_OUT_POS = 11,      /* [3] estimated position (xy rel. arming point, z height) */
  NF_OUT_VEL = 14,      /* [3] estimated velocity (W) */
  NF_OUT_THRUST = 17,   /* CTBR collective (N) fed to the rate loop */
  NF_OUT_RATE_SP = 18,  /* [3] body-rate setpoint */
  NF_OUT_ACTION = 21,   /* [4] last policy output */
  NF_OUT_F = 25
};
/* ---- per-env uint32 outputs (NF_OUT_U) */
enum {
  NF_OUT_ARMED = 0,
  NF_OUT_MODE = 1,          /* nf::FlightMode */
  NF_OUT_ACTIVE_FAULTS = 2, /* nf::Fault bits */
  NF_OUT_FAULT_HISTORY = 3,
  NF_OUT_FIRST_FAULT = 4,
  NF_OUT_ARM_FAIL = 5,      /* nf::ArmFail bits */
  NF_OUT_INFER_US = 6,
  NF_OUT_FLAGS = 7,         /* bit0 pos_valid, bit1 height_valid, bit2 policy_ok */
  NF_OUT_U = 8
};

typedef struct nf_sitl_config {
  int32_t rate_hz;        /* step() rate, default 1000 */
  int32_t control_hz;     /* policy rate, default 500 */
  float hold_height_m;    /* default target z */
  int32_t rpm_filter;     /* 0/1 */
  float gyro_lpf_hz;
  float accel_lpf_hz;
} nf_sitl_config;

int nf_sitl_abi_version(void);
/* cfg may be NULL (firmware defaults). */
void* nf_sitl_create(int32_t n_envs, const nf_sitl_config* cfg);
void nf_sitl_destroy(void* h);
int32_t nf_sitl_n_envs(void* h);

/* Load a float32 MLP (layout of *.nfpol.json, INTERFACES.md section 6) into env
 * (env < 0: all envs). dims[n_layers+1]; W/b are all layers concatenated
 * (W row-major [out][in]); acts[l] = 0 none / 1 tanh. The CRC is computed
 * from the copied weights (nf::policy_crc32) and returned in *crc_out.
 * Returns 1 if ControlLoop::policy_ok() afterwards (validates + CRC). */
int32_t nf_sitl_set_policy_f32(void* h, int32_t env, int32_t n_layers, const int32_t* dims, const float* mean,
                               const float* std, const float* W, const float* b, const int32_t* acts,
                               uint32_t* crc_out);
void nf_sitl_clear_policy(void* h, int32_t env);
/* Fault injection: overwrite one stored weight (which=0) or bias (which=1)
 * WITHOUT updating the CRC (simulated memory corruption mid-flight). */
int32_t nf_sitl_poke_policy(void* h, int32_t env, int32_t layer, int32_t which, int32_t index, float value);
/* Run only the loaded float model (for numpy cross-checks). Returns 1 on success. */
int32_t nf_sitl_policy_forward(void* h, int32_t env, const float* obs19, float* out4);
void nf_sitl_set_target(void* h, int32_t env, float x, float y, float z, float yaw);
/* Fault injection: simulated policy inference time reported to the
 * supervisor (us); 0 = nominal (inference takes zero simulated time). */
void nf_sitl_set_inference_us(void* h, int32_t env, uint32_t us);

/* Step all envs once at time now_us. in_f[n*NF_IN_F], in_i[n*NF_IN_I],
 * out_f[n*NF_OUT_F], out_u[n*NF_OUT_U]. */
void nf_sitl_step(void* h, uint64_t now_us, const float* in_f, const int32_t* in_i, float* out_f, uint32_t* out_u);

const char* nf_sitl_mode_name(int32_t mode);
const char* nf_sitl_fault_name(uint32_t single_bit);

/* ---- CTBR inner loop in isolation (INTERFACES.md section 9 parity checks).
 * N instances of {decode_ctbr -> nf::RatePid -> nf::Mixer} with default
 * (nominal) configs, stepped exactly like ControlLoop step 6 but without the
 * on-ground integrator reset. */
void* nf_ctbr_create(int32_t n, int32_t rate_hz);
void nf_ctbr_destroy(void* h);
void nf_ctbr_reset(void* h);
/* action[n*4] (CTBR, [-1,1]), gyro[n*3] -> cmd[n*4], saturated[n] (0/1),
 * torque[n*3] (optional, may be NULL). */
void nf_ctbr_step(void* h, const float* action, const float* gyro, float* cmd, int32_t* saturated, float* torque);

#ifdef __cplusplus
}
#endif
#endif /* NF_SITL_H_ */
