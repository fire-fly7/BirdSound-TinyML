/* Shared MFCC frontend for STM32 projects using CMSIS-DSP. */
#ifndef MFCC_FRONTEND_H
#define MFCC_FRONTEND_H

#include <stdint.h>

#include "arm_math.h"
#include "mfcc_frontend_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MFCC_FRONTEND_OK = 0,
    MFCC_FRONTEND_INVALID_ARGUMENT = -1,
    MFCC_FRONTEND_NOT_INITIALIZED = -2,
    MFCC_FRONTEND_FFT_INIT_FAILED = -3
} MfccFrontendStatus;

typedef struct {
    arm_rfft_fast_instance_f32 fft;
    float32_t fft_input[MFCC_FRONTEND_FFT_SIZE];
    float32_t fft_output[MFCC_FRONTEND_FFT_SIZE];
    float32_t power_spectrum[MFCC_FRONTEND_FFT_BINS];
    float32_t mel_db[MFCC_FRONTEND_FRAME_COUNT * MFCC_FRONTEND_MEL_COUNT];
    uint8_t initialized;
} MfccFrontend;

MfccFrontendStatus MfccFrontend_Init(MfccFrontend *frontend);

MfccFrontendStatus MfccFrontend_Compute(
    MfccFrontend *frontend,
    const int16_t pcm[MFCC_FRONTEND_CLIP_SAMPLES],
    float32_t output[MFCC_FRONTEND_FRAME_COUNT * MFCC_FRONTEND_MFCC_COUNT]
);

const char *MfccFrontend_ConfigSha256(void);

#ifdef __cplusplus
}
#endif

#endif
