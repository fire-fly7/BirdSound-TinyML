/* Shared MFCC frontend for STM32 projects using CMSIS-DSP. */
#include "mfcc_frontend.h"

#include <math.h>
#include <stddef.h>

#include "mfcc_frontend_data.h"


static void compute_power_spectrum(MfccFrontend *frontend)
{
    uint32_t bin;

    frontend->power_spectrum[0] =
        frontend->fft_output[0] * frontend->fft_output[0];
    frontend->power_spectrum[MFCC_FRONTEND_FFT_BINS - 1U] =
        frontend->fft_output[1] * frontend->fft_output[1];

    for (bin = 1U; bin < MFCC_FRONTEND_FFT_BINS - 1U; ++bin) {
        const float32_t real = frontend->fft_output[2U * bin];
        const float32_t imaginary = frontend->fft_output[2U * bin + 1U];
        frontend->power_spectrum[bin] =
            real * real + imaginary * imaginary;
    }
}


MfccFrontendStatus MfccFrontend_Init(MfccFrontend *frontend)
{
    arm_status status;

    if (frontend == NULL) {
        return MFCC_FRONTEND_INVALID_ARGUMENT;
    }

    frontend->initialized = 0U;
    status = arm_rfft_fast_init_f32(
        &frontend->fft,
        MFCC_FRONTEND_FFT_SIZE
    );
    if (status != ARM_MATH_SUCCESS) {
        return MFCC_FRONTEND_FFT_INIT_FAILED;
    }

    frontend->initialized = 1U;
    return MFCC_FRONTEND_OK;
}


MfccFrontendStatus MfccFrontend_Compute(
    MfccFrontend *frontend,
    const int16_t pcm[MFCC_FRONTEND_CLIP_SAMPLES],
    float32_t output[MFCC_FRONTEND_FRAME_COUNT * MFCC_FRONTEND_MFCC_COUNT]
)
{
    uint32_t frame;
    uint32_t sample;
    uint32_t mel;
    uint32_t bin;
    uint32_t coefficient;
    float32_t global_max_db = -INFINITY;
    const float32_t reference_db =
        10.0f * log10f(MFCC_FRONTEND_DB_REFERENCE);

    if (frontend == NULL || pcm == NULL || output == NULL) {
        return MFCC_FRONTEND_INVALID_ARGUMENT;
    }
    if (frontend->initialized == 0U) {
        return MFCC_FRONTEND_NOT_INITIALIZED;
    }

    for (frame = 0U; frame < MFCC_FRONTEND_FRAME_COUNT; ++frame) {
        const uint32_t frame_start = frame * MFCC_FRONTEND_HOP_LENGTH;

        for (sample = 0U; sample < MFCC_FRONTEND_FFT_SIZE; ++sample) {
            frontend->fft_input[sample] =
                (float32_t)pcm[frame_start + sample]
                * MFCC_FRONTEND_PCM_INT16_SCALE
                * g_mfcc_frontend_window[sample];
        }

        arm_rfft_fast_f32(
            &frontend->fft,
            frontend->fft_input,
            frontend->fft_output,
            0U
        );
        compute_power_spectrum(frontend);

        for (mel = 0U; mel < MFCC_FRONTEND_MEL_COUNT; ++mel) {
            float32_t mel_power = 0.0f;
            float32_t mel_db;
            const float *weights = &g_mfcc_frontend_mel_filterbank[
                mel * MFCC_FRONTEND_FFT_BINS
            ];

            for (bin = 0U; bin < MFCC_FRONTEND_FFT_BINS; ++bin) {
                mel_power += frontend->power_spectrum[bin] * weights[bin];
            }
            if (mel_power < MFCC_FRONTEND_DB_AMIN) {
                mel_power = MFCC_FRONTEND_DB_AMIN;
            }

            mel_db = 10.0f * log10f(mel_power) - reference_db;
            frontend->mel_db[
                frame * MFCC_FRONTEND_MEL_COUNT + mel
            ] = mel_db;
            if (mel_db > global_max_db) {
                global_max_db = mel_db;
            }
        }
    }

    for (frame = 0U; frame < MFCC_FRONTEND_FRAME_COUNT; ++frame) {
        for (mel = 0U; mel < MFCC_FRONTEND_MEL_COUNT; ++mel) {
            float32_t *mel_db = &frontend->mel_db[
                frame * MFCC_FRONTEND_MEL_COUNT + mel
            ];
            const float32_t minimum_db = global_max_db - MFCC_FRONTEND_TOP_DB;
            if (*mel_db < minimum_db) {
                *mel_db = minimum_db;
            }
        }

        for (
            coefficient = 0U;
            coefficient < MFCC_FRONTEND_MFCC_COUNT;
            ++coefficient
        ) {
            float32_t value = 0.0f;
            const float *dct_row = &g_mfcc_frontend_dct_matrix[
                coefficient * MFCC_FRONTEND_MEL_COUNT
            ];

            for (mel = 0U; mel < MFCC_FRONTEND_MEL_COUNT; ++mel) {
                value += frontend->mel_db[
                    frame * MFCC_FRONTEND_MEL_COUNT + mel
                ] * dct_row[mel];
            }
            output[frame * MFCC_FRONTEND_MFCC_COUNT + coefficient] = value;
        }
    }

    return MFCC_FRONTEND_OK;
}


const char *MfccFrontend_ConfigSha256(void)
{
    return MFCC_FRONTEND_CONFIG_SHA256;
}
