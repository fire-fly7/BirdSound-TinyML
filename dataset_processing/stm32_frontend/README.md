# STM32 CMSIS-DSP MFCC frontend

Add all `.c` and `.h` files in this directory to the STM32 project and enable
the CMSIS-DSP `TransformFunctions` implementation. The application owns the
frontend state and output buffer:

```c
static MfccFrontend frontend;
static float32_t mfcc[
    MFCC_FRONTEND_FRAME_COUNT * MFCC_FRONTEND_MFCC_COUNT
];

MfccFrontend_Init(&frontend);
MfccFrontend_Compute(&frontend, pcm16, mfcc);
```

`pcm16` must contain exactly 16,000 mono PCM16 samples. `mfcc` is flattened in
frame-major order and can be passed to a model with input shape `30 x 13 x 1`.

The generated `MFCC_FRONTEND_CONFIG_SHA256` must match the hash recorded beside
the training dataset and converted model. Regenerate constants after changing
the JSON configuration:

```text
python dataset_processing/generate_frontend_assets.py
```
