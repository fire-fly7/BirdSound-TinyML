"""训练端和STM32端共享的MFCC声学前端定义。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CONFIG_PATH = Path(__file__).resolve().with_name(
    "acoustic_frontend_config.json"
)


@dataclass(frozen=True)
class AcousticFrontendConfig:
    schema_version: int
    feature_type: str
    sample_rate: int
    clip_duration_seconds: float
    clip_sample_count: int
    pcm_int16_scale: float
    n_fft: int
    hop_length: int
    center: bool
    window: str
    spectrum: str
    n_fft_bins: int
    n_mels: int
    fmin_hz: float
    fmax_hz: float
    mel_scale: str
    mel_norm: str
    power_to_db_reference: float
    power_to_db_amin: float
    power_to_db_top_db: float
    dct_type: int
    dct_norm: str
    n_mfcc: int
    include_coefficient_zero: bool
    number_of_frames: int
    model_input_shape: tuple
    model_input_layout: str


def canonical_config_json(config_data):
    return json.dumps(
        config_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def config_sha256(config_data):
    content = canonical_config_json(config_data).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def load_config_data(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config):
    if config.feature_type != "MFCC":
        raise ValueError("声学前端只支持MFCC。")
    if config.center:
        raise ValueError("嵌入式统一前端必须使用center=false。")
    if config.window != "hann_periodic":
        raise ValueError("统一前端必须使用周期Hann窗。")
    if config.spectrum != "power":
        raise ValueError("统一前端必须使用功率谱。")
    if config.mel_scale != "slaney" or config.mel_norm != "slaney":
        raise ValueError("统一前端必须使用Slaney Mel刻度和归一化。")
    if config.dct_type != 2 or config.dct_norm != "ortho":
        raise ValueError("统一前端必须使用正交归一化DCT-II。")
    if config.n_fft_bins != config.n_fft // 2 + 1:
        raise ValueError("n_fft_bins与n_fft不一致。")

    expected_samples = int(
        round(config.sample_rate * config.clip_duration_seconds)
    )
    if config.clip_sample_count != expected_samples:
        raise ValueError("clip_sample_count与采样率、时长不一致。")

    expected_frames = 1 + (
        config.clip_sample_count - config.n_fft
    ) // config.hop_length
    if config.number_of_frames != expected_frames:
        raise ValueError("number_of_frames与FFT和步长不一致。")

    expected_shape = (
        config.number_of_frames,
        config.n_mfcc,
        1,
    )
    if tuple(config.model_input_shape) != expected_shape:
        raise ValueError("model_input_shape与MFCC输出不一致。")
    if config.fmin_hz < 0 or config.fmax_hz > config.sample_rate / 2:
        raise ValueError("Mel滤波器频率范围无效。")
    if config.n_mfcc > config.n_mels:
        raise ValueError("n_mfcc不能大于n_mels。")


def load_config(path=CONFIG_PATH):
    data = load_config_data(path)
    config = AcousticFrontendConfig(
        **{
            **data,
            "model_input_shape": tuple(data["model_input_shape"]),
        }
    )
    validate_config(config)
    return config


def hz_to_slaney_mel(frequencies):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    frequency_step = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / frequency_step
    log_step = np.log(6.4) / 27.0

    mels = frequencies / frequency_step
    logarithmic = frequencies >= min_log_hz
    mels[logarithmic] = min_log_mel + np.log(
        frequencies[logarithmic] / min_log_hz
    ) / log_step
    return mels


def slaney_mel_to_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    frequency_step = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / frequency_step
    log_step = np.log(6.4) / 27.0

    frequencies = frequency_step * mels
    logarithmic = mels >= min_log_mel
    frequencies[logarithmic] = min_log_hz * np.exp(
        log_step * (mels[logarithmic] - min_log_mel)
    )
    return frequencies


def make_periodic_hann(config):
    sample_indices = np.arange(config.n_fft, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(
        2 * np.pi * sample_indices / config.n_fft
    )
    return window.astype(np.float32)


def make_mel_filterbank(config):
    fft_frequencies = np.linspace(
        0.0,
        config.sample_rate / 2,
        config.n_fft_bins,
        dtype=np.float64,
    )
    minimum_mel, maximum_mel = hz_to_slaney_mel(
        np.asarray([config.fmin_hz, config.fmax_hz])
    )
    mel_points = np.linspace(
        minimum_mel,
        maximum_mel,
        config.n_mels + 2,
        dtype=np.float64,
    )
    mel_frequencies = slaney_mel_to_hz(mel_points)
    frequency_differences = np.diff(mel_frequencies)
    ramps = mel_frequencies[:, None] - fft_frequencies[None, :]

    weights = np.zeros(
        (config.n_mels, config.n_fft_bins),
        dtype=np.float64,
    )
    for index in range(config.n_mels):
        lower = -ramps[index] / frequency_differences[index]
        upper = ramps[index + 2] / frequency_differences[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))

    # librosa默认的Slaney面积归一化。
    normalization = 2.0 / (
        mel_frequencies[2:config.n_mels + 2]
        - mel_frequencies[:config.n_mels]
    )
    weights *= normalization[:, None]
    return weights.astype(np.float32)


def make_dct_matrix(config):
    mel_indices = np.arange(config.n_mels, dtype=np.float64)
    coefficient_indices = np.arange(config.n_mfcc, dtype=np.float64)
    matrix = np.cos(
        np.pi
        / config.n_mels
        * coefficient_indices[:, None]
        * (mel_indices[None, :] + 0.5)
    )
    matrix[0] *= np.sqrt(1.0 / config.n_mels)
    matrix[1:] *= np.sqrt(2.0 / config.n_mels)
    return matrix.astype(np.float32)


def frame_audio(audio, config):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) != config.clip_sample_count:
        raise ValueError(
            f"声学前端需要{config.clip_sample_count}个采样点，"
            f"实际收到{len(audio)}个。"
        )

    starts = np.arange(config.number_of_frames) * config.hop_length
    offsets = np.arange(config.n_fft)
    return audio[starts[:, None] + offsets[None, :]]


def extract_mfcc(
    audio,
    config=None,
    window=None,
    mel_filterbank=None,
    dct_matrix=None,
):
    config = config or DEFAULT_CONFIG
    window = window if window is not None else DEFAULT_WINDOW
    mel_filterbank = (
        mel_filterbank
        if mel_filterbank is not None
        else DEFAULT_MEL_FILTERBANK
    )
    dct_matrix = dct_matrix if dct_matrix is not None else DEFAULT_DCT_MATRIX

    frames = frame_audio(audio, config)
    windowed_frames = frames * window[None, :]
    spectrum = np.fft.rfft(windowed_frames, n=config.n_fft, axis=1)
    power_spectrum = np.asarray(
        np.abs(spectrum) ** 2,
        dtype=np.float32,
    )
    mel_power = power_spectrum @ mel_filterbank.T
    mel_db = 10.0 * np.log10(
        np.maximum(config.power_to_db_amin, mel_power)
    )
    mel_db -= 10.0 * np.log10(config.power_to_db_reference)

    if config.power_to_db_top_db is not None:
        mel_db = np.maximum(
            mel_db,
            np.max(mel_db) - config.power_to_db_top_db,
        )

    mfcc = mel_db @ dct_matrix.T
    return np.asarray(mfcc, dtype=np.float32)


DEFAULT_CONFIG_DATA = load_config_data()
DEFAULT_CONFIG_SHA256 = config_sha256(DEFAULT_CONFIG_DATA)
DEFAULT_CONFIG = load_config()
DEFAULT_WINDOW = make_periodic_hann(DEFAULT_CONFIG)
DEFAULT_MEL_FILTERBANK = make_mel_filterbank(DEFAULT_CONFIG)
DEFAULT_DCT_MATRIX = make_dct_matrix(DEFAULT_CONFIG)
