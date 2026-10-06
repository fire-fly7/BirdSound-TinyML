"""Cross-language check for INT8 ties, saturation and model input scales."""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.deployment_contract import CONTRACT, CONTRACT_SHA256, quantize_int8, render_c_header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='cc')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='shared-quantization-') as directory:
        root = Path(directory)
        (root / 'shared_frontend_contract.h').write_text(render_c_header(CONTRACT))
        (root / 'quantizer.c').write_text('#include "shared_frontend_contract.h"\nint q(float x, float s, int z) { return SharedQuantizeInt8(x,s,z); }\n')
        library = root / 'quantizer.so'
        subprocess.run([args.compiler, '-std=c11', '-O2', '-shared', '-fPIC',
                        str(root / 'quantizer.c'), '-lm', '-o', str(library)], check=True)
        q = ctypes.CDLL(str(library)).q
        q.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int]
        q.restype = ctypes.c_int
        count = 0
        random = np.random.default_rng(42)
        for scale in [1.0, 4.34705114364624, 0.3137255012989044, 0.012634358368813992]:
            for zero_point in [-128, -1, 0, 73, 127]:
                halves = np.arange(-256.5, 256.5, 1, dtype=np.float32) * np.float32(scale)
                values = np.concatenate([halves, np.nextafter(halves, np.float32(-np.inf)),
                                         np.nextafter(halves, np.float32(np.inf)),
                                         random.uniform(-1000, 1000, 1000).astype(np.float32),
                                         np.array([-1e30, 1e30], dtype=np.float32)])
                expected = quantize_int8(values, scale, zero_point)
                actual = np.array([q(float(x), scale, zero_point) for x in values], dtype=np.int8)
                if not np.array_equal(actual, expected):
                    index = int(np.flatnonzero(actual != expected)[0])
                    raise AssertionError((scale, zero_point, float(values[index]), int(actual[index]), int(expected[index])))
                count += len(values)
        print(json.dumps({'passed': True, 'cross_language_cases': count,
                          'frontend_contract_sha256': CONTRACT_SHA256}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
