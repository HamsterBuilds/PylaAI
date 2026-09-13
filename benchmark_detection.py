"""Compare current detector with committed baseline: py -3.11 benchmark_detection.py."""
import subprocess
import time
import types
import numpy as np
from detect import Detect


def main():
    baseline = types.ModuleType('baseline_detect')
    exec(subprocess.check_output(['git', 'show', 'HEAD:detect.py'], text=True), baseline.__dict__)
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    old = baseline.Detect('models/mainInGameModel.onnx')
    new = Detect('models/mainInGameModel.onnx')
    np.testing.assert_allclose(old.preprocess_image(frame)[0], new.preprocess_image(frame)[0], atol=1e-7)
    for shape in [(1920, 1080, 3), (640, 640, 3), (1080, 1920, 3)]:
        sample = rng.integers(0, 256, shape, dtype=np.uint8)
        new.preprocess_image(sample)
        fresh = object.__new__(baseline.Detect)
        fresh.input_size = (640, 640)
        fresh._padded_img_buffer = np.full((1, 3, 640, 640), 128 / 255, np.float32)
        # Newer baselines cache resize geometry.  A deliberately fresh
        # detector must initialise those fields just as __init__ does; without
        # them this equivalence check tests an invalid object and crashes.
        fresh._preprocess_source_shape = None
        fresh._preprocess_target = None
        fresh._normalization_scale = np.float32(1.0 / 255.0)
        np.testing.assert_allclose(fresh.preprocess_image(sample)[0], new.preprocess_image(sample)[0], atol=1e-7)
    print('Pixel equivalence and aspect-ratio padding: PASS', flush=True)
    for label, count, operation in [('preprocess', 200, 'preprocess_image'), ('full detection', 15, 'detect_objects')]:
        timings = [[], []]
        for detector in (old, new):
            for _ in range(3):
                getattr(detector, operation)(frame)
        for round_no in range(4):
            for index in ([0, 1] if round_no % 2 == 0 else [1, 0]):
                detector = (old, new)[index]
                start = time.perf_counter()
                for _ in range(count):
                    getattr(detector, operation)(frame)
                timings[index].append((time.perf_counter() - start) / count)
        before, after = [float(np.median(values)) for values in timings]
        print(f'{label}: before={before*1000:.3f} ms after={after*1000:.3f} ms; time reduction={(1-after/before)*100:.1f}%; throughput={(before/after-1)*100:.1f}%', flush=True)


if __name__ == '__main__':
    main()
