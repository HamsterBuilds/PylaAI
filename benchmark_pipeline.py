"""Replay perception work in isolated processes; never sends game controls.

py -3.11 benchmark_pipeline.py benchmark_frame.png --version current
Use --version baseline for the original source at BASELINE.
This covers both networks and state recognition, not complete gameplay.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import types
import cv2
import numpy as np

BASELINE = 'b648138398310e3143d8631692539e149573b17e'


def source_module(name):
    module = types.ModuleType('baseline_' + name)
    exec(subprocess.check_output(['git', 'show', f'{BASELINE}:{name}.py'], text=True), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--version', choices=['baseline', 'current'], required=True)
    parser.add_argument('--cycles', type=int, default=60)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('cycles must be positive')
    if args.version == 'baseline':
        detector, states = source_module('detect'), source_module('state_finder')
    else:
        import detect as detector
        import state_finder as states
    frames = []
    paths = []
    for item in args.images:
        path = Path(item)
        paths.extend(sorted(path.glob('*.png')) if path.is_dir() else [path])
    if not paths:
        parser.error('No input images')
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            parser.error(f'Cannot read {path}')
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    entity = detector.Detect('models/mainInGameModel.onnx', classes=['enemy', 'teammate', 'player'])
    walls = detector.Detect('models/tileDetector.onnx', classes=['wall', 'bush', 'close_bush'])
    # Warm both networks, including lazy driver initialization, before timing.
    for _ in range(3):
        entity.detect_objects(frames[0], .55)
        walls.detect_objects(frames[0], .6)
        states.get_state(frames[0])
    durations = []
    player_frames = 0
    state_counts = {}
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    for i in range(args.cycles):
        frame = frames[i % len(frames)]
        tick = time.perf_counter()
        data = entity.detect_objects(frame, .55)
        player_frames += bool(data.get('player'))
        # Fixed work schedule: 10 entity, 5 wall, 10 state checks / second.
        if i % 2 == 0 and (args.version == 'baseline' or data.get('player')):
            walls.detect_objects(frame, .6)
        state = states.get_state(frame)
        state_counts[state] = state_counts.get(state, 0) + 1
        durations.append(time.perf_counter() - tick)
        time.sleep(max(0, .1 - durations[-1]))
    elapsed = time.perf_counter() - start_wall
    print(json.dumps({
        'version': args.version, 'cycles': args.cycles,
        'cpu_seconds': time.process_time() - start_cpu,
        'elapsed_seconds': elapsed,
        'work_median_ms': float(np.median(durations) * 1000),
        'work_p95_ms': float(np.percentile(durations, 95) * 1000),
        'player_frames': player_frames, 'states': state_counts,
        'source_images': [p.name for p in paths],
    }), flush=True)


if __name__ == '__main__':
    main()
