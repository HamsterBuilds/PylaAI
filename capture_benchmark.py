"""Record a bounded local replay from ADB without sending game controls."""
import argparse
from pathlib import Path
import time
from adbutils import adb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--interval', type=float, default=.2)
    args = parser.parse_args()
    if not 1 <= args.frames <= 300 or args.interval < .1:
        parser.error('Use 1-300 frames and interval >= 0.1 seconds')
    directory = Path('benchmark_captures') / time.strftime('%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    device = adb.device(serial=args.serial)
    for index in range(args.frames):
        start = time.perf_counter()
        device.screenshot().save(directory / f'{index:04d}.png')
        time.sleep(max(0, args.interval - (time.perf_counter() - start)))
    print(directory.resolve())


if __name__ == '__main__':
    main()
