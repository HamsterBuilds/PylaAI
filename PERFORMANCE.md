# Performance verification

## Offline pathfinding check

This benchmark does not launch Brawl Stars or connect to ADB:

```powershell
py -3.11 benchmark_pathfinding.py
```

It compares the legacy direct/X/Y/cardinal policy with the new planner and
reports route success, relative success improvement, route length, direction
changes and A* cache reuse for open space, a single wall, an L-wall, a corridor
and offset barriers.

## Subsequent changes (not executed or tested)

State recognition now uses state-specific fast paths. Match frames check end
results, lobby frames check lobby first, and matchmaking/brawler-selection
frames check their own marker before falling back to the complete classifier.
A configurable full scan every two seconds catches unexpected transitions.
No template threshold or transition consensus was removed. Untested.

Complete attack taps are now rate-limited to a configurable 80 ms minimum
interval, preventing redundant high-FPS commands from building up on scrcpy's
control channel. Charged-attack down/up edge events bypass the limiter, as do
gadget, super and hypercharge controls. Attack timing resets between matches.
The method reports whether a tap was sent for future playstyles. Untested.

Entity inference is skipped in confirmed non-game menus. During matchmaking or
an unknown transition, a configurable 4 Hz probe checks for a freshly detected
player. Two consecutive detections promote the state to match. Probe output is
reused by the first decision on that exact frame, avoiding duplicate inference,
and temporary data clears on perception reset. Untested.

Fixed a temporal-perception regression found by manual gameplay: an empty
`player` list was treated as valid because validation checked only for the key.
All detection classes now discard malformed boxes and gameplay requires at
least one four-coordinate player box. The playstyle boundary has an additional
empty-player guard that releases movement instead of indexing the list. Two
regression tests were added but not executed per the user's instruction.

Temporal tracks now estimate per-edge velocity with a bounded exponential
filter and project detections 0.25 frame ahead to compensate for capture and
inference latency. A single retained player/teammate frame advances along the
estimated trajectory instead of freezing. Prediction is capped at 0.75 frames;
prediction and velocity smoothing are configurable. Untested.

Attack and super targeting now keep separate previous-target positions. A
previous hittable target remains selected while it can still be matched and no
alternative is over 18% closer. Blocked or disappeared targets clear memory,
and all target memory resets outside matches. Matching distance and switch
ratio are configurable. This block is untested.

The earlier farther-enemy shortcut now preserves candidates close enough to
match the remembered target. Unrelated farther enemies still skip wall checks;
target hysteresis therefore works without restoring full per-enemy work.

State classification now passes through a bounded consensus filter: transitions
require two consecutive matching observations, while an established state is
returned immediately. Confirmed states carry the source frame timestamp and
expire after one second without fresh confirmation, preventing controls from
using stale screen context. Direct fallback recognition supplies its own
timestamp. Confirmation count and maximum age are configurable. Untested.

Each playstyle decision now owns a frame-local memoization cache. Repeated
queries for the nearest enemy, nearest teammate and poison-gas directions reuse
their result within that decision, avoiding duplicate distance, line-of-sight,
wall and HSV scans. The cache is cleared before every new frame, and poison
results are copied at the script boundary to prevent script mutations from
corrupting cached state. This block is untested.

Ability readiness now stops scanning an ability's pixels after it becomes ready;
the corresponding use method clears that state and resumes checks. Scaled crop
coordinates are cached and shared readiness code removes three duplicated image
paths while preserving each ability's HSV limits, threshold and debug output.
This block is untested.

Frame consumers now wait on a shared condition and are notified by the video
thread on each new frame. This removes 10 ms polling from the main and state
loops while retaining bounded timeouts for stop and stale-feed handling.

Automatic capture rate uses 15 FPS on up to four logical CPUs, 20 FPS on up to
eight, and 30 FPS above that. `capture_max_fps` can override this from 1-120;
an explicit bot FPS remains an upper bound. Resolution is unchanged. Untested.

Wall models are now loaded lazily after a player first appears, reducing menu
startup work and resident model memory before gameplay. A bounded 96x54
grayscale scene sample gates wall inference: meaningful visual change refreshes
immediately, while unchanged scenes reuse cached geometry for at most 0.75 s.
The comparison threshold and maximum age are configurable. A scene sample is
accepted only after successful inference, so failures remain eligible to retry.
This block has not been executed or manually verified.

A new bounded temporal perception layer smooths matching player, teammate and
enemy boxes. It bridges one missed player/teammate frame, while enemies expire
immediately so attacks never target a retained ghost. Per-class storage is
limited to 16 tracks and resets outside matches together with cached walls.

State recognition is scheduled at 4 Hz during matches and 10 Hz in menus,
giving entity inference priority on slower hardware. Crash checks reuse their
already-fetched foreground package instead of issuing a duplicate ADB request.
Joystick updates within two pixels of the last command are suppressed when
movement reapplication is disabled. These changes require manual gameplay
verification and have not been executed at the user's request.

Closest-enemy selection skips wall visibility scans for candidates at least as
far away as an already hittable target. The original preference for hittable
targets, nearest blocked fallback, and first-in-order distance ties remains in
the selection logic. Swept-circle checks calculate the squared player radius
once per query rather than once per intersected wall. These edits have not been
run or benchmarked.

At the user's request, tests and live automation are no longer run. Earlier
measurements below apply only to the revisions measured at that time.

The video receiver now waits for socket readability instead of repeatedly
polling an empty socket every 10 ms. The wait is bounded to 100 ms for shutdown
and wakes on incoming data. Match/state handlers now refresh the captured
frame after potentially blocking menu work, before running the play logic.

Optional debug recording now limits raw pre-roll image storage to 32 MiB and
uses a deque to evict oldest frames without rebuilding a list. At 1920x1080
RGB, three seconds at 30 FPS previously required roughly 534 MiB in raw copies.
This trades pre-roll duration for bounded RAM; actual recording resolution is
unchanged. Disabled recording does not benefit from this limit. Failure to open
a video writer no longer attempts to flush buffered images into a missing writer.

## Real lobby-frame replay

`benchmark_pipeline.py` pins the original baseline to
`b648138398310e3143d8631692539e149573b17e`. Run baseline and current in separate
processes with identical input files, e.g.:

```
py -3.11 benchmark_pipeline.py benchmark_frame.png --version baseline --cycles 30
py -3.11 benchmark_pipeline.py benchmark_frame.png --version current --cycles 30
```

One real BlueStacks lobby image, replayed for 30 cycles, measured 5.953125 vs
1.890625 process CPU seconds (68.2% reduction). Median perception work was
215.78 vs 65.21 ms. Both classified all 30 frames as lobby and found no player.
This is a preliminary single-pair replay result with identical repeated pixels;
it does NOT establish a 30% improvement during matches, complete bot operation,
RAM consumption, or better decisions. The script covers entity inference,
scheduled wall inference and state matching; controls and capture are excluded.
Live ADB connection and game launch are now verified. Match verification remains
outstanding. Local screenshots are git-ignored to avoid publishing account data.

## Additional workload reductions

State matching now reuses scores only for byte-identical image regions. The
per-thread cache is limited to 8 MiB of pixel data and 64 entries; template
storage uses a 128-entry LRU. Every changed region is matched again with the
original RGB algorithm and thresholds. No state transitions are deliberately
delayed. Tests: `py -3.11 -m unittest test_performance -v` (3 passed).

Synthetic 1920x1080 state-detection measurements against the original revision:
unchanged frames 124.62 -> 1.49 ms (98.8% less elapsed time); completely changing
random frames 123.63 -> 126.62 ms (2.4% overhead). Real gameplay savings depend
on how many checked regions stay identical. These are not whole-bot CPU or RAM
measurements; the cache uses additional bounded memory to save computation.

Capture is capped at 30 FPS independently of inference by default. Optional
`capture_max_fps` in general_config.toml accepts 1-120; explicit bot FPS limits
can lower capture further. Full image resolution is preserved. Connection
discovery now uses at most 8 workers instead of one per candidate port.

Run from this directory with `py -3.11 benchmark_detection.py`. The script
compares the working detector against `git show HEAD:detect.py`; after committing,
HEAD must be replaced with the original revision to retain that comparison.

Measured on Intel Core i5-8259U / Iris Plus 655, using DirectML, a seeded
synthetic 1920x1080 RGB image, warmup, and alternating measurement order:

| Operation | Original | Updated | Time reduction |
| --- | ---: | ---: | ---: |
| Image preparation | 2.355 ms | 1.166 ms | 50.5% |
| Complete entity detection | 40.510 ms | 39.173 ms | 3.3% |

These are microbenchmarks, not gameplay or trophy farming measurements.
They do not establish a 40% overall improvement. CPU inference alone measured
101.3 / 58.3 / 43.6 ms with 1 / 2 / 4 threads on a zero input; DirectML remains
enabled. Emulator contention and actual gameplay can change these results.

Changes reuse resize storage and normalize directly into the model input,
disable ONNX thread-pool busy waiting, configure DirectML sequential execution
and memory-pattern requirements, and skip repeated inference on the same
captured frame. Frame timestamps travel with the exact image to avoid a race.
Aspect-ratio changes now clear padding; the benchmark verifies pixel equivalence
against a fresh reference buffer for landscape, portrait, and square frames.

BlueStacks Pie64 has ADB enabled on port 5555, 1920x1080 resolution and a 30 FPS
limit. Pyla's configured emulator port now matches it. No connected ADB device
was available during verification, so live gameplay remains unverified.
