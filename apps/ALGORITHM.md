# IM Feature Filter — Algorithm Report

Comprehensive description of the three-stage MS1 denoising + centroiding
pipeline — intensity smoothing, the vertical-IM feature filter, and the
watershed centroider. All three stages are now part of the public `tdfpy`
package (`tdfpy.smooth` / `box_smooth`, `tdfpy.VerticalNoiseFilter`,
`tdfpy.WatershedCentroider`); the timsTOF viewer app under
[`timstof_viewer/`](timstof_viewer/) exposes them as tunable knobs on Bruker
timsTOF MS1 frames. This document is the algorithm reference behind those
package APIs.

---

## Table of contents

1. [Data model](#data-model)
2. [Pipeline overview](#pipeline-overview)
3. [Stage 1 — Vertical-IM feature filter](#stage-1--vertical-im-feature-filter)
4. [Stage 2 — Pre-centroid intensity smoothing](#stage-2--pre-centroid-intensity-smoothing)
5. [Stage 3 — Watershed centroider](#stage-3--watershed-centroider)
6. [Post-centroid noise filter](#post-centroid-noise-filter)
7. [Parameter reference](#parameter-reference)
8. [Computational complexity](#computational-complexity)
9. [Failure modes & tuning guidance](#failure-modes--tuning-guidance)
10. [Glossary](#glossary)

---

## Data model

The pipeline operates in **integer (scan_number, TOF_index)** space — before
any m/z or 1/K0 conversion. This is the native form of Bruker raw data and
avoids floating-point binning artifacts.

A single MS1 frame consists of:

- **`num_scans`** ion-mobility scans (typically ~700), each a list of `(tof_idx, intensity)` pairs.
- Flattened to four parallel arrays of length `N` (total raw points):
  - `scan_indices : int64`   — which IM scan (0..num_scans−1) the point lives in
  - `mz_indices   : int64`   — TOF index (integer)
  - `intensities  : float64` — per-pixel intensity
  - For display: `mz_values : float64`, `ook0_values : float64` (computed via Bruker calibration).

The filter and centroider work on the integer arrays; conversion to physical
units happens only when emitting final centroids or rendering plots.

---

## Pipeline overview

```
Bruker .d frame
   │
   ▼
[Load raw points]                  (scan, mz_idx, intensity, mz, ook0)
   │
   ▼
[Stage 1: Vertical-noise feature filter]   ← iterates num_iterations times
   │
   ├─→ keep_point_mask  (cumulative boolean mask)
   │
   ▼ (apply mask to all per-point arrays)
filtered_points
   │
   ▼
[Stage 2: Box-average smoothing]   ← optional, replaces intensities only
   │
   ▼
[Stage 3: Watershed centroider]
   │
   ▼
centroids  [mz, intensity, ook0]
   │
   ▼
[Post-centroid noise filter]       ← optional intensity-floor on centroid totals
   │
   ▼
final centroids displayed
```

Each stage is independently togglable in the dashboard. The data flow is
strictly left-to-right; later stages never feed back into earlier ones.

---

## Stage 1 — Vertical-noise feature filter

**Function**: `filter_vertical_im_features` (algorithm), wrapped by
`_run_filter` (which adds iteration).

### Intuition

A real ion produces a **vertical streak** in (mz_index × scan_number) space:
intensity at roughly the same TOF index across many consecutive IM scans.
TOF satellites and noise tend to be either short streaks, isolated single
hits, or random scatter.

The filter walks each TOF index, looks at the IM profile in a small mz window
around it, and keeps only points belonging to long-enough vertical features.

### Algorithm (single pass)

For each **unique TOF index** `c` present in the input (points sharing a TOF
index see identical windows, so we evaluate once per unique index):

1. **Build a column window** around `c`:
   ```
   window = { points with mz_index ∈ [c − mz_idx_half_width, c + mz_idx_half_width] }
   ```
   Located via `np.searchsorted` on the mz-sorted point array — O(log N).

2. **Sum intensities per scan** inside the window:
   ```
   profile[s] = Σ intensity of window points at scan = s
   ```
   Computed via `np.bincount` with `minlength=num_scans`. The array has length
   `num_scans` and is mostly zeros.

3. **Mark occupied scans**: a scan is occupied iff `profile[s] > 0`. The
   bincount zero-fills empty scans, so this naturally excludes them.

4. **Find gap-closed runs**: walk the sorted occupied-scan list. Two
   consecutive occupied scans split into separate runs only when their gap
   exceeds `max_gap_scans` empty scans. Equivalently, break where
   `diff > max_gap_scans + 1` (the diff between consecutive occupied scan
   indices). Closing gaps gives the IM filter a morphological-close behavior:
   it tolerates short empty patches inside a feature.

5. **Filter by run length**: a run survives if its total span
   (`last_occupied − first_occupied + 1`, inclusive of internal gaps) is
   `≥ min_streak_scans`.

6. **Filter by run intensity**: compute total intensity over the span
   (including sub-threshold cells inside internal gaps — they represent real
   signal that just didn't clear the per-scan bar). Drop runs whose total
   intensity is below `min_streak_intensity`.

7. **Map kept runs back to points**: for every input point at TOF index `c`,
   keep it iff its scan number falls inside any kept run.

Diagnostics returned:

- `feature_span_intensities`: total intensity per run that **cleared
  `min_streak_scans`** (before the intensity-floor filter). This feeds the
  histogram in the dashboard so you can pick `min_streak_intensity`
  visually.

### Iteration

The whole single-pass filter is wrapped in a loop:

```
cumulative_mask = ones(N)
for pass in 1..num_iterations:
    survivors = points[cumulative_mask]
    new_mask  = filter_vertical_im_features(survivors, ...)
    # Compose into the original point order:
    active_idx       = where(cumulative_mask)
    kept_globally    = active_idx[new_mask]
    cumulative_mask  = zeros(N); cumulative_mask[kept_globally] = True
    if cumulative_mask.sum() == 0: break
```

Each pass operates on the survivors of the previous one. Points that *only
just* made the cut on pass 1 — because they sat next to a barely-thick noise
streak — get dropped on pass 2 when those streaks are gone. The mask is
always composed against the original point order, so downstream stages see
the same indexing as the raw input.

The per-pass attrition (`raw → p1 → p2 → ...`) is surfaced in the dashboard
as a one-line caption beneath the metric strip.

### Parameters

| Name | Type | Role |
|---|---|---|
| `mz_idx_half_width` | int (TOF idx) | Column half-width: window spans `[c − half_width, c + half_width]` |
| `max_gap_scans` | int (scans) | Max consecutive empty scans tolerated inside a streak (morphological-close radius) |
| `min_streak_scans` | int (scans) | Minimum total span of a kept run (gap-inclusive) |
| `min_streak_intensity` | float | Total summed intensity (column window × span) required for a run to be kept |
| `num_iterations` | int | How many times to re-apply the filter to its own survivors |

---

## Stage 2 — Pre-centroid intensity smoothing

**Function**: `smooth_intensities_box_average`.

### Intuition

Per-pixel raw intensities are noisy. A single noisy bright pixel can:

- Hijack seed priority in the centroider (the spike claims seed status before
  the actual peak it sits inside),
- Inflate the centroid total or destabilize the intensity-weighted (mz, IM)
  position.

A simple box-average over a small neighborhood replaces each point's
intensity with a more representative local estimate.

### Algorithm

For each surviving point at `(scan, mz_idx)`:

```
new_intensity = mean(
    point.intensity for point in original_input
    where |Δscan|   ≤ smooth_scan_half_width
      AND |Δmz_idx| ≤ smooth_mz_idx_half_width
)
```

The point is always in its own box, so the divisor is at least 1. Positions
are untouched — only intensities are rewritten.

**Implementation**: a sparse bucket grid keyed by
`(scan // smooth_scan_half_width, mz_idx // smooth_mz_idx_half_width)`. Each
query inspects the 3×3 cell neighborhood (guaranteed to contain every point
within the tolerance, because cell side equals tolerance).

### What this fixes downstream

In Stage 3 the centroider sorts by descending intensity to pick seeds. With
raw intensities, a noisy spike can outrank the actual peak summit, becoming
a seed off-center. With smoothed intensities, the seed ordering reflects
local average density — much more stable.

### Parameters

| Name | Type | Role |
|---|---|---|
| `smooth_enabled` | bool | Master toggle. Default on. |
| `smooth_scan_half_width` | int (scans) | ±scan window for averaging |
| `smooth_mz_idx_half_width` | int (TOF idx) | ±mz_idx window for averaging |

---

## Stage 3 — Watershed centroider

**Function**: `watershed_centroid`. Replaces an earlier greedy span-and-grow
centroider that had trouble with peak-shape variability.

### Intuition

Think of intensity as elevation. **Watershed segmentation** finds catchment
basins by gradually lowering a water level — peaks emerge as islands first,
then progressively flood; where two basins meet, that's a boundary.

We don't actually flood. Equivalently, we process points in **descending
intensity order** (highest island first, then lower elevations as the water
recedes). For each point we ask: is the nearest already-claimed pixel close
enough to belong to its catchment? If yes, attach. If no, start a new one.

### Algorithm

```
sort points by intensity descending
grid = {}    # bucket grid keyed by (scan_cell, mz_cell)
seeds = []   # one entry per group: just the seed's intensity (for tiebreaks)
group_id[i] = -1 for every i   # -1 means orphan / unassigned

for p in points (descending intensity order):
    # Find best in-box neighbor across the 3x3 cell neighborhood.
    best = None
    for ds, dm in product(-1..1, -1..1):
        for q in grid[p.scan_cell + ds, p.mz_cell + dm]:
            if |p.scan - q.scan| > attach_scan_half_width: continue
            if |p.mz - q.mz| > attach_mz_idx_half_width: continue
            d = manhattan(p, q)
            # Tiebreak: smaller d wins; on tie, higher seed-intensity wins.
            if better than current best: best = q

    if best is not None:
        group_id[p] = group_id[best]      # join (region growing)
        grid[p.cell].append(p)            # contribute to future queries
    elif p.intensity >= min_seed_intensity:
        new_group = len(seeds)
        group_id[p] = new_group
        seeds.append(p.intensity)
        grid[p.cell].append(p)
    else:
        # Orphan: too weak to seed, no group within reach.
        # NOT added to the grid — orphans don't claim territory.
        drop p

# Aggregate centroids: one per group.
for group g:
    total       = Σ intensity of group members
    centroid_mz   = Σ (mz × intensity)   / total
    centroid_ook0 = Σ (ook0 × intensity) / total
    if total >= min_centroid_intensity:
        emit (centroid_mz, total, centroid_ook0)
```

### Three things make this work

1. **Bucket grid sized to the tolerance.** Cells are exactly
   `(attach_scan_half_width, attach_mz_idx_half_width)`. Any point within the tolerance must live in one
   of the 3×3 cells around the query, so a small fixed neighborhood is
   sufficient. No KD-tree, no rebuilding — just a dict-of-lists, O(1)
   amortized insertion.

2. **Nearest-neighbor lookup is over *all assigned points*, not just seeds.**
   This is the crucial difference from a "distance-to-seed" approach. As a
   group grows, its **followers** are also entries in the grid. A later
   point can attach to a follower, extending the group's reach beyond what
   the anchor alone covered. The group naturally takes the shape of the
   peak.

3. **Watershed boundary emerges from intensity ordering.** When two peaks
   are far apart, both anchors get promoted. Their groups grow outward and
   meet in the middle. Each valley point joins whichever group's nearest
   member is closest in Manhattan distance — and because we process top-down,
   the boundary lands where the catchments naturally divide.

### Tiebreaking

When two already-assigned points are Manhattan-equidistant from the
candidate, we prefer the one whose **group's seed has higher intensity**.
This keeps the watershed boundary stable (it favors the more prominent peak)
and deterministic.

### Why this fixes peak splitting

Earlier greedy approaches walked outward from a seed along the IM axis,
stopping at apparent "valleys" inside the peak. They would over-split
asymmetric or doubly-peaked shapes. The watershed approach has no per-peak
walk — every point either joins a group or starts one based on a single
local distance check. A broad peak with an internal dip is one group as
long as the dip's points are within the box of some already-assigned
neighbor (which they almost always are).

### Orphans

A point with **no in-box neighbor** AND `intensity < min_seed_intensity` is
dropped. It does not enter the grid (so it can't claim territory for later
weaker points). This filters out genuinely isolated noise pixels that the
upstream IM filter happened to keep.

### Centroid filtering

After aggregation, groups whose summed intensity is below
`min_centroid_intensity` are dropped. This is a final coarse filter for very
weak groups that survived the seed-intensity threshold but didn't accrete
much real signal.

### Diagnostics returned

| Field | Meaning |
|---|---|
| `num_seeds_emitted` | Centroids in final output (post `min_centroid_intensity`) |
| `num_seeds_promoted` | Total candidates that cleared `min_seed_intensity` |
| `num_followers` | Points attached to an existing group |
| `num_orphans_dropped` | Points with no in-box neighbor AND below `min_seed_intensity` |

These appear in the dashboard's centroid metric strip.

### Parameters

| Name | Type | Role |
|---|---|---|
| `attach_scan_half_width` | int (scans) | NN reach + seed exclusion on the scan axis |
| `attach_mz_idx_half_width` | int (TOF idx) | Same on the TOF-index axis |
| `min_seed_intensity` | float | Floor for promoting orphans to new seeds |
| `min_centroid_intensity` | float | Drop final centroids whose summed group intensity falls below this |

---

## Post-centroid noise filter

After centroiding, optionally apply an intensity floor to centroid totals:

```
threshold = estimate_noise_level(centroid_intensities, method=...)
keep      = centroids[ centroid.intensity >= threshold ]
```

The threshold can be:

- **`absolute`**: a user-provided number.
- **`mad`** / **`percentile`** / **`histogram`** / **`baseline`** /
  **`iterative_median`**: data-driven estimators from `tdfpy.noise`.

This is identical to the noise-filter logic used in the main `tdfpy` package
and gives consistent post-processing across the two centroiders.

---

## Parameter reference

All knobs, by stage, in the order they appear in the sidebar:

### Stage 1 — Vertical-noise filter
- `mz_idx_half_width` (default 3)
- `min_streak_scans` (default 5)
- `max_gap_scans` (default 1)
- `min_streak_intensity` (default 50)
- `num_iterations` (default 2)

### Stage 2 — Smoothing
- `smooth_enabled` (default on)
- `smooth_scan_half_width` (default 5)
- `smooth_mz_idx_half_width` (default 3)

### Stage 3 — Watershed centroider
- `attach_scan_half_width` (default 10)
- `attach_mz_idx_half_width` (default 3)
- `min_seed_intensity` (default 0)
- `min_centroid_intensity` (default 0)
- `max_mz_idx_from_seed` (default 10)

### Post-centroid
- `centroid_noise_mode` (default off; one of off / absolute / mad / percentile / histogram / baseline / iterative_median)

The asymmetric `(10, 3)` defaults on the watershed and smoothing boxes
reflect MS data biology: IM peaks span several scans (each scan is a
fine-grained ramp position), while TOF peaks are sharp (~3 indices wide).

---

## Computational complexity

Let:
- `N` = number of raw points in the frame (~10⁵ for typical MS1)
- `S` = `num_scans` (~700)
- `U` = number of unique TOF indices seen (~10⁴)
- `k` = average points per 3×3 cell neighborhood

| Stage | Cost | Notes |
|---|---|---|
| Load + indexing | O(N log N) | argsort by mz_idx |
| Stage 1 (single pass) | O(U · (K + S)) where K = pts per window | Per-column bincount + run-finding |
| Stage 1 (iterated) | × num_iterations | Survivor sets shrink, so passes get cheaper |
| Stage 2 (smoothing) | O(N · k) | Bucket-grid 3×3 neighborhood |
| Stage 3 (centroider) | O(N · k) | Same grid pattern |
| Post-centroid noise filter | O(C) where C = #centroids | Trivial |

In pure NumPy/Python (no Numba) typical end-to-end time per frame on a
modern laptop is a couple of seconds, dominated by Stage 2 if smoothing is
on with dense data. All stages are cached by Streamlit, so re-tuning any
single parameter only re-runs the affected stage and its downstream
dependents.

---

## Failure modes & tuning guidance

### Stage 1 — filter

**Symptom: too aggressive (real peaks dropped)**
- Lower `min_streak_scans` (default 5).
- Raise `max_gap_scans` if features have gaps in their IM profile.
- Lower `min_streak_intensity` if low-intensity peaks are getting masked.
- Reduce `num_iterations` — each pass is strictly more aggressive than the
  last.

**Symptom: not aggressive enough (noise survives)**
- Raise `min_streak_scans`.
- Raise `min_streak_intensity` (use the feature-span-intensity histogram for
  calibration).
- Increase `num_iterations`.

**Symptom: peak-edge clipping**
- Tighten `max_gap_scans` (default 1). Too-loose gap-closing can extend
  features past their real boundaries.
- Widen `mz_idx_half_width` if real peaks span more TOF indices than the
  column window catches.

### Stage 2 — smoothing

**Symptom: small adjacent peaks merge**
- Reduce `smooth_scan_half_width` (smaller IM window).
- Reduce `smooth_mz_idx_half_width`.
- Or turn smoothing off and rely on the centroider's distance criterion alone.

**Symptom: noisy centroid totals**
- Increase the smoothing box. A larger box averages over more pixels →
  smoother per-point intensity.

### Stage 3 — centroider

**Symptom: real peaks split into multiple centroids**
- Increase `attach_scan_half_width` (most common cause — IM peak is wider
  than the box).
- Increase `attach_mz_idx_half_width` if the peak's TOF profile is wider
  than 3 indices.
- Verify Stage 2 smoothing is enabled — raw spikes can offset the seed and
  cause unintended splits when boundaries land "diagonally."

**Symptom: adjacent peaks merge**
- Decrease `attach_scan_half_width` or `attach_mz_idx_half_width`.
- If only some merges are problematic, raise `min_seed_intensity` — weaker
  peaks may not deserve their own group anyway.

**Symptom: too many tiny centroids**
- Raise `min_seed_intensity` (block weak orphans from seeding).
- Raise `min_centroid_intensity` (drop weak groups post-aggregation).

**Symptom: too few centroids / good peaks dropped**
- Lower `min_seed_intensity` and `min_centroid_intensity`.
- Confirm Stage 1 isn't dropping the input points — the per-pass attrition
  caption shows this directly.

### Post-centroid noise filter

Use `mad` or `iterative_median` for adaptive thresholds. `absolute` is for
when you know your noise floor from external context.

---

## Glossary

| Term | Meaning |
|---|---|
| **Scan** / **scan number** | Bruker's index for an ion-mobility ramp position. Higher scan numbers = different mobility. Within a frame, scans are integers 0..`num_scans−1`. |
| **TOF index** / **mz_idx** | Integer time-of-flight bucket. Converts to m/z via the instrument's calibration polynomial. Working in TOF index space avoids floating-point binning. |
| **Frame** | A complete TIMS ramp = `num_scans` IM scans. Each MS1 frame has a single retention time. |
| **Point / pixel** | One `(scan, mz_idx, intensity)` triple — a single Bruker raw datum. |
| **Column** | A vertical strip in `(scan, mz_idx)` space at a given mz center, ±`mz_idx_half_width` indices wide. |
| **Profile** | A 1D array of length `num_scans` giving summed intensity inside a column at each IM scan. |
| **Run** | A maximal contiguous block of occupied scans in a profile, allowing gaps ≤ `max_gap_scans`. |
| **Feature** | In Stage 1, a run that survives the length + intensity filters. Each surviving feature contributes points to the output mask. |
| **Box** | A rectangular tolerance region `[−half_width_scan, +half_width_scan] × [−half_width_mz_idx, +half_width_mz_idx]` around a point. Used by smoothing (`smooth_*_half_width`) and the centroider (`attach_*_half_width`). |
| **Seed** | In Stage 3, a point promoted to be the anchor of a new centroid group because nothing in its box is already assigned. |
| **Follower** | A point that joined an existing group by virtue of having an in-box assigned neighbor. |
| **Orphan** | A point with no in-box assigned neighbor AND `intensity < min_seed_intensity`. Discarded. |
| **Group** | One seed plus its followers. Aggregates into one final centroid. |
| **Centroid** | A single `(mz, intensity, ook0)` output value computed as the intensity-weighted average over a group. |
