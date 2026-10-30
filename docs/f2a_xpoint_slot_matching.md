# F2a X Point Slot Matching Notes

This note records the current understanding of X point slot semantics, the likely
slot mismatch in F2a training, and the recommended handling strategy. It is meant
as a project memo. Do not treat it as a request to modify official environment
code.

## Context

For F2a, the official target has four fixed X point slots:

```text
T0 = upper secondary X point
T1 = upper primary X point
T2 = lower primary X point
T3 = lower secondary X point
```

In the codebase this corresponds to:

```text
X_SECONDARY_SLOTS = (0, 3)
X_PRIMARY_SLOTS = (1, 2)
```

The official target is assembled in `eval_files/trajectory_utils.py` as:

```python
rX = [Xpt_sec[0].r, Xpt_main[0].r, Xpt_main[1].r, Xpt_sec[1].r]
zX = [Xpt_sec[0].z, Xpt_main[0].z, Xpt_main[1].z, Xpt_sec[1].z]
```

So the slot order is not simply "all observed X points in whatever order HFM
returns". It is a semantic target order.

## HFM Observation Layout

The HFM observation provides:

```text
nX
rX
zX
FX
```

The safest assumption is:

```text
Only the first nX entries are valid.
The remaining entries are padding.
The array indices should not be trusted as official target slot ids.
```

For a divertor-like state with two X points, the observation may look like:

```text
nX = 2
rX/zX = [C_upper_primary, C_lower_primary, 0, 0, ...]
```

But for F2a target-slot semantics, the desired internal slotting is:

```text
slot0/T0 = invalid
slot1/T1 = C_upper_primary
slot2/T2 = C_lower_primary
slot3/T3 = invalid
```

## Behavior Of environment.xpt_utils

`environment/xpt_utils.py` is official/environment-side code and should generally
be left unchanged.

Its core function is `assign_xpoints_to_slots(obs, reference_rX, reference_zX)`.
The relevant behavior is:

```python
if force_sort_only or nx == slots:
    # Sort valid current X points by descending z and directly fill slots.
else:
    # Match candidates to reference slots.
```

With the default `slots=4`:

```text
nx == 4:
  The function ignores reference_rX/reference_zX and directly maps descending-z
  candidates to slot0..slot3.

nx != 4:
  The function calls _match_to_reference_slots().
```

`_match_to_reference_slots()` builds a distance matrix between reference slots
and current candidates, but the helper named `_linear_sum_assignment()` is a
greedy row-order assignment rather than a true global Hungarian assignment.

For `nx=2`, this can behave like:

```text
T0 picks the nearest current candidate first.
T1 picks the remaining current candidate.
T2/T3 are left invalid.
```

This is risky for F2a, because the two divertor X points usually correspond to
the primary slots T1/T2, not T0/T1.

## Suspected Training Impact

The previous F2a training logs showed:

```text
x_position_error_cm remained very large.
x_position_component was close to zero for long periods.
LCFS improved but stayed outside the target threshold.
Ip showed some degradation.
```

X point slot mismatch is a strong suspect because it can corrupt both reward and
policy input:

```text
Reward:
  x_position_cm, x_flux_diff, psix_score, and progress baselines can be computed
  against the wrong target slot.

Policy input:
  xpt_r, xpt_z, xpt_position_error_3cm, xpt_valid, and x_flux_diff can be
  provided with the wrong slot semantics.

Shape control:
  X points are tied to the separatrix and LCFS. A wrong X target can pull the
  LCFS in a wrong or inconsistent direction.
```

Therefore reward tuning alone may amplify an incorrect training signal unless
slotting is fixed at the training/submission boundary.

## Recommended Strategy

Do not modify `environment/` for this. Instead:

```text
1. Add an F2a-specific slotting function under training/ or a shared allowed
   helper used by training.
2. Mirror the same slotting logic in submission/inference_common.py.
3. Use this logic consistently for both features and reward.
4. Keep visualization scripts explicit about which slotting rule is being used.
```

The matching rule should not trust observation array indices.

Suggested topology-aware slotting:

```text
nx < 2:
  No reliable X position reward.
  Keep x_valid all false or only use topology/nX shaping.

nx == 2:
  Treat the two X points as primary divertor X points.
  Match only to T1/T2.
  Prefer upper candidate -> T1 and lower candidate -> T2, or do distance
  matching constrained to primary slots.
  T0/T3 remain invalid.

nx == 3:
  First match two primary slots T1/T2.
  Then match the remaining candidate to the most plausible secondary slot T0/T3.

nx >= 4:
  Match all four slots.
  Prefer hemisphere constraints:
    upper candidates -> T0/T1
    lower candidates -> T2/T3
  Within each hemisphere, use distance-based assignment.
```

The `x_valid` vector should reflect semantic validity:

```text
nx == 2:
  x_valid = [0, 1, 1, 0]

nx == 4:
  x_valid = [1, 1, 1, 1]
```

## Reward Gating

A practical reward schedule is:

```text
nx < 2:
  Disable x_position reward.
  Keep LCFS/Ip/coil constraints and topology formation terms.

nx == 2:
  Enable primary X position reward on T1/T2 only.
  Do not penalize missing secondary slots T0/T3 as position error.
  Keep a moderate topology/nX incentive toward full XPT formation.

nx >= 4:
  Enable full T0/T1/T2/T3 X position reward.
  Enable secondary X point terms and full XPT precision terms.
```

This avoids the two extremes:

```text
Waiting for nx==4 before any X reward gives weak guidance early.
Using all four target slots when nx==2 can pull primary X points toward
secondary targets.
```

## Visualization Checks

The check should display:

```text
current X candidates
target slots T0/T1/T2/T3
the chosen current-to-target assignment
x_valid
per-slot distance in cm
```

Expected behavior for a divertor-like reset:

```text
upper current X -> T1
lower current X -> T2
T0/T3 invalid
```

If a plot shows:

```text
upper current X -> T0
lower current X -> T1
```

then the slotting is still using reference-order greediness or raw descending-z
slot fill rather than F2a semantic matching.

## Implementation Boundary

Allowed/safe places to implement project-specific matching:

```text
training/
submission/inference_common.py
tools/visualize_xpt_reference_check.py
```

Avoid changing:

```text
environment/
evaluation/
root configs/
```

unless there is a separate, explicit reason to alter local-only diagnostic code.
