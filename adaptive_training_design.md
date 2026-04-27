# Adaptive Training Design — ResNet50 Student Distillation

## 1. Rethinking Loss Weights: Signal-Based vs Milestone-Based

### The problem with fixed-phase blocks

The current design has three step-function phases switching at epoch 16 and epoch 51. All six
loss weights change simultaneously at each boundary. This is a "milestone" schedule — it assumes
that after N epochs, every loss has matured enough to advance to the next stage. The logs show
this assumption is wrong:

- `cls_kd` was still falling at epoch 15 (17,000 → 295, a 98% drop) — the phase boundary
  interrupts an active training dynamic
- `feat_kd` never converged in any phase, so giving it a new weight achieves nothing
- `id_loss` activated cold at epoch 16 with no warmup, causing a 2× val loss spike
- `emb_kd` was activated simultaneously with `id_loss`, even though `emb_kd` depends on
  the teacher projector being trained first by `id_loss`

### The right model: curriculum distillation with dependency chains

Each loss has a natural dependency on what must be learned before it can contribute a
meaningful signal. The ordering is:

**Tier 1 — No dependencies (start from epoch 1):**
- `det` — backbone is pretrained; detection can learn immediately
- `cls_kd` — teacher logits are available from epoch 1

**Tier 2 — Depends on Tier 1 stabilising (~epoch 6–8):**
- `box_kd` — needs `det` to produce stable positive-cell assignments first
- `feat_kd` — needs FPN features to stop changing rapidly before adapter can align

**Tier 3 — Depends on Tier 2 (~epoch 15–20):**
- `id_loss` — needs some embedding structure (from `feat_kd`) before cross-entropy is useful
- `emb_kd` — needs teacher projector to be trained by `id_loss` before its cosine targets mean anything

The current schedule activates all of Tier 2 and 3 simultaneously at epoch 16 — a violation
of the dependency chain.

### Adaptive weight principle: keep weighted gradient contribution roughly constant

The raw scale of each loss changes dramatically during training:
- `cls_kd`: 17,000 at epoch 1 → 25 at epoch 80 (600× reduction)
- `det`: 149 at epoch 1 → 2.1 at epoch 80 (71× reduction)
- `id_loss`: ~11 when first activated, slowly falls to ~0.5

If you hold the weight constant, the effective gradient contribution collapses with the loss.
The theoretically correct approach is an **adaptive weight** that keeps the contribution stable:

```
w_loss(epoch) = target_contribution / ema_loss(epoch)
```

where `ema_loss` is an exponential moving average of the raw loss value. This is complex to
implement. The practical proxy: **use a weight that increases as the loss decreases**, which is
what a ramp-down schedule achieves — you start low when the loss is huge (to prevent domination)
and raise it as the loss stabilises.

For `cls_kd` specifically: start weight 0.05, the loss starts at ~17,000 giving a contribution
of ~850. By epoch 15 the loss is ~295 and weight 0.05 gives contribution ~15. This is more
stable than starting at weight 0.25 (contribution 4,250 at epoch 1, overwhelming everything).

### Consistency between learning objective and checkpoint saving

Once you move to per-loss ramps, the "total val loss" becomes even less meaningful as a
checkpoint criterion — it changes shape every epoch as weights ramp. The checkpoint saving
must shift to a **metric-decoupled criterion**.

Proposal: Save `best.pt` based on `val_det` (the raw unweighted detection loss), not `val_loss`.
`val_det` is stable across all phases (56–63 in Run 3) and directly correlates with mAP. It
does not spike at phase transitions because it is never 0.

Additionally: save `epoch_{N}.pt` every 5 epochs regardless of criterion, starting from epoch 20.
This gives you a full history to retrospectively identify the best checkpoint by running
`eval_detection.py --sweep-threshold` against each saved file.

---

## 2. feat_kd — True Fix for Each Root Cause

### Root Cause 1: L2 normalization saturates MSE gradient

**Superficial fix**: remove `normalize=True` — creates scale instability across levels.

**True fix**: Replace the normalization + MSE with **per-channel z-score standardisation**:

```python
def _standardise(x):
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True) + 1e-6
    return (x - mu) / sigma
```

Then compute MSE on standardised tensors. This preserves relative channel differences
(unlike L2 which collapses all to unit norm) while making the loss scale-invariant.
The gradient is bounded by `1/sigma` per channel — nonzero even when features are
partially aligned, giving the adapter a persistent training signal.

Even better: replace MSE with **CKA loss (Centered Kernel Alignment)**. CKA measures
structural similarity of feature spaces and is invariant to rotation and scaling —
it works regardless of whether the student and teacher FPN have the same representation
style. This is the theoretically correct metric for cross-architecture feature alignment.

### Root Cause 2: Adapter gradient competes with detection head through shared FPN

**Superficial fix**: raise feat weight — still loses to cls_kd/det.

**True fix**: Detach the FPN features before passing them to the adapter:

```python
feat_kd = feature_distill_loss(
    {k: v.detach() for k, v in student_outputs["features"].items()},
    teacher_outputs["features"],
    adapters=self._adapters(),
)
```

This means `feat_kd` trains **only the adapter**, not the FPN. The adapter learns to
translate student features into teacher feature space. The FPN is not constrained to
produce teacher-like features — it is free to optimise for detection.

To guide the FPN toward teacher-compatible features without a shared gradient path,
add an **Attention Transfer (AT) loss** as a separate auxiliary:

```python
at_loss = attention_transfer_loss(
    student_outputs["features"],
    teacher_outputs["features"]
)
# Attention map: sum of squared activations across channels → (B, 1, H, W)
```

AT works on spatial attention maps (shape-independent), so it bridges the architectural
gap between ResNet50-FPN and YOLO26x without requiring channel projection. It provides
gradient directly to the FPN without conflicting with the adapter.

### Root Cause 3: Gradient is spatially diffuse — background dominates

**Superficial fix**: raise feat weight — amplifies background noise equally.

**True fix**: Apply feat_kd only at **foreground-assigned spatial cells** — the same
`pos_mask` used in `box_kd`:

```python
for level_name, stu in adapted.items():
    pos_mask = assignments[level_name].pos_mask.squeeze(1)  # (B, H, W)
    if not pos_mask.any():
        continue
    stu_pos = stu.permute(0, 2, 3, 1)[pos_mask]  # (N_pos, C)
    tea_pos = tea.permute(0, 2, 3, 1)[pos_mask]
    total = total + mse_or_cka(stu_pos, tea_pos)
```

This is **masked feature distillation**. The adapter only needs to align features where
pedestrians appear — foreground cells. Background cell features in the teacher are
near-zero and provide no meaningful alignment target. This makes the loss dense and
targeted rather than spatially diffuse.

### Root Cause 4: FPN serves conflicting objectives — detection vs feature alignment

**Superficial fix**: rebalance weights between det and feat — still both gradient through FPN.

**True fix (architectural)**: Add an **auxiliary adapter branch** that forks off the FPN
output *before* the detection head, with a stop-gradient boundary:

```
FPN output (pyramid_features)
    |
    +---> Detection head (focal + IoU loss)
    |
    +---> Adapter branch (detached) ---> feat_kd loss (trains only adapter)
    |
    +---> AT auxiliary (no detach) ---> AT loss (guides FPN toward teacher attention maps)
```

The adapter has its own gradient path. The detection head has its own gradient path.
The AT loss provides the only shared gradient that touches the FPN, and it's a weak
spatial signal that does not conflict with detection.

### Root Cause 5: p4/p5 adapter faces impossible channel doubling

**Superficial fix**: raise feat weight for all levels — amplifies the impossible tasks.

**True fix**: Apply feat_kd only on **p3**. The 384→384 adapter for p3 is near-identity
and achievable. Drop p4/p5 from feature distillation entirely.

For p4/p5, use AT loss (attention maps are (B,1,H,W) regardless of channels) instead
of channel-level MSE. This gives the FPN spatial guidance at all three levels without
requiring channel projection.

### Combined feat_kd fix — summary

The fixes are layered and complementary:

1. Replace MSE-on-normalised with MSE-on-z-score (or CKA) — fixes gradient saturation
2. Detach FPN before adapter — removes competition with detection head
3. Apply masked distillation (pos_mask only) — removes background noise
4. Add AT loss as FPN-level guide — gives FPN a non-conflicting spatial signal
5. Apply channel-level feat_kd only on p3, use AT on p4/p5 — removes impossible doubling

These fixes are independent. You can implement them one at a time.

---

## 3. ID Loss — Recommended Option

### Recommendation: Label Smoothing now (A), transition to SupCon (C)

**Why not elimination:** The downstream task is MOT16 tracking. Data association during
tracking requires discriminative embeddings. Eliminating `id_loss` in phase 3 would improve
detection mAP (by removing a conflicting gradient) but would produce an embedding space with
no discriminative structure — tracking quality would likely degrade substantially.

**Why not just Option B (cosine classifier):** Fixing normalisation of the linear head addresses
magnitude overfitting but not the closed-set problem. The model still memorises 500+ training
identities and fails to generalise to val-set identities.

**Why Option A + C together:**

Option A (label smoothing, ε=0.1) is a one-line change in `id_loss.py`:
```python
return F.cross_entropy(logits, labels, label_smoothing=0.1)
```
This immediately prevents the model from becoming infinitely confident on train IDs. Val
`id_loss` growth slows significantly. Zero risk, zero architecture change.

Option C (SupCon loss) is the principled fix. It replaces closed-set classification with
open-set metric learning. The key insight: you don't need to know all possible IDs in
advance — you just need crops of the same person to be closer than crops of different
people in embedding space. This generalises to val-set identities by construction.

With SupCon, the `id_classifier = nn.Linear(emb_dim, num_id_classes)` becomes unnecessary.
This also eliminates `teacher_id_classifier` — instead, the teacher projector is trained
by SupCon directly on its embeddings. The `emb_kd` loss (cosine similarity between student
and teacher embeddings) works even better when the teacher projector is trained by SupCon
because its embedding space becomes metric-structured rather than logit-structured.

**Implementation path:**
- Run 4: Add label smoothing (ε=0.1) to existing `F.cross_entropy`. Reuse existing architecture.
- Run 5: Replace `F.cross_entropy` with SupCon loss. Remove `id_classifier` and
  `teacher_id_classifier`. The SupCon loss batches positive/negative pairs from within-batch
  track IDs — any two crops with the same `track_id` are a positive pair.

---

## 4. Transition Spikes — Concrete Per-Loss Ramp Schedule

### Design principles

1. **One loss activates at a time.** Never introduce two new losses simultaneously.
2. **Ramp duration ≥ 5 epochs.** Adam's β2=0.999 needs ~7 epochs to re-calibrate squared
   gradient estimates after a new gradient direction appears.
3. **Later-tier losses activate after earlier-tier losses stabilise**, not at a fixed epoch number.
4. **Some losses should be ramped out**, not just ramped in — closing the training with
   detection focus, not ID-classification focus.

### Proposed 80-epoch schedule

All weights are scalars applied to the raw (unweighted) loss value. "Ramp" means linear
interpolation between the two values over the stated epoch range. Values outside stated
ranges hold at the boundary value.

#### `det` (detection loss)

- Epochs 1–3: 0.3 (gentle start; backbone is pretrained but needs warmup)
- Epochs 3–8: ramp 0.3 → 1.0
- Epochs 8–60: hold 1.0
- Epochs 60–75: ramp 1.0 → 2.0 (emphasise detection in final stretch)
- Epochs 75–80: hold 2.0

Rationale: Always active, never zero. The detection signal must never be interrupted.

#### `cls_kd` (teacher soft classification)

- Epochs 1–3: 0.03 (raw loss ~17,000; contribution ~510, manageable)
- Epochs 3–15: ramp 0.03 → 0.1 (loss falls to ~295; contribution ~30, stable)
- Epochs 15–50: hold 0.1
- Epochs 50–65: ramp 0.1 → 0.03 (fade out as detection matures)
- Epochs 65–80: hold 0.03

Rationale: `cls_kd` trains the student to imitate the teacher's per-cell confidence
distribution. Useful early for knowledge transfer. Should fade out late to let `det`
dominate — we want the model to find GT pedestrians, not just copy teacher confidence maps.

#### `box_kd` (teacher box regression copy)

- Epochs 1–7: 0.0 (let `det` establish stable positive-cell assignments first)
- Epochs 7–12: ramp 0.0 → 0.4
- Epochs 12–65: hold 0.4
- Epochs 65–75: ramp 0.4 → 0.2
- Epochs 75–80: hold 0.2

Rationale: `box_kd` only has a signal at positive cells (from `pos_mask`). Before those
assignments stabilise (roughly epoch 7–8 based on when `det` flattens), `box_kd` adds
noise. Activating after `det` stabilises and fading late ensures it doesn't compete with
detection in the final epochs.

#### `feat_kd` (feature map alignment, p3 only after fix)

- Epochs 1–8: 0.0 (let FPN stabilise; activating before this wastes adapter capacity)
- Epochs 8–13: ramp 0.0 → 5.0 (high weight while FPN is still plastic)
- Epochs 13–40: hold 5.0
- Epochs 40–55: ramp 5.0 → 1.0
- Epochs 55–70: hold 1.0
- Epochs 70–80: ramp 1.0 → 0.0 (full fade-out; detection is the focus)

Rationale: The adapter needs high weight while the FPN is still changing, to shape the
feature space. As the FPN converges (around epoch 40), the adapter's job is done and the
weight should decay to avoid over-constraining the FPN away from detection objectives.

#### `emb_kd` (embedding cosine alignment)

- Epochs 1–22: 0.0 (teacher projector not yet trained; targets are meaningless)
- Epochs 22–27: ramp 0.0 → 0.3
- Epochs 27–70: hold 0.3
- Epochs 70–80: ramp 0.3 → 0.5 (final emphasis on embedding quality for tracking)

Rationale: `emb_kd` depends on the teacher projector being discriminative. The teacher
projector is trained by `id_loss`, which activates at epoch 15. Give `id_loss` 7 epochs
(until epoch 22) before relying on the teacher projector's embeddings as targets.

#### `id_loss` (identity classification / SupCon)

- Epochs 1–15: 0.0 (embeddings are not structured enough for ID classification to work)
- Epochs 15–20: ramp 0.0 → 0.2
- Epochs 20–55: hold 0.2
- Epochs 55–70: ramp 0.2 → 0.05 (fade out; let `emb_kd` carry the embedding signal)
- Epochs 70–80: hold 0.05

Rationale: `id_loss` activates first among the Tier-3 losses (before `emb_kd`) because it
trains the teacher projector — `emb_kd` depends on it. The fade-out in late training reduces
the closed-set overfitting effect and shifts embedding supervision to the metric-based `emb_kd`.
With label smoothing + eventual SupCon, the 0.05 residual weight in the final epochs is safe.

### Activation order at a glance

```
Epoch 1  : det (ramping up), cls_kd (ramping up)
Epoch 7  : box_kd begins ramping in
Epoch 8  : feat_kd begins ramping in
Epoch 15 : id_loss begins ramping in (one new loss only)
Epoch 22 : emb_kd begins ramping in (teacher projector now partially trained)
Epoch 40 : feat_kd begins ramping down
Epoch 50 : cls_kd begins ramping down
Epoch 55 : id_loss begins ramping down
Epoch 60 : det begins ramping up (final detection emphasis)
Epoch 70 : feat_kd → 0, emb_kd ramps up, id_loss at minimum
```

Each activation is separated by at least 5–7 epochs from the previous one.
No two losses change activation status in the same epoch window.

### How this translates to config

The current `phases` list (3 blocks) becomes a per-loss ramp specification. This requires
a code change to `PhaseWeightSchedule.for_epoch()` to support per-loss linear interpolation
rather than a single weight dict per phase. The config shape changes from:

```yaml
phases:
  - start_epoch: 1
    end_epoch: 15
    weights: {det: 0.5, ...}
```

to a ramp-table per loss:

```yaml
loss_schedule:
  det:
    - [1, 3, 0.3, 0.3]      # [start, end, w_start, w_end]
    - [3, 8, 0.3, 1.0]
    - [8, 60, 1.0, 1.0]
    - [60, 75, 1.0, 2.0]
    - [75, 80, 2.0, 2.0]
  cls_kd:
    - [1, 3, 0.03, 0.03]
    ...
```

This format fully specifies piecewise-linear weight trajectories for all six losses
independently.

### Why this solves the checkpoint consistency problem

With this schedule, `val_det` is the dominant and most stable component from epoch 8 onward.
It never spikes at a phase boundary (because `det` is always active). The `best.pt` saver
on `val_det` will correctly track epochs where the detection head is best.

The total `val_loss` will still vary (as `emb_kd` ramps up at epoch 22, for example), but
the `best.pt` criterion ignores that — it watches `val_det` independently.

---

## Implementation Priority

1. **Add label smoothing to `id_loss`** — one line, run 4 (ε=0.1)
2. **Change `best.pt` criterion to `val_det`** — one line in `trainer.py`
3. **Add periodic checkpoint saving every 5 epochs** — 3 lines in `trainer.py`
4. **Implement per-loss ramp schedule** — new `PhaseWeightSchedule` design
5. **Fix feat_kd: detach FPN + masked distillation + z-score** — `kd_loss.py` + `trainer.py`
6. **Add AT loss for p4/p5** — new loss function
7. **Replace id_loss with SupCon** — `id_loss.py` refactor, remove id_classifier
