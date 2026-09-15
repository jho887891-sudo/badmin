# The real-domain gap is APPEARANCE, and it is now visible rather than inferred

Date: 2026-09-15  |  Figure: `real_vs_synthetic.jpg`  |  Data: the verified real positives vs the
near-field synthetic renders at comparable size.

## 1. Two corrections to the frozen real-image set, found while building this figure

**The "10 verified real positives" are 6 distinct images.** Four of the ten rows are byte-identical
duplicates under different filenames, confirmed by sha256 rather than by eye:

| row | sha256 (first 12) | status |
|---|---|---|
| real_001, real_014, real_037 | `b59ff4ffac1f` | the SAME file three times |
| real_002, real_015, real_038 | `6a6b1cc3abc7` | the SAME file three times |
| real_008, real_017, real_020, real_042 | distinct | four unique images |

So every earlier statement of the form "recall 0.000 on the 10 verified real positives" overstates the
evidence: it is **0.000 on 6 distinct photographs**. The frozen set carries duplicates that inflate its
row count. That is a defect in the frozen set, recorded here rather than quietly re-denominated.

**One of the six is a plastic shuttle.** real_017 shows yellow-green nylon shuttles on a racket, not a
feathered cork-and-feather shuttlecock. It is legitimately a shuttlecock, but it is a different
appearance class from the other five, and it should be counted separately rather than pooled.

## 2. The figure, and what it shows

`real_vs_synthetic.jpg` puts the six distinct real positives in the top row with their annotated boxes,
and six synthetic near-field renders at 512 px and 800 px in the bottom row.

The difference is not subtle and it is not a matter of resolution:

| | real photographs | synthetic renders |
|---|---|---|
| feather skirt | individual translucent feathers, separated, with visible fine structure and gaps | smooth opaque panels with flat facets, no separation |
| cork | white/cream, distinct rounded shape | uniform flat grey |
| surface | photographic texture, self-shadowing between feathers | flat-shaded polygons, no texture at all |
| overall read | unmistakably a feathered shuttlecock | reads as a plastic or metal cone |

## 3. Why this is the answer, and why it was predictable

The mechanism was already measured and recorded earlier in this project: **the shuttlecock GLB contains
only POSITION and NORMAL attributes - no TEXCOORD_0 and no texture** - so the renderer can produce
nothing but flat-shaded polygons. A real shuttlecock is largely defined by translucent feather
structure, which that asset cannot express at any resolution.

This closes the question left open by the round-3 result. Round 3 showed that extending the training
range to 807 px moved the synthetic detection ceiling past 1023 px, while the real photographs stayed at
0.000 - so size was excluded. The remaining candidates were appearance and perspective, and the figure
shows the appearance difference is large enough on its own to explain the failure.

Perspective is not excluded and is not claimed to be: the real positives are close-ups and the S7 ladder
is a long-lens render. But it is no longer NEEDED to explain the result, and appearance is now supported
by direct evidence rather than by elimination.

## 4. What this changes

- **The binding constraint is real-domain training data**, and now there is a visible reason rather than
  a statistical one. The project owns no real training images; every real image it has is frozen test
  data.
- **Rendering more synthetic large targets will not fix it.** Two data returns have now been spent on
  the synthetic axis and the axis is closed: 1.000 recall from 192 px to 1023 px.
- **A texture-less asset is the root cause on the rendering side.** Giving the shuttle real feather
  appearance would require a re-authored visual asset, which is a different piece of work from anything
  the detection subproject controls.
- **The Level decision does not change**: this is still a Level 1/2 data problem, not architecture.
