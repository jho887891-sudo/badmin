# Level-2 input resolution: measured, and negative — plus a correction to the appearance claim

## 1. Input resolution (spec 07 section 3.2), inference-only, same weights, same images

| imgsz | real-positive recall | FP on 30 shuttle-free scenes | clean scenes | median ms/image | peak VRAM |
|---|---|---|---|---|---|
| **640** | 0/10 | **46** | **9/30** | 43.7 | 111 MB |
| 960 | 0/10 | **83** | 7/30 | 39.1 | 142 MB |
| 1280 | 0/10 | **113** | 6/30 | 48.8 | 201 MB |

Raising the input resolution does not help the real photographs at all - 0/10 at every size - while it
more than doubles the false positives (46 -> 113) and costs 81% more VRAM. This closes the first Level-2
item as a clear negative. Spec 07 section 3.2 requires latency and VRAM alongside recall, and both are
reported; the median is dominated by file reads on this slow mount, so it is an upper bound on the
pure inference cost rather than a clean latency figure.

The mechanism for the false-positive increase is plausible and unmeasured: a larger input gives the
small-white-blob textures in those scenes more pixels to look like a shuttlecock. It is recorded as a
hypothesis, not a finding.

## 2. Correction: I overstated the appearance gap, and showed an unrepresentative crop

`REAL_VS_SYNTHETIC_FINDING.md` said the synthetic renders "read as a plastic or metal cone" and offered
that as visible proof that appearance explains the real-image failure. Having now rendered the same
geometry at native scale with a face-on pose, **that claim was wrong and the figure was not
representative**.

What `renderer_check.png` and `renderer_comparison.png` show: the project renderer at 702 px produces
**distinct feather panels with gaps between them, a visible rib lattice, and a rounded cork** - a
recognisable shuttlecock. The impression of a smooth cone in the earlier figure came from two choices of
mine: a 90-degree side view, where the feathers are seen edge-on and naturally look like a fan, and a
crop centred on the image rather than on the subject, which magnified a corner of the object.

Using the SAME asset, the Isaac Sim preview is brighter and more evenly lit than the project renderer,
and the real photographs are brighter still with fine feather structure. So the honest position is:

- **geometry: faithful.** The renderer reproduces the asset correctly, feathers and gaps included.
- **material and lighting: simpler.** The project renderer uses a near-white albedo
  (`FEATHER_ALBEDO = (0.93, 0.94, 0.90)`, matching the asset's declared White material) with a
  wrapped-diffuse model, so lit faces reach about 237/255 and unlit ones fall to about 130/255. The real
  photographs, and the Isaac Sim render, are brighter and more even.
- **fine structure: absent, because the asset has no texture.** This part of the earlier finding stands:
  the GLB carries only POSITION and NORMAL, no TEXCOORD_0 and no texture, so no renderer can produce
  individual feather barbs from it.

**What this means for the conclusion:** the appearance difference is real but subtler than I portrayed,
and **this figure no longer supports the claim that appearance is sufficient to explain the 0.000 real
recall.** What is established is the elimination: size is excluded (round 3), resolution is excluded
(this round), and threshold is excluded (Level 1). The remaining candidate is still the real domain, but
the specific mechanism is not proven.

## 3. What would actually settle it, and it is not another render

Two candidate measurements, both of which need something the project does not have:

1. **real close-up photographs of a shuttlecock** at a known camera pose and distance, which would
   separate appearance from perspective in one shot. Every real image the project owns is frozen test
   data and cannot be trained on.
2. **a rendered training pool produced by a different renderer** - the Isaac Sim preview proves the
   project can render this asset far better than the numpy path does. If the numpy renderer's material
   simplification is the cause, training on Isaac Sim renders should move the real recall; if it does
   not, the cause is elsewhere. That is a large piece of work and it is the single most informative
   experiment still available.
