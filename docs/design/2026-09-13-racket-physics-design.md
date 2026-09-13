# Badminton Racket Physics Asset Design

## 1. Goal

Build a traceable badminton-racket asset pipeline for Isaac Sim / Isaac Lab that separates final visual geometry, physical collision geometry, mass properties, racket contact frame, and the PiPER fixed attachment.

## 2. Source policy

- BWF Laws define legal upper bounds, not the exact geometry of a specific racket: overall frame length <= 680 mm, overall width <= 230 mm, stringed area <= 280 mm x 220 mm (subject to the throat-extension exception).
- A real product reference is used only for fields the manufacturer actually publishes. VICTOR THRUSTER Onigiri publishes 675 mm length and a 6.4 mm shaft, with 4U as a supported weight class. VICTOR defines 4U as 80.0--84.9 g unstrung.
- The selected open-license visual source is the previously chosen Sketchfab model by game_travel, CC Attribution. The racket mesh must be extracted separately before final visual authoring.
- Product pages do not publish the exact head outer dimensions, handle dimensions, balance point, whole-racket COM or inertia tensor for the exact physical unit that will be mounted to PiPER. Those fields therefore remain measured-unit inputs instead of invented constants.

## 3. Coordinate convention

Racket-local convention:

- Origin: `racket_tcp`, the mechanical attachment / tool-center point.
- +X: racket-face normal, pointing toward the opponent in the canonical hit pose.
- +Z: from handle toward the racket head.
- +Y: right-handed completion, across racket width.
- `racket_contact_frame`: center of the effective string-bed/sweet-spot contact region, same nominal orientation as `racket_tcp`.

The builder must expose `T_tcp_contact`. `build_piper_with_racket.py` adds `T_link6_tcp`; composition yields `T_link6_contact`.

## 4. Physical geometry

Final physics is not authored from a single box or a visual triangle mesh.

- Handle collider: rounded/cylindrical proxy fitted to measured handle geometry.
- Shaft collider: cylinder using the traceable shaft diameter or measured installed geometry.
- Head frame collider: segmented elliptical/closed-frame proxy fitted to measured outer head dimensions.
- String-bed collider: thin convex elliptical plate inside the head. It is the primary shuttle-contact surface and is independent from the visual strings.

The collision proxy is intentionally simpler than the visual mesh for stable high-rate contact and vectorized simulation, but its envelope must be fitted to measured geometry. The simplification may reduce grommet/frame-edge detail; that loss is acceptable only after dimensional validation.

## 5. Mass properties

Final mode requires physical-unit measurements:

- installed racket mass (including strings/grip/adapter portions represented by this asset),
- balance point / COM along +Z relative to `racket_tcp`,
- either a measured inertia tensor or permission to derive a proxy inertia from the measured geometry and calibrated mass split.

No final USD may silently use a category midpoint as if it were measured truth. A TEMP/debug mode may use a clearly labelled reference estimate only when explicitly requested.

## 6. Visual asset

Final visual source:

- Title: Badminton Racket And Shuttlecock (Low Poly)
- Author: game_travel
- License: CC Attribution
- UID: 795e4cca2e544d7ab86a8e5c7ded2362
- URL: https://sketchfab.com/3d-models/badminton-racket-and-shuttlecock-low-poly-795e4cca2e544d7ab86a8e5c7ded2362

Extract the racket only and convert to `assets/third_party/racket_visual.usd`. Record source URL, author, license, retrieval date and SHA256 in the project third-party asset manifest.

## 7. PiPER attachment

`build_piper_with_racket.py` composes the frozen PiPER no-gripper USD and the racket USD into a new stage. The racket is attached with a `UsdPhysics.FixedJoint` between the composed PiPER link6 body and the racket rigid body. The mount transform is configuration data and must be replaced with the measured adapter transform on hardware.

The source PiPER USD is referenced, not modified.

## 8. Acceptance

- BWF legal envelope is validated.
- Product-reference fields exactly match their cited manufacturer values.
- Missing physical-unit measurements block final mass/physics authoring.
- `racket_tcp`, `racket_contact_frame`, face normal, and transform composition are deterministic.
- String-bed collider is independent and finite.
- PiPER wrapper preserves the source robot and uses a FixedJoint.
- USD authoring is optional in ordinary Python and tested when `pxr` is available.
