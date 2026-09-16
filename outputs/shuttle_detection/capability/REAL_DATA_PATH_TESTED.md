# The real-training-data path, tested: ad-hoc web acquisition cannot close the gap

Date: 2026-09-15  |  Attempt: acquire REAL shuttlecock photographs and auto-label them with the candidate
model, as the Level-1 data adjustment spec 07 section 2.2 names as "补真实数据".

## 1. What was acquired, and what it actually contained

22 candidates downloaded from Wikimedia Commons across eight searches for shuttlecock imagery, with
provenance and licence recorded per image. Reviewed on a contact sheet with the model's own proposals
drawn on them:

| | count |
|---|---|
| candidates acquired | 22 |
| **actually showing a shuttlecock** | **12** |
| not a shuttlecock | 10 |

The ten rejects are instructive about the source rather than about the search: seven are historical
engravings, paintings or line drawings, two are people in badminton poses with no shuttle visible, and one
is an empty sports hall.

## 2. The auto-labelling step did not work, and that is the finding

The plan was to use the candidate model to propose boxes and verify them visually rather than label by
hand. On the 12 real shuttles:

| | count |
|---|---|
| model proposed a box | **8 of 12** |
| of those, confidence >= 0.5 | **4** |
| model proposed NOTHING | 4 (rt_005, rt_006, rt_007, rt_013 - all clear, well-lit shuttlecock photos) |

A model that misses a third of deliberately close-up, well-lit shuttlecock photographs is not a labelling
tool. Note what this also means in passing: those are LARGE, obvious shuttles, and the model still missed a
third of them - a reminder that the 0.600 real-photo recall is a small-sample figure.

## 3. The scale problem, which is what actually settles it

**Four auto-labelable real images against 922 training positives is 0.4%.** Even adding all eight proposed
boxes - accepting the imprecise ones - reaches 0.9%. Neither is a quantity that can move a model trained on
922 positives, and testing it would cost a 100-epoch retrain and four protocol runs to measure a change
that is very likely inside noise.

So the honest conclusion is not "we tried and it did not help" but **"this route cannot produce enough data
to try"**.

## 4. What this means for the project

The real-domain gap has been characterised precisely: the release detects 0.600 of real photographs,
everything achievable with synthetic data has been done, and the binding constraint is real training data -
which this project holds NONE of, because every real image it owns is frozen test data.

This attempt shows that the shortage is not for lack of trying the obvious thing. Web acquisition yields a
handful of usable images against hundreds of training positives, and its labelling cannot even be
automated with the model under test.

**The unblocking action is therefore specific and it is not an engineering task: real photographs from the
actual deployment camera, in the actual working distance range, taken in numbers (hundreds, not dozens),
with boxes drawn by whoever operates that camera.** That would also settle the appearance-versus-perspective
question that has been open since the round-3 result, because a real close-up at a known distance separates
them in one shot.

## 5. Artefacts kept

`outputs/shuttle_capability/real_train/` holds the 22 images, their licence metadata and the model's
proposals (`proposals.json`), and `outputs/evidence/real_train_proposals.jpg` is the contact sheet they
were judged on. They are kept so the judgement can be audited rather than taken on trust, and so the
attempt is in the record.
