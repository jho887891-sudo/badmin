# Level-1 threshold operating point scan — the missing item before Level 2

Spec 07 section 2.4 makes "the threshold operating point has been scanned" an entry condition for
Level 2, and every measurement in this project until now used a fixed conf >= 0.05. Measured with
`baseline_round3` on the same images at each threshold:

| confidence | real-positive recall | FP on 30 shuttle-free scenes | clean scenes | S8 size-curve recall |
|---|---|---|---|---|
| **0.05** (used everywhere so far) | 0/10 | **46** | 9/30 | 0.466 |
| 0.10 | 0/10 | 32 | 13/30 | 0.434 |
| 0.15 | 0/10 | 24 | 16/30 | 0.422 |
| 0.20 | 0/10 | 22 | 17/30 | 0.406 |
| 0.25 | 0/10 | 21 | 17/30 | 0.400 |
| 0.30 | 0/10 | 18 | 20/30 | 0.388 |
| **0.40** | 0/10 | **11** | **22/30** | 0.344 |
| 0.50 | 0/10 | 11 | 22/30 | 0.322 |
| 0.60 | 0/10 | 7 | 24/30 | 0.284 |
| 0.70 | 0/10 | 4 | 26/30 | 0.228 |

## 1. The threshold is a real lever for precision, and a real cost for recall

At conf 0.40 the false positives on shuttle-free scenes fall from 46 to 11, a **76% reduction**, and
the number of completely clean scenes rises from 9 to 22 of 30. The price is the S8 size-curve recall,
which falls from 0.466 to 0.344.

That is a genuine operating-point decision rather than a free win, and it is now a decision the human
partner can make from numbers: which matters more, keeping recall across 4-512 px, or cutting spurious
detections by three quarters. The earlier measurements at conf 0.05 were pessimistic about precision
and optimistic about recall by the same choice, and both halves of that are now visible.

## 2. And it does nothing for the real photographs

**Real-positive recall is 0/10 at every threshold from 0.05 to 0.70.** A confidence threshold cannot
create a detection the model never produces. This is worth stating plainly because it removes threshold
tuning from the list of possible fixes for the real-domain failure, which the previous round traced to
appearance.

## 3. Two honest caveats about this scan

**The denominator is 10 rows, not 6 distinct images.** This script de-duplicated by filename, and the
four duplicates in the frozen set carry DIFFERENT filenames (real_001 vs real_014 vs real_037) with
identical sha256. So the scan measured 10 rows that represent 6 distinct photographs, and the
"0/10" column should be read as "0 of 6 distinct images, appearing as 10 rows". The result does not
change - 0 is 0 either way - but the evidence count does.

**The S8 recall column is computed on the whole S8 sweep**, which pins one pose and one scene at a time
and spans 4-512 px, so it is a summary rather than a per-bucket curve. A per-bucket threshold curve
would show where the cost falls, and the small buckets are where it will fall first.

## 4. Where this leaves spec 07

Entry conditions for Level 2, item by item:

| Condition | Status |
|---|---|
| data coverage sufficient | yes for the synthetic axis, which is now closed to 1023 px; NOT for the real domain, where the project owns no training images |
| labels verified | yes, audits pass on every manifest |
| targeted training executed | yes, two data-return rounds with retrains |
| **threshold operating point scanned** | **done by this file** |
| fixed test still not meeting requirements | yes, and the remaining failure is real-domain appearance |

Level 1 is therefore complete, with one exception that is not a measurement: there is no real-domain
training data to add. Level 2 compares input resolution, ROI and training organisation - none of which
addresses an appearance gap either, and the honest reading is that the next productive step is real
data, not Level 2.
