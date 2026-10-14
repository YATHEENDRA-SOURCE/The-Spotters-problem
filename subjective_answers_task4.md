# Task 4: Subjective Answers

Output of `task4.py`:

```
scene             matches   votes      IoU    agree?
scene_easy            316      15    0.000     0.000
scene_medium          457      18    0.948     0.948
scene_rotated         403      14    0.599     0.000
scene_hard            353      14    0.943     0.943
```

So the feature method finds Waldo in medium, hard and **rotated**, and fails on **easy**. On easy, the biggest cluster of votes lands on another character near (733, 84), who wears the same red-and-white striped shirt with a grey bobble hat. In grayscale, that person looks a lot like Waldo. Task 2 (in colour) got easy right and rotated wrong, so the two methods fail on opposite scenes.

## 4.6 Why the median fails

The change: in `find_waldo` I replaced the densest-cluster step with a plain median over every prediction:

```python
cx, cy = np.median(predictions[:, :2], axis=0)
scale = np.median(predictions[:, 2])
box = box_from_centre(cx, cy, scale, big.shape[:2])
```

| scene | predictions | within 12 px of Waldo | median centre | true centre | IoU (median) |
| --- | --- | --- | --- | --- | --- |
| scene_easy | 316 | 10 (3.2%) | (481, 340) | (627, 581) | 0.000 |
| scene_medium | 457 | 18 (3.9%) | (542, 382) | (689, 170) | 0.000 |
| scene_rotated | 403 | 13 (3.2%) | (442, 362) | (253, 322) | 0.000 |
| scene_hard | 353 | 14 (4.0%) | (506, 415) | (146, 425) | 0.000 |

IoU is **0 on every scene**. The median centre always lands near the **middle of the image** (the images are 1000x700, so the middle is 500, 350), wherever Waldo actually is.

Why: a median only gives a sensible answer if **more than half** of the data is correct. That's its "breakdown point", 50%. Below that, the wrong values decide where the middle is. Here only **3–4%** of the predictions are correct. The other 96% come from wrong matches, which point at other people spread all over the picture. So the median of all predictions is just the middle of that spread-out crowd, roughly the centre of the image. The 10–18 correct votes on Waldo are far too few to pull it.

The densest-cluster method works because it doesn't need a majority. It just needs the correct votes to agree with *each other* more tightly than the wrong ones do. Wrong votes are scattered, so even 14 votes that agree within 12 px can win.

## 4.7 The abnormality: each method fails where the other does not

### (a) What's missing from Task 2's search space

**Rotation.** Task 2 searches over position (x, y) and size (the 8 values in `SCALES`), but the template is always upright. In scene_rotated, Waldo is tilted by about 31°, so his stripes are diagonal. The upright template has horizontal stripes, and at the true position the stripes cross each other instead of lining up, so NCC is low. Task 2 then picked an upright person somewhere else (score 0.829, IoU 0).

Why 20 more scales can't help: resizing only makes the template bigger or smaller. It never turns it. Every template in the search is upright with horizontal stripes, whatever its size, so no entry in `SCALES` produces a tilted Waldo to compare against. Even the shape is wrong: rotated Waldo's bounding box is 50x65, while every scaled template keeps the 1:2 shape (28x56). More scales would just mean more chances for an upright lookalike to get a lucky high score. You'd need to add rotation as another dimension of the search (many angles x many scales), which gets very slow.

### (b) The step that gives Task 4 rotation

It's **`dominant_orientation`**, together with passing those angles into `describe_keypoints`. Each keypoint gets its own angle (the peak of its gradient-direction histogram), and the descriptor is sampled on a grid **rotated by that angle**, with gradient directions measured **relative** to it. So when Waldo is rotated by 31°, each of his keypoints' angles also shifts by about 31°, and the descriptor comes out the same as upright. The rotation is "subtracted out". Then `predict_centres` uses the angle difference to rotate the offset to the centre. In my rotated run, most of the correct matches had an angle difference of about +30°, which fits Waldo's tilt.

With `orientations=None` (I also set the angles to 0 so the voting ignores rotation), I measured:

| scene | normal IoU | upright IoU |
| --- | --- | --- |
| scene_easy | 0.000 | 0.899 |
| scene_medium | 0.948 | 0.000 |
| **scene_rotated** | **0.599** | **0.000** |
| scene_hard | 0.943 | 0.943 |

On scene_rotated it **fails** (IoU 0). The upright grid sees Waldo's diagonal stripes as diagonal, which doesn't match the template's horizontal ones, and the votes that do happen don't account for the tilt. So it loses exactly the case that Task 4 is supposed to win. (A surprise I have to report: upright descriptors actually *worked* on easy. I think that's because the orientation estimate on stripes is sometimes off by 180°, since stripes have gradients pointing both up and down. In an upright scene, skipping orientation avoids those mistakes. But it gave up medium and rotated, so orientation is still worth it overall.)

### (c) Why the more capable method is the less precise one

Task 2 returns **exact** things. It literally tried the template at that exact pixel and that exact size, so the box is the exact rectangle it tested and the score is the real NCC there. When it's right, it's pixel-perfect (IoU 1.000 on easy, medium and hard).

Task 4 only **estimates**:
- the **centre** is the median of a cluster of votes, and each vote is built from keypoint positions (rounded to whole pixels) plus an offset that is scaled and rotated;
- the **scale** comes from sigma ratios, and sigma only takes the steps of my ladder (factors of 1.15), so it's rounded;
- the **angle** comes from a 36-bin histogram, so it's rounded to 10°;
- and the box is drawn with the template's upright 1:2 shape, even though on the rotated scene the true box is 50x65.

All of these small errors add up, so on scene_rotated it finds the right person but the box only gets IoU 0.599, and on medium and hard it gets about 0.94–0.95 instead of 1.0. Being able to handle rotation and scale comes from summarising the object into a few keypoints, and that summary is less exact than comparing every pixel.

## 4.8 The check that every tutorial recommends

The line I added in `find_waldo`, right after `match_ratio_test` (`rows` are scene indices, `j1` the matched template indices):

```python
distances = descriptor_distances(desc_s, desc_t)
rows, j1 = matches[:, 1], matches[:, 0]
matches = matches[np.argmin(distances, axis=0)[j1] == rows]   # <- added
```

| scene | n_matches without | n_votes without | n_matches with | n_votes with |
| --- | --- | --- | --- | --- |
| scene_easy | 316 | 15 | 39 | 0 |
| scene_medium | 457 | 18 | 36 | 0 |
| scene_rotated | 403 | 14 | 38 | 0 |
| scene_hard | 353 | 14 | 27 | 0 |

The "recommended" check **broke everything**. Matches dropped by about 90%, and **no scene got a detection at all** (0 votes means no cluster reached 8). So all three correct detections were lost.

Why: the mutual check allows **at most one scene keypoint per template part**. It keeps a match only if that scene point is the single best one *in the whole scene* for its template part. But in this problem, many scene keypoints can legitimately correspond to the same template part:

- **Within Waldo himself:** his shirt has several identical stripes. One template stripe part really does match 2–3 stripes on the scene Waldo, and all of those matches are correct. Similar parts can also be detected at neighbouring sigma levels.
- **Across the crowd:** there are many striped people. For each template part, the single best scene point out of 3000 is quite likely on some *other* person. So even the matches that survive are spread across different people instead of piling up on Waldo.

A one-to-one pairing is the right model when both images show **the same single thing**, like two photos of the same building for stitching, where each point should appear once in each image. Here the template is a small dictionary of about 80 parts, and the scene contains one Waldo plus hundreds of people made of similar parts. It's a one-to-many situation. Each scene point asking "which part am I?" (with no mutual check) keeps all the evidence, and the voting then sorts out which ones agree. The mutual check throws away most of the correct votes before the voting even gets to see them.

## 4.9 Agreement without an answer key

The last column is the IoU between my Task 2 box and my Task 4 box:

| scene | agreement | Task 2 correct? | Task 4 correct? |
| --- | --- | --- | --- |
| scene_easy | 0.000 | yes | no |
| scene_medium | 0.948 | yes | yes |
| scene_rotated | 0.000 | no | yes |
| scene_hard | 0.943 | yes | yes |

**Why disagreement is useful even without ground truth:** there's only one Waldo, so if the two boxes don't overlap, they can't *both* be on him. At least one detector is wrong. That's a free warning sign ("check this one by hand") without needing any labels. But it can't tell you *which* one is wrong. In my results, easy and rotated both have agreement 0, and in easy Task 2 is right, while in rotated Task 4 is right. The number looks the same in both cases.

**Agreement doesn't prove correctness either.** Two methods can agree and both be wrong if they fail for the *same reason*. A concrete case: imagine a scene where the real Waldo is tilted by 30° **and** half hidden behind a sign, and somewhere else there's an upright decoy wearing exactly Waldo's outfit (red-white striped shirt, red-white bobble hat, glasses) at one of the template scales.
- Task 2 can't search rotations, so it can't see the tilted Waldo, and the upright decoy gets the best NCC. It picks the decoy.
- Task 4 can handle the tilt, but the sign hides half of Waldo's keypoints, while the fully visible decoy gives more matching parts and more votes. It also picks the decoy.

Both boxes land on the decoy, agreement is about 1.0, and both are wrong. The methods are built differently, but they rely on the same clue (the striped outfit), so a good enough copy of that clue fools both at once. Agreement is only strong evidence when the two methods' mistakes are truly independent.
