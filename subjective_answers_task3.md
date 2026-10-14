# Task 3: Subjective Answers

Output of `task3.py`:

```
Harris corners on the template : 9
DoG keypoints on the template  : 29
sigmas found                   : [1.   1.25 1.56 1.95]
descriptors                    : (29, 128) row norms in [1.000, 1.000]
DoG keypoints on scene_easy    : 3000
```

My choices: `TEMPLATE_KEYPOINTS = 80` (the question sheet talks about "~80 template parts", and the 2x template gives about 79 anyway), and `SCENE_KEYPOINTS = 3000`. With 100–230 people per scene, that's roughly 15–20 keypoints per person, enough to cover Waldo. When I tried 5000 or 8000, the extra keypoints mostly added votes to the wrong people.

Another thing I fixed: at first `detect_corners` gave 53 corners on the template, but many were just the weak "shoulders" right next to a strong corner (on a plain white square it returned 24 points instead of 4). I added a check that a corner must be the largest Harris value in its own neighbourhood, and now the square gives exactly 4 and the template gives 9 clean corners (hat, glasses, chin, shoulders, trousers).

One bug I hit that's worth mentioning: at first I kept a DoG point if it was `>=` all its neighbours. Because these scenes are synthetic, a perfectly straight edge has *exactly* equal values along it, so a whole vertical line of 21 "keypoints" tied as maxima. They all matched the same template part and outvoted Waldo. Changing it to **strictly** greater than all 26 neighbours fixed that.

## 3.6 Corner or blob?

The third number, sigma, comes from the **scale-space ladder**. I blur the image with a series of Gaussians (sigma0, sigma0·k, sigma0·k², …), take the difference of neighbouring levels, and keep a point only if it's the biggest (or smallest) value among its 26 neighbours: 8 around it at the same level, 9 in the level above, and 9 below. The level where that happens tells me *how big the blob is*. A small dot responds most at a small sigma, a big dot at a larger sigma. So sigma is basically the size of the feature.

Harris doesn't have this. It's computed at one fixed blur (sigma_d = 1, sigma_i = 2), so it only says "there is a corner here", not how big the structure around it is. That's why it's (y, x) only.

Why Task 4 can't work with Harris alone:
- `predict_centres` needs s = sigma_scene / sigma_template to know how much bigger or smaller Waldo is in the scene, so it can scale the offset from the keypoint to his centre. With Harris, `with_unit_scale` gives every point sigma = 1, so s is always 1. On scene_medium (scale 1.35) and scene_hard (1.5), a correct match would point to the wrong place, because the offset isn't stretched. Different parts of Waldo would disagree about where his centre is, so they wouldn't form a cluster.
- The descriptor window also scales with sigma. Without a sigma, the same part of Waldo at two different sizes gets described over different-sized neighbourhoods, so the descriptors don't match well in the first place.

## 3.7 The ladder is a trade

I ran `dog_keypoints` on the native template (28x56) with different settings:

| sigma0 | n_scales | k | keypoints | sigma range found | top of ladder |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 8 | 1.25 | 29 | 1.00 – 1.95 | 3.81 |
| 0.5 | 8 | 1.25 | 33 | 0.78 – 1.53 | 2.38 |
| 1.6 | 8 | 1.25 | 2 | 2.00 – 4.88 | 7.63 |
| 0.8 | 4 | 1.25 | 22 | 1.00 – 1.00 | 1.56 |
| 0.8 | 12 | 1.15 | 41 | 0.92 – 2.13 | 3.72 |
| 0.8 | 8 | 1.60 | 6 | 1.28 – 5.24 | 21.47 |

What breaks at each end:

- **Bottom end (small sigma0):** starting at 0.5 gives slightly more keypoints, but they're at the pixel level. A blur of 0.5 px barely changes the image, so these "blobs" are mostly single-pixel details and noise. They aren't stable: add a little noise or resize a bit and they move or disappear. Also, the lowest level can never be a keypoint (it has no level below it to compare with), so the first rung is always "wasted".
- **Top end (big sigma0 or big k):** with sigma0 = 1.6 I got only **2** keypoints, and with k = 1.6 only **6**. Waldo's template is only 28 px wide. Once the blur is around 4–5 px, his stripes, glasses and hat all merge into one smooth blob, so there's nothing left to detect. Big blobs also run into the border and get thrown away.
- **Too few levels (n_scales = 4):** there are only 3 DoG levels, and only the middle one can have keypoints, so every keypoint gets sigma = 1.00. That's no scale information at all, the same problem as Harris in 3.6.
- **Too coarse k:** big jumps between levels mean blobs that fall "in between" two levels get missed, and the sigma values are rounded very coarsely. That makes the size ratio s in Task 4 less accurate.

So the ladder is a trade: go too fine or too low and you get unstable, noise-level points; go too high or too coarse and you blur away everything and lose scale precision. I used a finer ladder (sigma0 0.8, 12 levels, k 1.15) inside `find_waldo`. It gave the most keypoints (41) and finer sigma steps, which made the scale ratio s less rounded. (To be honest, I chose this after trying a few settings on these same scenes, so it's tuned to this data.)

## 3.8 More keypoints, fewer matches

The loop I wrote:

```python
scene = to_grayscale(load_image_as_rgb("imgs/scene_easy.png"))
template = to_grayscale(load_image_as_rgb("imgs/waldo_template.png"))
for up in [1.5, 2.0, 3.0, 4.0]:
    r = find_waldo(scene, template, upsample=up)
    print(up, len(r["kp_template"]), r["n_matches"], r["n_votes"])
```

| upsample | template keypoints | n_matches | (votes) |
| --- | --- | --- | --- |
| 1.5 | 63 | 522 | 18 |
| 2.0 | 79 | 316 | 15 |
| 3.0 | 79 | 406 | 22 |
| 4.0 | 34 | 196 | 0 |

**The trend:** enlarging the template does *not* give more matches. Overall the number of matches **falls** (522 down to 196), even though going from 1.5 to 2.0 gave *more* template keypoints (63 to 79) and *fewer* matches (522 to 316). It isn't perfectly smooth (3.0 bumps back up), but more keypoints clearly didn't mean more matches, and at 4.0 the voting finds nothing at all.

**Why: the stripes and the ratio test.** A scene keypoint is kept only if

```
d1 < ratio * d2      (ratio = 0.8)
```

where d1 is the distance to its closest template part and d2 is the distance to the second-closest. So the best part has to be clearly better than the runner-up.

The key visual property is that **Waldo's shirt (and hat) is a repeating pattern**: identical red/white stripes every 5 pixels (rows 29–41 on the shirt, 10–17 on the hat). When the template is enlarged, each stripe becomes big enough to be its own blob, so the template gets several keypoints that are near-copies of each other. At upsample 2.0, 33 of the 79 template keypoints are on stripes, and for 28 of those 33, the closest other template part in descriptor space is *another stripe*. So when a scene point lands on a stripe (on Waldo, or on any of the many striped people in the crowd), its closest and second-closest template parts are often two of these near-identical stripes. Then d1 is almost equal to d2, and d1 < 0.8·d2 fails. The match is thrown away even when it's correct, because the test can't tell *which* stripe it is. Every extra copy of the stripe adds another "almost as good" runner-up.

To be honest about what I measured: this isn't the only effect in my implementation. At 2.0, scene points whose best match was a stripe still passed about 22% of the time, while the others passed only about 5%. Most of the drop from 1.5 to 2.0 came from non-stripe points (pass rate fell from 14% to 5%), because the 16 new template parts sit close to existing ones in descriptor space and become runner-ups that shrink d2. The underlying reason is the same, though: a more crowded template "database" makes near-ties more likely, and the stripes are the most extreme case of that, since they're literal copies.

At **4.0** something else happens as well. The stripes are now 20 px apart, wider than the biggest blob my ladder looks for, so DoG stops firing on them (only 2 stripe keypoints are left, 34 in total). The remaining template keypoints have sigma 1.2–2.8 in the enlarged template, which is 0.3–0.7 px in the real scene's scale. That's below the scene's smallest sigma (0.92), so no scene keypoint is the right size to match them, and there are 0 votes.

## 3.9 Why does this task throw the colour away?

A gradient orientation means "the direction in which the image gets brighter fastest". That only makes sense if every pixel has **one number**. With three channels, each channel has its own gradient, and they can point in completely different directions.

Example: a vertical **red-to-green edge**, red on the left and green on the right.
- Red channel: goes from 1 to 0 left-to-right, so its gradient points **left** (180°).
- Green channel: goes from 0 to 1, so its gradient points **right** (0°).
- Blue channel: 0 on both sides, no gradient.

So two channels point in exactly opposite directions. If I add the gradient vectors they cancel to roughly zero, and the edge disappears, even though it's a very visible edge. If I average the angles, I get 90°, which points *along* the edge, which is nonsense. If I pick "the strongest channel", the answer can flip between 0° and 180° from a tiny change in colour. None of these gives one clear, stable direction, and the whole descriptor and the dominant orientation depend on that direction.

With luminance (0.299R + 0.587G + 0.114B), red becomes 0.299 and green becomes 0.587, so the edge gets brighter to the right. That's one clear direction (0°), and the gradient is well defined. That's why Tasks 3–4 convert to grayscale first: the gradient maths needs a scalar field, even though we lose the "red" information that Task 2 benefits from.
