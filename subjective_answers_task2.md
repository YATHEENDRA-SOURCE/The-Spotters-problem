# Task 2: Subjective Answers

My Task 2 results (colour, all 8 scales, cross-scale NMS):

| scene | score | scale | IoU |
| --- | --- | --- | --- |
| scene_easy | 0.889 | 1.00 | 1.000 |
| scene_medium | 0.863 | 1.35 | 1.000 |
| scene_rotated | 0.829 | 1.15 | 0.000 |
| scene_hard | 0.809 | 1.50 | 1.000 |

It finds Waldo in three scenes and fails on the rotated one (it picks a different, upright person).

## 2.7 Why NCC and not sum-of-squared-differences?

Let the window be I (n pixels) with mean Ī, and the template T with mean T̄. Replace I by I' = aI + b with a > 0.

- New mean: Ī' = aĪ + b
- So each centred pixel becomes I' − Ī' = aI + b − aĪ − b = **a(I − Ī)**. The b cancels out completely.

Put that into NCC:

NCC(I') = Σ a(I − Ī)(T − T̄) / ( ‖a(I − Ī)‖ · ‖T − T̄‖ )
        = a · Σ (I − Ī)(T − T̄) / ( |a| · ‖I − Ī‖ · ‖T − T̄‖ )
        = (a / |a|) · NCC(I) = **NCC(I)** because a > 0.

So subtracting the mean removes the brightness offset b, and dividing by the norm removes the contrast gain a. (If a were negative, NCC would flip sign, which makes sense since it's a "negative" image.)

**SSD under the same change:**
SSD(I') = Σ (aI + b − T)². This changes with both a and b. For example, if the window is an exact copy (I = T) but lit differently, SSD = Σ((a−1)T + b)², which is not zero and can be large. A brighter copy of Waldo could score worse than some random dark patch. That's why SSD would break on the uneven lighting in scene_medium, and NCC doesn't care.

My code check: I pasted 0.5·T + 0.2 into a random image and NCC at that spot was 0.99999999, i.e. exactly 1.

## 2.8 Grayscale against colour, measured

Lines I changed (in the `__main__` loop, only for this experiment):

```python
template = to_grayscale(load_image_as_rgb("imgs/waldo_template.png"))
...
scene = to_grayscale(load_image_as_rgb("imgs/{}.png".format(name)))
```

(My `ncc_map` accepts a 2-D image too; it just treats it as one channel.)

| scene | colour score | scale | IoU | gray score | scale | IoU |
| --- | --- | --- | --- | --- | --- | --- |
| scene_easy | 0.889 | 1.00 | 1.000 | 0.862 | 1.00 | 1.000 |
| scene_medium | 0.863 | 1.35 | 1.000 | 0.844 | 1.35 | 1.000 |
| scene_rotated | 0.829 | 1.15 | 0.000 | 0.797 | 1.15 | 0.000 |
| scene_hard | 0.809 | 1.50 | 1.000 | 0.786 | 1.50 | 1.000 |

Honestly, this surprised me a bit. **The same three scenes still work in grayscale**, and the winning score only drops by about 0.02–0.03 in each scene.

Why the change is this small, based on Waldo's actual colours: his template is mostly **red and white stripes** (the average template colour is R 0.62, G 0.47, B 0.48). In luminance, red is fairly dark (about 0.3–0.4) and white is very bright (about 0.95). So the red/white stripes are already a very strong *brightness* pattern, and grayscale keeps almost all of it. Most of what makes Waldo "Waldo" for the correlation is that light/dark stripe rhythm, not the hue.

Colour only adds information where two things have the same brightness but different hue, and it did add some: the correct match scored higher in colour every time. That extra score is the part of the match that comes from the red being specifically red. It would matter much more against a decoy with, say, dark-blue and white stripes, whose grayscale pattern looks like Waldo's but whose colour doesn't. In these four scenes the true Waldo already wins in grayscale, so the extra margin didn't change any result. Colour makes the win safer but isn't what decides it here.

## 2.9 The mask you were given and never used

The mask covers 60.9% of the template rectangle; the other 39% is grey background around Waldo.

My masked version: I centre each channel using the mean over masked pixels only, set the template to zero outside the mask, and compute the window sums (Σ I and Σ I²) with the mask as weights. Integral images can't do this because the mask isn't a box, so I used einsum with the mask.

```python
def masked_ncc_map(image, template, mask):
    th, tw = template.shape[:2]
    M = (mask > 0.5).astype(np.float64); n = M.sum()
    oh, ow = image.shape[0] - th + 1, image.shape[1] - tw + 1
    num = np.zeros((oh, ow)); we = np.zeros((oh, ow)); te = 0.0
    for c in range(image.shape[2]):
        ch, t = image[..., c], template[..., c]
        tc = (t - (t * M).sum() / n) * M            # centred by masked mean, 0 off-mask
        te += (tc ** 2).sum()
        win = sliding_windows(ch, th, tw)
        num += np.einsum("ijkl,kl->ij", win, tc)
        s1 = np.einsum("ijkl,kl->ij", win, M)
        s2 = np.einsum("ijkl,kl->ij", sliding_windows(ch ** 2, th, tw), M)
        we += np.maximum(s2 - s1 ** 2 / n, 0)
    d = np.sqrt(we * te); out = np.zeros((oh, ow)); v = d > 1e-12
    out[v] = num[v] / d[v]
    return out
```

Results on scene_hard (best position at each scale; the mask is resized along with the template):

| scale | unmasked best | IoU | masked best | IoU |
| --- | --- | --- | --- | --- |
| 0.60 | 0.628 | 0.00 | 0.714 | 0.00 |
| 0.72 | 0.676 | 0.00 | 0.762 | 0.00 |
| 0.85 | 0.624 | 0.00 | 0.800 | 0.00 |
| 1.00 | 0.704 | 0.00 | 0.814 | 0.00 |
| 1.15 | 0.699 | 0.00 | 0.846 | 0.00 |
| 1.35 | 0.450 | 0.82 | 0.519 | 0.82 |
| **1.50** | **0.809** | **1.00** | **0.932** | **1.00** |
| 1.70 | 0.345 | 0.77 | 0.444 | 0.00 |

**Without mask:** best score 0.809 at scale 1.50 (correct).
**With mask:** best score 0.932 at scale 1.50 (still correct).

So the score on Waldo went **up** a lot, from 0.81 to 0.93. But scores went up everywhere, including on wrong people (at scale 1.15 a wrong person went from 0.699 to 0.846). The gap between Waldo and the runner-up actually got a bit **smaller** (0.105 without the mask, 0.086 with it). So the mask made Waldo look like a better match, but didn't make him easier to pick out.

Why the direction: the unmasked NCC is forced to include every pixel in the rectangle, including the 39% that is just grey background in the template. In the scene, those same pixels are whatever happens to be behind Waldo (coloured shapes, other people, the occluder), which has nothing to do with the template's flat grey. Those pixels add mismatch and pull the correlation down. The mask throws them out, so only Waldo's body is compared, and the score rises.

Two notes, though:
1. The occluder in scene_hard sits partly *on Waldo himself* (about 7% of him), and the mask still includes those pixels, so the mask can't fully remove the occlusion. It only removes the background part.
2. The background was also being unfair to the wrong people in the same way. Removing it helps them too, which is why the margin didn't improve.

## 2.10 Scores are not comparable across scales

For scene_hard (unmasked, colour):

| scale | template pixels | max | std | max / std |
| --- | --- | --- | --- | --- |
| 0.60 | 578 | 0.628 | 0.0995 | 6.3 |
| 0.72 | 800 | 0.676 | 0.0999 | 6.8 |
| 0.85 | 1152 | 0.624 | 0.0998 | 6.3 |
| 1.00 | 1568 | 0.704 | 0.0928 | 7.6 |
| 1.15 | 2048 | 0.699 | 0.0968 | 7.2 |
| 1.35 | 2888 | 0.450 | 0.0921 | 4.9 |
| 1.50 | 3528 | 0.809 | 0.0892 | 9.1 |
| 1.70 | 4560 | 0.345 | 0.0846 | 4.1 |

**Would dividing each map by its own std make the scales comparable? No, not really.** Looking at my own numbers:

- The std hardly changes. It goes from 0.085 to 0.100, only about a 15% spread, even though the template size changes by 8x (578 to 4560 pixels). Meanwhile the maxima range from 0.35 to 0.81. Dividing by std means dividing every map by almost the same number, so it basically just rescales everything and changes almost nothing. The ranking by max/std (1.50 > 1.00 > 1.15 > 0.72 > ...) is the same as the ranking by raw max.
- The problem with small templates is about the **highest** value in the map, not the typical spread. The maximum depends on how many roughly independent positions there are and how heavy the tail of the score distribution is. A small template gives many more "independent tries", so its luckiest position is higher by chance. The std describes the middle of the distribution, not how far its extreme tail reaches, so it can't correct for this.
- In this scene the maxima aren't really set by chance anyway. They're set by other real people who happen to fit at that size (at scale 1.0, a different person scores 0.704). Standardising can't tell a real lookalike from a chance peak.

So the std is roughly the same at every scale, and dividing by it can't fix a difference it doesn't capture. The fix would need to model the maximum itself (for example, compare against the max of the same map on scenes without Waldo), not just the spread.
