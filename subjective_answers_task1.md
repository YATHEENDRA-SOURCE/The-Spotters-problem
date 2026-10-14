# Task 1: Subjective Answers

## 1.2 Why normalise by the sum and not by 1/(2πσ²)?

The analytic constant 1/(2πσ²) only makes the Gaussian integrate to 1 over the *whole infinite plane*. My kernel is a small finite grid, so it cuts off the tails. I also sample it at integer points instead of integrating. Because of that, the samples multiplied by 1/(2πσ²) don't add up to 1. I checked this:

| size | sigma | sum using the analytic constant |
| --- | --- | --- |
| 5 | 1.0 | 0.9818 |
| 5 | 2.0 | 0.6297 |
| 9 | 2.0 | 0.9546 |
| 13 | 2.0 | 0.9980 |

A 5x5 kernel with sigma 2 only sums to 0.63, so the blurred image would come out about 37% darker. Dividing by the actual sum of the samples fixes this, and a flat region stays the same brightness after blurring.

## 1.7 Predict first, then measure

**My prediction (written before running):**
- PSNR: I expected the 9x9 box filter to make PSNR **better**, because averaging 81 pixels should cancel most of the random noise.
- SSIM: I expected SSIM to get **better** too, for the same reason.

**What I measured (first row of the table):**

| filter | PSNR in | PSNR out | SSIM in | SSIM out |
| --- | --- | --- | --- | --- |
| box 9x9 on gaussian | 24.53 | 21.55 | 0.42 | 0.72 |

- PSNR prediction: **wrong**. It went *down*, from 24.53 to 21.55 dB.
- SSIM prediction: **right**. It went up, from 0.42 to 0.72.

Why I got PSNR wrong: I only thought about the noise and forgot the blur also damages the image itself. The scene is full of small people with sharp stripes and edges only a few pixels wide. A 9x9 box smears every one of those edges, and at each edge the error is large. PSNR is based on the average squared error, and those big edge errors are larger than the noise that was removed. So the output is further from the original than the noisy input was.

## 1.8 The two metrics disagree

PSNR only looks at the pixel-by-pixel error, squared and averaged. SSIM compares local structure: local mean, contrast, and whether the two images go up and down together (the covariance term).

After the box filter:
- The noise, which is random and uncorrelated, is mostly gone. So in each small window, the image and the original now vary *together* much better. The covariance term goes up a lot, which pushes SSIM up.
- But every sharp edge is now a soft ramp. Near edges each pixel value is quite far off, and because PSNR squares the errors, these few large ones dominate. So PSNR goes down.

So the image is "further" from the original in raw pixel error, but "more similar" in structure. The noise made everything jittery, and the blur removed the jitter while softening the edges.

**Which one I would trust before template matching:** SSIM. NCC in Task 2 is itself a correlation. It asks whether the window rises and falls in the same pattern as the template, and it ignores brightness and contrast shifts. That's basically what SSIM's structure term measures. PSNR punishes things NCC doesn't care about, like a small overall brightness change.

## 1.9 What colour costs you

Line I changed in the `__main__` block (only for this experiment; I changed it back afterwards):

```python
# clean = load_image_as_rgb("imgs/scene_easy.png")
clean = to_grayscale(load_image_as_rgb("imgs/scene_easy.png"))
```

(The last plot does `ssim_map.mean(axis=2)`, which needs 3 channels, so I only ran the table part for grayscale.)

**Colour (3 channels):**

| filter | PSNR in | PSNR out | SSIM in | SSIM out |
| --- | --- | --- | --- | --- |
| box 9x9 on gaussian | 24.53 | 21.55 | 0.4241 | 0.7225 |
| gaussian s=2 on gaussian | 24.53 | 23.45 | 0.4241 | 0.8078 |
| median 5x5 on salt+pepper | 18.82 | 24.79 | 0.4084 | 0.9129 |
| gaussian s=1.5 on salt+pepper | 18.82 | 24.12 | 0.4084 | 0.7580 |

**Grayscale (1 channel):**

| filter | PSNR in | PSNR out | SSIM in | SSIM out |
| --- | --- | --- | --- | --- |
| box 9x9 on gaussian | 24.48 | 22.04 | 0.4040 | 0.7411 |
| gaussian s=2 on gaussian | 24.48 | 23.93 | 0.4040 | 0.8190 |
| median 5x5 on salt+pepper | 18.82 | 25.12 | 0.3971 | 0.9129 |
| gaussian s=1.5 on salt+pepper | 18.82 | 24.52 | 0.3971 | 0.7607 |

The numbers barely move (at most about 0.5 dB and 0.02 SSIM). That's expected, because all these filters are applied to each channel separately and the same way. Nothing mixes between R, G and B. The three colour channels of this scene have very similar edges, so filtering three channels is almost like filtering one channel three times. Also, both metrics are just averages of per-channel errors, so they don't reward the fact that the channels are *different* from each other. The noise level is the same per channel, so the "PSNR in" is almost the same too.

This doesn't mean colour is useless. These metrics just don't measure what colour gives you. What matters for finding Waldo is that his red is a different *direction* in RGB from, say, a grey or a blue stripe that has the same brightness. A filter doesn't care about that, but the matcher does. Task 2 is where colour pays off (see 2.8).

## 1.10 Padding

I made a copy of `convolve2d` with `mode="constant"` (zero padding) and blurred `scene_easy` with a 31x31 Gaussian (sigma 5). The plot is in `plots/task1_padding_zero_vs_reflect.png`.

What I saw: a **dark frame all around the border**. The average brightness of the outer 5-pixel strip dropped from 0.786 (reflect) to 0.537 (zero), and the corner pixel went from 0.795 to 0.232. The darkening fades in over roughly the kernel radius (about 11–15 pixels).

Why: with zero padding, the part of the kernel that hangs off the image is averaging in pure black pixels that aren't really there. At a corner, about three quarters of the kernel is over black, so the pixel ends up much too dark. Reflect padding fills the outside with a mirror copy of the nearby image, so the averages stay realistic.

Why reflect is safer before template matching: the dark frame is a fake gradient/edge that isn't in the real scene. A template sitting near the border would be correlated against this fake dark band, and its score would drop. Worse, the strong artificial edge could create false matches or false corners/keypoints along the border. Reflect doesn't invent any new structure, so it's the safer default.

## Other things the code asks about

**Median is not a convolution.** Convolution is linear: it's always a fixed weighted sum of the neighbours. The median isn't linear, since median(a + b) is not median(a) + median(b), and which neighbour becomes the output depends on the values themselves (it's a sorting operation). So no fixed kernel can compute it. That's also why it's so good against salt-and-pepper: an extreme 0 or 1 just gets sorted to the end of the list and ignored, whereas a Gaussian still averages it in (median 24.79 dB / SSIM 0.91 vs Gaussian 24.12 dB / SSIM 0.76 on the same noise).

**Per-channel median on colour.** Taking the median of R, G and B separately can produce a colour that doesn't exist anywhere in the window, because the red median might come from one neighbour and the blue median from another. A "vector median" picks the one actual input pixel that is closest to all the others, so the output colour is always a real one from the window. It costs more to compute.

**Why separable blur is faster.** A 2-D Gaussian is exp(-x²/2σ²) times exp(-y²/2σ²), so blurring rows and then columns gives exactly the same result (my test matched the full 2-D convolution to about 1e-16). For a kernel of width k, that's 2k operations per pixel instead of k². For sigma = 2 (k = 13), that's 26 instead of 169, about 6.5 times less work.
