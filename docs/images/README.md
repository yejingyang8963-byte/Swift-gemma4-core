# Image assets

All images in this directory are **hand-written SVG** with no external
references and no embedded raster data. They are diff-friendly, render
natively on GitHub, and never break due to a CDN going down.

## Files

| File | Used in | Source of truth |
|---|---|---|
| `banner.svg` | Top of every README | This directory — edit and commit |
| `architecture.svg` | Standalone architecture diagram | This directory — edit and commit |

## Adding a real demo screencast

The README references a future `demo.gif` placeholder. When you're
ready to record one:

1. Open the example app or your own integration on a real iPhone.
2. Use QuickTime Player → File → New Movie Recording → select your
   iPhone as the camera source.
3. Record 5–8 seconds of:
   - Tap the mic button
   - Say a short prompt
   - First audio chunk plays back
4. Trim to ~6 seconds in QuickTime.
5. Convert to GIF with `ffmpeg`:
   ```bash
   ffmpeg -i demo.mov -vf "fps=15,scale=480:-1" -loop 0 demo.gif
   ```
6. Save as `docs/images/demo.gif` and reference it from the README.

Keep the GIF under 2 MB so GitHub doesn't truncate it on the README
preview.
