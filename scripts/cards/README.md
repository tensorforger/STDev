# Steppe cards

Generators of the Steppe release visuals: README cards, covers, the registry icon and comparison videos. Pure PIL
(+ ffmpeg for videos), one shared style in `style.py` (palette, fonts, supersampled layers, grids, the Steppe mark).

```bash
python scripts/cards/make_all.py            # everything -> ../tensorforger/assets/steppe
python scripts/cards/make_hero.py --output-directory dev/cards_preview/out   # one card
```

Inputs: ComfyUI results in `../comfyui_results` (`--input-directory`), benchmark results and predictions in
`raw_results/upscale_benchmark_axis_v1_1k`, ground truth in `benchmark_datasets/axis_v1_1k_wan_2x`, autoresearch logs.

| script | output | size |
|---|---|---|
| `make_hero.py` | `hero_cat` | 2400x1350 |
| `make_benchmark_plot.py` | `benchmark_plot` | 2400x1350 |
| `make_benchmark_table.py` | `benchmark_table` | 2400x1350 |
| `make_benchmark_gallery.py` | `benchmark_gallery` | 2400x1350 |
| `make_hff.py` | `hff_portrait` | 2400x1350 |
| `make_autoresearch.py` | `autoresearch` (+ `autoresearch_data.py` log parser) | 2400x1350 |
| `make_comparisons.py` | `compare_anime`, `compare_cloth`, `compare_leaves` | 2400x1350 |
| `make_how_it_works.py` | `how_it_works` | 2400x1350 |
| `make_models_lineup.py` | `models_lineup` | 2400x1350 |
| `make_workflow.py` | `workflow` (screenshot at native resolution) | 1920x(fit) |
| `make_covers.py` | `cover_banner_21x9`, `cover_16x9`, `cover_social_2x1`, `icon_400` | 2520x1080, 1920x1080, 1280x640, 400x400 |
| `make_video.py` | `video_compare.mp4`, `video_compare_zoom.mp4` | 2568x1024, 32 fps |

Every card is written as PNG (master) and JPG (web). Patch comparisons: the bicubic tile is upscaled by the same
factor as the Steppe output, then both get the same display resize.

Fonts: Inter and JetBrains Mono (SIL Open Font License, see `fonts/`). Benchmark images: Axis-V1 by refine-axis
(CC BY 4.0), credited on the cards that show them.
