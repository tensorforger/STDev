"""
Renders all Steppe cards, covers, the icon and the comparison videos into the assets directory.

Run: python scripts/cards/make_all.py [--input-directory ../comfyui_results] [--output-directory ../tensorforger/assets/steppe] [--skip-video]
"""

import make_autoresearch
import make_benchmark_gallery
import make_benchmark_plot
import make_benchmark_table
import make_comparisons
import make_covers
import make_hero
import make_hff
import make_how_it_works
import make_models_lineup
import make_video
import make_workflow
import style


def main() -> None:
    parser = style.standard_arguments(__doc__)
    parser.add_argument("--skip-video", action="store_true")
    arguments = parser.parse_args()
    input_directory, output_directory = arguments.input_directory, arguments.output_directory

    style.save_card(make_hero.build(input_directory), output_directory, "hero_cat")
    style.save_card(
        make_benchmark_plot.build(make_benchmark_plot.RESULTS_FILE),
        output_directory,
        "benchmark_plot",
    )
    style.save_card(
        make_benchmark_table.build(make_benchmark_plot.RESULTS_FILE),
        output_directory,
        "benchmark_table",
    )
    style.save_card(make_benchmark_gallery.build(), output_directory, "benchmark_gallery")
    style.save_card(make_hff.build(input_directory), output_directory, "hff_portrait")
    style.save_card(make_autoresearch.build(), output_directory, "autoresearch")
    style.save_card(
        make_comparisons.build_anime(input_directory), output_directory, "compare_anime"
    )
    style.save_card(
        make_comparisons.build_cloth(input_directory), output_directory, "compare_cloth"
    )
    style.save_card(
        make_comparisons.build_leaves(input_directory), output_directory, "compare_leaves"
    )
    style.save_card(make_how_it_works.build(), output_directory, "how_it_works")
    style.save_card(
        make_models_lineup.build(make_benchmark_plot.RESULTS_FILE),
        output_directory,
        "models_lineup",
    )
    style.save_card(make_workflow.build(input_directory), output_directory, "workflow")

    low = style.load_image(str(input_directory / make_covers.LOW_RESOLUTION_FILE))
    high = style.load_image(str(input_directory / make_covers.HIGH_RESOLUTION_FILE))
    for name, size in make_covers.COVERS:
        style.save_card(make_covers.build_cover(size, low, high), output_directory, name)
    make_covers.build_icon().save(output_directory / "icon_400.png", optimize=True)
    print("icon_400: 400x400")

    if not arguments.skip_video:
        low_path = input_directory / make_video.LOW_RESOLUTION_FILE
        high_path = input_directory / make_video.HIGH_RESOLUTION_FILE
        make_video.build_video(
            low_path, high_path, output_directory / "video_compare.mp4", zoom=False
        )
        make_video.build_video(
            low_path, high_path, output_directory / "video_compare_zoom.mp4", zoom=True
        )


if __name__ == "__main__":
    main()
