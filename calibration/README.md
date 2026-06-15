# Calibration

`construct_historical_cieluv_target.py` builds the historical grey-relative CIE-LUV target and writes all outputs under:

`calibration/outputs/01_establish_historical_cieluv_target/`

It accepts either:

- raw spectra tables with columns `rep,id,r,g,b,nm,power`
- measured XYZ tables with columns `rep,id,r,g,b,X,Y,Z`

If the input is spectral, the script converts it internally to true XYZ 1931 by default using `color_matching_functions/ciexyz31.txt`.

The script supports grouped handles so multiple lightness levels can be treated as one hue cluster for ellipse fitting and target definition.

Final target points are solved so neighboring colors have equal Euclidean distance in the 2D fit plane. The ellipse arc length between neighbors can vary and is saved as metadata.

Example:

```bash
python calibration/construct_historical_cieluv_target.py \
  --input-files \
    tmp/Apollo_spec_webcolor_measured_XYZ.csv \
    tmp/Curiosity_spec_webcolor_measured_XYZ.csv \
    tmp/Sputnik_spec_webcolor_measured_XYZ.csv \
    tmp/Voyager_spec_webcolor_measured_XYZ.csv \
  --handle-groups '3,9,15;4,10,16;5,11,17;6,12,18;7,13,19;8,14,20' \
  --handle-labels 'RED,YELLOW,GREEN,CYAN,BLUE,MAGENTA'
```

The same command also works with the corresponding raw spectral files instead of the precomputed XYZ files.

Key outputs:

- `data/average_grey_xyz.csv`: per-tablet mean grey XYZ plus one overall row averaging those tablet means equally
- `data/vector_samples.csv`: per-sample white-referenced LUV values and grey-relative fit vectors
- `data/fit_clusters.csv`: grouped hue clusters used for the ellipse fit
- `data/ellipse_fit.json`: saved ellipse parameters and spacing summary
- `data/historical_cieluv_target.csv`: final target vectors
- `data/compact_LUV_displacenments.csv`: compact grey-relative target displacements with columns `id,description,delta_L,delta_U,delta_V`
- `plots/*.png`: fit, handle alignment, and final target plots

`fit_new_screen_xyz_to_rgb_model.py` fits the new-screen XYZ -> RGB model and writes all outputs under:

`calibration/outputs/02_fit_screen_model/`

It accepts either:

- raw spectra tables with columns `rep,id,r,g,b,nm,power`
- precomputed XYZ tables with columns `rep,id,r,g,b,X,Y,Z`

By default it uses:

`input_files/color_battery_amoled_B.csv`

If the input is spectral, the script converts it to true XYZ 1931 by default using `color_matching_functions/ciexyz31.txt`. The script now fits two standalone sigmoid-hidden MLPs:

- `xyz_to_rgb_mlp.py`: measured XYZ -> display RGB
- `rgb_to_xyz_mlp.py`: display RGB -> measured XYZ

The train/test split is ID-disjoint: a color ID is either fully in training or fully held out in test, and all reps for that ID stay together. Within either split, each rep is still used as a separate data point.

For the `XYZ -> RGB` model, XYZ inputs are z-scored, RGB targets are trained in `[-1, 1]`, and predictions are mapped back to `0..255` with warnings recorded when raw predictions land outside range before clipping.

For the `RGB -> XYZ` model, RGB inputs are z-scored and XYZ is learned directly in measured XYZ units. Negative predicted XYZ values are flagged in the saved diagnostics and clipped to zero in the saved predictions / reported metrics.

Example:

```bash
python calibration/fit_new_screen_xyz_to_rgb_model.py \
  --input-files input_files/color_battery_amoled_B.csv \
  --hidden-dim 16 \
  --max-steps 10000
```

Key outputs:

- `data/combined_measured_xyz.csv`: converted / combined XYZ measurements
- `data/combined_measured_xyz_clean.csv`: cleaned XYZ rows used for fitting
- `data/id_split_counts.csv`: row counts per ID in train and test, with held-out IDs fully separated
- `data/training_history.csv`: XYZ -> RGB train/test loss history
- `data/rgb_to_xyz_training_history.csv`: RGB -> XYZ train/test loss history
- `data/train_predictions.csv`, `data/test_predictions.csv`, `data/all_predictions.csv`: XYZ -> RGB per-row predictions and errors
- `data/train_id_summary.csv`, `data/test_id_summary.csv`, `data/all_id_summary.csv`: XYZ -> RGB per-ID swatch summaries
- `data/rgb_to_xyz_train_predictions.csv`, `data/rgb_to_xyz_test_predictions.csv`, `data/rgb_to_xyz_all_predictions.csv`: RGB -> XYZ per-row predictions and errors
- `data/rgb_to_xyz_train_id_summary.csv`, `data/rgb_to_xyz_test_id_summary.csv`, `data/rgb_to_xyz_all_id_summary.csv`: RGB -> XYZ per-ID summaries
- `data/screen_xyz_to_rgb_mlp_model.pth`: saved XYZ -> RGB model
- `data/screen_rgb_to_xyz_mlp_model.pth`: saved RGB -> XYZ model
- `data/fit_summary.json`: shared split metadata plus metrics for both models
- `plots/01_loss_curves.png`: XYZ -> RGB train/test loss and MAE curves
- `plots/02_train_swatches.png`, `plots/03_test_swatches.png`: XYZ -> RGB target vs predicted swatch grids
- `plots/04_rgb_to_xyz_loss_curves.png`: RGB -> XYZ train/test loss and MAE curves
- `plots/05_rgb_to_xyz_train_scatter.png`, `plots/06_rgb_to_xyz_test_scatter.png`: RGB -> XYZ measured vs predicted XYZ scatter plots

`map_cieluv_targets_to_new_screen.py` maps the historical grey-relative CIE-LUV targets onto the new screen using the stage-01 displacement file and the stage-02 forward / inverse MLPs. It writes all outputs under:

`calibration/outputs/03_map_cieluv_targets_to_new_screen/`

By default it uses:

- `outputs/02_fit_screen_model/data/screen_xyz_to_rgb_mlp_model.pth`
- `outputs/02_fit_screen_model/data/screen_rgb_to_xyz_mlp_model.pth`
- `outputs/02_fit_screen_model/data/combined_measured_xyz_clean.csv`
- `outputs/01_establish_historical_cieluv_target/data/compact_LUV_displacenments.csv`

The white reference is the average measured XYZ for `--white-id` (default `1`). The real background grey is then chosen in the new screen's white-referenced CIE-LUV space at either:

- `--grey-l` if provided
- `0.70 * white_L` by default via `--grey-l-fraction 0.70`

Alternatively, `--set-grey-xyz X Y Z` overrides the optimized background grey entirely. In that mode, the provided XYZ is mapped into the new screen's white-referenced CIE-LUV space to define the real grey's `L*`, `u*`, and `v*`, and all generated output filenames receive the suffix `_set_grey`.

The script samples the new screen gamut with the saved RGB -> XYZ model, searches the fixed-L u/v slice for the grey point with the largest inscribed circle, then applies the saved `delta_L`, `delta_U`, and `delta_V` target displacements at the real grey and at virtual grey points above and below it in luminance. By default:

- `--num-additional-lum-levels 10`
- `--lum-step-size 2.0`

This yields `11` luminance levels total, ordered from lowest `L*` to highest `L*`.

Example:

```bash
python calibration/map_cieluv_targets_to_new_screen.py \
  --measured-xyz-path calibration/outputs/02_fit_screen_model/data/combined_measured_xyz_clean.csv \
  --target-displacements-path calibration/outputs/01_establish_historical_cieluv_target/data/compact_LUV_displacenments.csv
```

Example with a fixed background grey:

```bash
python calibration/map_cieluv_targets_to_new_screen.py \
  --set-grey-xyz 0.1539 0.1537 0.1704
```

Key outputs:

- `data/optimal_grey_point.json`: chosen real grey point, inscribed-circle radius, and background RGB
- `data/virtual_grey_levels.csv`: the real grey plus the virtual grey points at each luminance level
- `data/new_screen_target_colors_detailed.csv`: full per-color per-level XYZ, LUV, RGB, and gamut diagnostics
- `data/new_screen_colors_rgb.tsv`: final tab-delimited RGB table with background grey first, then luminance blocks from lowest `L*` to highest `L*`
- `data/new_screen_colors_xyz.tsv`: same structure in XYZ coordinates
- `data/out_of_gamut_summary.csv`: per-level out-of-gamut counts
- `data/run_summary.json`: overall counts and output paths
- `plots/01_sampled_gamut_luv_cloud.png`: sampled full-gamut cloud in CIE-LUV
- `plots/02_grey_slice_and_inscribed_circle.png`: the chosen fixed-L grey slice with the grey point and its largest inscribed circle
- `plots/03_gamut_slices_every_20L.png`: sampled gamut slices at `L*=0,20,40,60,80,100`
- `plots/04_out_of_gamut_summary.png`: final target colors by luminance level, highlighting out-of-gamut points
- `plots/05_color_wheels_by_luminance/`: one plot per luminance level showing the final RGB colors as a circular wheel on the real background grey

When `--set-grey-xyz` is used, these files are written with `_set_grey` added before the extension, for example `new_screen_colors_rgb_set_grey.tsv` and `02_grey_slice_and_inscribed_circle_set_grey.png`.
