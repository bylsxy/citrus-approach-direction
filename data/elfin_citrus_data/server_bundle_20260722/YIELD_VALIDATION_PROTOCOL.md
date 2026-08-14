# Yield Validation Protocol

This protocol defines how to evaluate the Stage 7 fruit reconstruction, counting, size estimation, and yield prediction pipeline once manual ground truth is available.

## Required Ground Truth

- `fruit_count_ground_truth` per evaluated scene or tree row.
- Per-fruit diameter or major/minor axis measurements if available.
- Harvested fruit weight per scene or tree row.
- Optional matched fruit IDs between visual observations and harvested fruits.

## Count Metrics

- Mean Absolute Error (MAE): `mean(abs(pred_count - gt_count))`
- Mean Absolute Percentage Error (MAPE): `mean(abs(pred_count - gt_count) / gt_count)`

## Size Metrics

- MAE for diameter or axes.
- RMSE for diameter or axes.
- R2 for diameter or axis regression.
- Optional Bland-Altman plot for estimated diameter versus manual caliper diameter.

## Yield Metrics

- RMSE of total mass.
- MAPE of total mass.
- R2 of scene-level or tree-level yield.

## Notes

- The current pipeline uses RGB-D scale and an assumed density. Real yield validation requires cultivar-specific density and calibration factor.
- Multi-frame de-duplication should be evaluated on sequences with known fruit IDs or carefully counted static scenes.
- If VINS poses are unavailable for a raw bag, fusion is camera-frame only and should not be interpreted as full global orchard yield.
