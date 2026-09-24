# Evaluation Summary

- Input directory: `/tmp/sdxl-tada-metrics-84439/tmp/sdxl_tada_matched_metric_views/forward_2pct`
- Samples: 1

- Noise comparison artifacts: `noise_comparisons/`

- Normality artifacts: `normality/`

- Image reconstruction artifacts: `image_comparisons/`

- Masked noise-region artifacts: `masked_noise_regions/`

## Aggregate

### initial_latent_stats

- `mean`: 0.004156995099037886
- `std`: 1.0025171041488647
- `min`: -4.2890625
- `p01`: -2.3125
- `p05`: -1.6416015625
- `p50`: 0.0008029937744140625
- `p95`: 1.6572265625
- `p99`: 2.3333053588867188
- `max`: 4.078125
- `skew`: 0.0061390595510602
- `excess_kurtosis`: -0.022577762603759766
- `abs_mean_error_from_normal`: 0.004156995099037886
- `abs_std_error_from_normal`: 0.002509474754333496
- `normal_kl_from_standard`: 1.4934921637177467e-05

### final_latent_stats

- `mean`: 0.12238584458827972
- `std`: 0.536048948764801
- `min`: -2.10546875
- `p01`: -1.1298828125
- `p05`: -0.7470703125
- `p50`: 0.1407470703125
- `p95`: 0.98828125
- `p99`: 1.443359375
- `max`: 2.517578125
- `skew`: 0.02857820689678192
- `excess_kurtosis`: 0.24413394927978516
- `abs_mean_error_from_normal`: 0.12238584458827972
- `abs_std_error_from_normal`: 0.4639551639556885
- `normal_kl_from_standard`: 0.2746986746788025

### initial_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### final_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### inverted_noise_stats

- `mean`: 0.019804945215582848
- `std`: 0.9196821451187134
- `min`: -4.48828125
- `p01`: -2.076171875
- `p05`: -1.481689453125
- `p50`: 0.012340545654296875
- `p95`: 1.5419921875
- `p99`: 2.146484375
- `max`: 3.927734375
- `skew`: 0.026125028729438782
- `excess_kurtosis`: -0.08054876327514648
- `abs_mean_error_from_normal`: 0.019804945215582848
- `abs_std_error_from_normal`: 0.08032488822937012
- `normal_kl_from_standard`: 0.006832093000411987

### initial_vs_inverted_noise

- `mse`: 0.16410857439041138
- `rmse`: 0.4051031768321991
- `mae`: 0.3139306902885437
- `l2`: 103.70671844482422
- `cosine`: 0.9147230386734009
- `relative_rmse_to_reference_std`: 0.4040891230106354
- `candidate_to_reference_std_ratio`: 0.91737300157547
- `mean_shift`: 0.015647949650883675

### inverted_noise_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### latent_normality

- `corr`: 0.0
- `corr_std`: 0.0
- `kl`: 0.008028961718082428
- `reverse_kl`: 0.007149539887905121
- `symmetric_kl`: 0.007589250802993774
- `per_location_kl`: 82054299648.0
- `per_location_reverse_kl`: 82054299648.0
- `per_location_symmetric_kl`: 82054299648.0

### initial_vs_inverted_noise_per_channel

- `channel_0/mse`: 0.23231710493564606
- `channel_0/rmse`: 0.48199284076690674
- `channel_0/mae`: 0.37749406695365906
- `channel_0/l2`: 61.69511413574219
- `channel_0/cosine`: 0.8784896731376648
- `channel_0/relative_rmse_to_reference_std`: 0.47868356108665466
- `channel_0/candidate_to_reference_std_ratio`: 0.8487295508384705
- `channel_0/mean_shift`: 0.01258104108273983
- `channel_1/mse`: 0.137186661362648
- `channel_1/rmse`: 0.37038716673851013
- `channel_1/mae`: 0.29039266705513
- `channel_1/l2`: 47.40958023071289
- `channel_1/cosine`: 0.9301490783691406
- `channel_1/relative_rmse_to_reference_std`: 0.3674631416797638
- `channel_1/candidate_to_reference_std_ratio`: 0.9445039629936218
- `channel_1/mean_shift`: 0.0069590723142027855
- `channel_2/mse`: 0.1168024018406868
- `channel_2/rmse`: 0.3417636752128601
- `channel_2/mae`: 0.2657507359981537
- `channel_2/l2`: 43.74577331542969
- `channel_2/cosine`: 0.9402251839637756
- `channel_2/relative_rmse_to_reference_std`: 0.34205344319343567
- `channel_2/candidate_to_reference_std_ratio`: 0.9717258214950562
- `channel_2/mean_shift`: 0.01812438666820526
- `channel_3/mse`: 0.17012819647789001
- `channel_3/rmse`: 0.41246598958969116
- `channel_3/mae`: 0.3220852315425873
- `channel_3/l2`: 52.79567337036133
- `channel_3/cosine`: 0.9102588891983032
- `channel_3/relative_rmse_to_reference_std`: 0.41415515542030334
- `channel_3/candidate_to_reference_std_ratio`: 0.9001694917678833
- `channel_3/mean_shift`: 0.024927305057644844

### inversion_error_stats

- `signed_mean`: 0.015647951513528824
- `signed_std`: 0.4048008620738983
- `abs_mean`: 0.3139306902885437
- `abs_std`: 0.2560392916202545
- `abs_p50`: 0.25341796875
- `abs_p90`: 0.6623687744140625
- `abs_p95`: 0.81298828125
- `abs_p99`: 1.1371169090270996
- `abs_max`: 2.24658203125

### inversion_error_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### inversion_error_structure

- `signed_error_vs_final_latent_cosine`: 0.007321503013372421
- `abs_error_vs_abs_final_latent_cosine`: 0.6108078956604004

### final_vs_reconstructed_image

- `mse/mean`: 0.004386543296277523
- `mse/std`: 0.0
- `mse/min`: 0.004386543296277523
- `mse/max`: 0.004386543296277523
- `rmse/mean`: 0.06623098254203796
- `rmse/std`: 0.0
- `rmse/min`: 0.06623098254203796
- `rmse/max`: 0.06623098254203796
- `mae/mean`: 0.04376319423317909
- `mae/std`: 0.0
- `mae/min`: 0.04376319423317909
- `mae/max`: 0.04376319423317909
- `psnr_db/mean`: 23.57877540588379
- `psnr_db/std`: 0.0
- `psnr_db/min`: 23.57877540588379
- `psnr_db/max`: 23.57877540588379
- `ssim/mean`: 0.7101447582244873
- `ssim/std`: 0.0
- `ssim/min`: 0.7101447582244873
- `ssim/max`: 0.7101447582244873
- `max_abs_error/mean`: 0.5882353186607361
- `max_abs_error/std`: 0.0
- `max_abs_error/min`: 0.5882353186607361
- `max_abs_error/max`: 0.5882353186607361
- `mean_signed_error/mean`: -0.006687051150947809
- `mean_signed_error/std`: 0.0
- `mean_signed_error/min`: -0.006687051150947809
- `mean_signed_error/max`: -0.006687051150947809
- `plain_threshold/mean`: 0.02500000037252903
- `plain_threshold/std`: 0.0
- `plain_threshold/min`: 0.02500000037252903
- `plain_threshold/max`: 0.02500000037252903
- `plain_pixel_count/mean`: 808415.0
- `plain_pixel_count/std`: 0.0
- `plain_pixel_count/min`: 808415.0
- `plain_pixel_count/max`: 808415.0
- `plain_pixel_fraction/mean`: 0.7709646224975586
- `plain_pixel_fraction/std`: 0.0
- `plain_pixel_fraction/min`: 0.7709646224975586
- `plain_pixel_fraction/max`: 0.7709646224975586
- `plain_mse/mean`: 0.0036451714113354683
- `plain_mse/std`: 0.0
- `plain_mse/min`: 0.0036451714113354683
- `plain_mse/max`: 0.0036451714113354683
- `plain_rmse/mean`: 0.06037525460124016
- `plain_rmse/std`: 0.0
- `plain_rmse/min`: 0.06037525460124016
- `plain_rmse/max`: 0.06037525460124016
- `plain_mae/mean`: 0.03959970921278
- `plain_mae/std`: 0.0
- `plain_mae/min`: 0.03959970921278
- `plain_mae/max`: 0.03959970921278
- `plain_max_abs_error/mean`: 0.5372549295425415
- `plain_max_abs_error/std`: 0.0
- `plain_max_abs_error/min`: 0.5372549295425415
- `plain_max_abs_error/max`: 0.5372549295425415
- `plain_mean_signed_error/mean`: -0.010371720418334007
- `plain_mean_signed_error/std`: 0.0
- `plain_mean_signed_error/min`: -0.010371720418334007
- `plain_mean_signed_error/max`: -0.010371720418334007
- `non_plain_pixel_count/mean`: 240161.0
- `non_plain_pixel_count/std`: 0.0
- `non_plain_pixel_count/min`: 240161.0
- `non_plain_pixel_count/max`: 240161.0
- `non_plain_pixel_fraction/mean`: 0.2290353775024414
- `non_plain_pixel_fraction/std`: 0.0
- `non_plain_pixel_fraction/min`: 0.2290353775024414
- `non_plain_pixel_fraction/max`: 0.2290353775024414
- `non_plain_mse/mean`: 0.006882102694362402
- `non_plain_mse/std`: 0.0
- `non_plain_mse/min`: 0.006882102694362402
- `non_plain_mse/max`: 0.006882102694362402
- `non_plain_rmse/mean`: 0.08295843750238419
- `non_plain_rmse/std`: 0.0
- `non_plain_rmse/min`: 0.08295843750238419
- `non_plain_rmse/max`: 0.08295843750238419
- `non_plain_mae/mean`: 0.05777806416153908
- `non_plain_mae/std`: 0.0
- `non_plain_mae/min`: 0.05777806416153908
- `non_plain_mae/max`: 0.05777806416153908
- `non_plain_max_abs_error/mean`: 0.5882353186607361
- `non_plain_max_abs_error/std`: 0.0
- `non_plain_max_abs_error/min`: 0.5882353186607361
- `non_plain_max_abs_error/max`: 0.5882353186607361
- `non_plain_mean_signed_error/mean`: 0.0057160574942827225
- `non_plain_mean_signed_error/std`: 0.0
- `non_plain_mean_signed_error/min`: 0.0057160574942827225
- `non_plain_mean_signed_error/max`: 0.0057160574942827225
- `lpips/mean`: 0.19775845110416412
- `lpips/std`: 0.0
- `lpips/min`: 0.19775845110416412
- `lpips/max`: 0.19775845110416412

### initial_vs_inverted_noise_normality

- `sample_size/mean`: 5000.0
- `sample_size/std`: 0.0
- `sample_size/min`: 5000.0
- `sample_size/max`: 5000.0
- `num_quantiles/mean`: 201.0
- `num_quantiles/std`: 0.0
- `num_quantiles/min`: 201.0
- `num_quantiles/max`: 201.0
- `initial_shapiro_p_value/mean`: 0.7464752197265625
- `initial_shapiro_p_value/std`: 0.0
- `initial_shapiro_p_value/min`: 0.7464752197265625
- `initial_shapiro_p_value/max`: 0.7464752197265625
- `inverted_shapiro_p_value/mean`: 0.20809976756572723
- `inverted_shapiro_p_value/std`: 0.0
- `inverted_shapiro_p_value/min`: 0.20809976756572723
- `inverted_shapiro_p_value/max`: 0.20809976756572723
- `gaussian_kl_initial_to_inverted/mean`: 0.009272292256355286
- `gaussian_kl_initial_to_inverted/std`: 0.0
- `gaussian_kl_initial_to_inverted/min`: 0.009272292256355286
- `gaussian_kl_initial_to_inverted/max`: 0.009272292256355286
- `gaussian_kl_inverted_to_initial/mean`: 0.008187755942344666
- `gaussian_kl_inverted_to_initial/std`: 0.0
- `gaussian_kl_inverted_to_initial/min`: 0.008187755942344666
- `gaussian_kl_inverted_to_initial/max`: 0.008187755942344666
- `gaussian_kl_symmetric/mean`: 0.008730024099349976
- `gaussian_kl_symmetric/std`: 0.0
- `gaussian_kl_symmetric/min`: 0.008730024099349976
- `gaussian_kl_symmetric/max`: 0.008730024099349976

### initial_vs_inverted_noise_by_plain_region

- `plain_abs_error/mean`: 0.3282093405723572
- `plain_abs_error/std`: 0.0
- `plain_abs_error/min`: 0.3282093405723572
- `plain_abs_error/max`: 0.3282093405723572
- `non_plain_abs_error/mean`: 0.2921072840690613
- `non_plain_abs_error/std`: 0.0
- `non_plain_abs_error/min`: 0.2921072840690613
- `non_plain_abs_error/max`: 0.2921072840690613
- `plain_initial_noise_std/mean`: 0.9824392199516296
- `plain_initial_noise_std/std`: 0.0
- `plain_initial_noise_std/min`: 0.9824392199516296
- `plain_initial_noise_std/max`: 0.9824392199516296
- `non_plain_initial_noise_std/mean`: 1.0324267148971558
- `non_plain_initial_noise_std/std`: 0.0
- `non_plain_initial_noise_std/min`: 1.0324267148971558
- `non_plain_initial_noise_std/max`: 1.0324267148971558
- `plain_inverted_noise_std/mean`: 0.8909876942634583
- `plain_inverted_noise_std/std`: 0.0
- `plain_inverted_noise_std/min`: 0.8909876942634583
- `plain_inverted_noise_std/max`: 0.8909876942634583
- `non_plain_inverted_noise_std/mean`: 0.9618687033653259
- `non_plain_inverted_noise_std/std`: 0.0
- `non_plain_inverted_noise_std/min`: 0.9618687033653259
- `non_plain_inverted_noise_std/max`: 0.9618687033653259
- `plain_pixel_fraction/mean`: 0.6044921875
- `plain_pixel_fraction/std`: 0.0
- `plain_pixel_fraction/min`: 0.6044921875
- `plain_pixel_fraction/max`: 0.6044921875

## Notes

- This runner evaluates saved generation trajectories.
- If present, pred_noises are included as forward DDIM reference targets.
- If present, inverted_noise is compared against initial latent noise x_T.
- If present, normality diagnostics compare initial and inverted noise.
- Latent normality uses patch top-k correlation and global Gaussian KL, matching the authors' KL definition; per-location KL is also reported separately.
- If present, reconstructed.png is compared against final.png.
- LPIPS uses the AlexNet v0.1 perceptual metric when available.
- Plain-area reconstruction metrics use final.png local pixel differences.
- Inversion diagnostics include latent-location and prediction-error metrics.
- CLIP target and directional metrics require target_prompt.
- Editing metrics need paired edited images.
