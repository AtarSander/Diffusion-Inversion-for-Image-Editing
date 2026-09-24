# Evaluation Summary

- Input directory: `/tmp/sdxl-tada-metrics-84439/tmp/sdxl_tada_matched_metric_views/forward_4pct`
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

- `mean`: 0.015822920948266983
- `std`: 0.9492810368537903
- `min`: -4.5546875
- `p01`: -2.162109375
- `p05`: -1.533203125
- `p50`: 0.01053619384765625
- `p95`: 1.580078125
- `p99`: 2.21875
- `max`: 3.927734375
- `skew`: 0.013209296390414238
- `excess_kurtosis`: -0.0687708854675293
- `abs_mean_error_from_normal`: 0.015822920948266983
- `abs_std_error_from_normal`: 0.050726234912872314
- `normal_kl_from_standard`: 0.002743557095527649

### initial_vs_inverted_noise

- `mse`: 0.5183924436569214
- `rmse`: 0.7199947237968445
- `mae`: 0.563494086265564
- `l2`: 184.31900024414062
- `cosine`: 0.7291561365127563
- `relative_rmse_to_reference_std`: 0.718192458152771
- `candidate_to_reference_std_ratio`: 0.946897566318512
- `mean_shift`: 0.01166592538356781

### inverted_noise_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### latent_normality

- `corr`: 0.0
- `corr_std`: 0.0
- `kl`: 0.003164108842611313
- `reverse_kl`: 0.0029395632445812225
- `symmetric_kl`: 0.0030518360435962677
- `per_location_kl`: 259196256256.0
- `per_location_reverse_kl`: 259196256256.0
- `per_location_symmetric_kl`: 259196256256.0

### initial_vs_inverted_noise_per_channel

- `channel_0/mse`: 0.7242806553840637
- `channel_0/rmse`: 0.8510468006134033
- `channel_0/mae`: 0.674720287322998
- `channel_0/l2`: 108.9340591430664
- `channel_0/cosine`: 0.6089150905609131
- `channel_0/relative_rmse_to_reference_std`: 0.8452036380767822
- `channel_0/candidate_to_reference_std_ratio`: 0.9005476236343384
- `channel_0/mean_shift`: 0.009578507393598557
- `channel_1/mse`: 0.45975834131240845
- `channel_1/rmse`: 0.6780548095703125
- `channel_1/mae`: 0.5358635187149048
- `channel_1/l2`: 86.79108428955078
- `channel_1/cosine`: 0.764759361743927
- `channel_1/relative_rmse_to_reference_std`: 0.672701895236969
- `channel_1/candidate_to_reference_std_ratio`: 0.9580820798873901
- `channel_1/mean_shift`: 0.004324748180806637
- `channel_2/mse`: 0.3689049482345581
- `channel_2/rmse`: 0.60737544298172
- `channel_2/mae`: 0.4752948582172394
- `channel_2/l2`: 77.74411010742188
- `channel_2/cosine`: 0.812220573425293
- `channel_2/relative_rmse_to_reference_std`: 0.6078903675079346
- `channel_2/candidate_to_reference_std_ratio`: 0.9828883409500122
- `channel_2/mean_shift`: 0.012654893100261688
- `channel_3/mse`: 0.5206259489059448
- `channel_3/rmse`: 0.7215441465377808
- `channel_3/mae`: 0.5680978894233704
- `channel_3/l2`: 92.35770416259766
- `channel_3/cosine`: 0.7238338589668274
- `channel_3/relative_rmse_to_reference_std`: 0.7244990468025208
- `channel_3/candidate_to_reference_std_ratio`: 0.9446524977684021
- `channel_3/mean_shift`: 0.02010555937886238

### inversion_error_stats

- `signed_mean`: 0.011665926314890385
- `signed_std`: 0.7199002504348755
- `abs_mean`: 0.563494086265564
- `abs_std`: 0.44818171858787537
- `abs_p50`: 0.4609375
- `abs_p90`: 1.18310546875
- `abs_p95`: 1.43218994140625
- `abs_p99`: 1.9572701454162598
- `abs_max`: 3.6762847900390625

### inversion_error_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### inversion_error_structure

- `signed_error_vs_final_latent_cosine`: -0.051989804953336716
- `abs_error_vs_abs_final_latent_cosine`: 0.619971752166748

### final_vs_reconstructed_image

- `mse/mean`: 0.0035475909244269133
- `mse/std`: 0.0
- `mse/min`: 0.0035475909244269133
- `mse/max`: 0.0035475909244269133
- `rmse/mean`: 0.059561654925346375
- `rmse/std`: 0.0
- `rmse/min`: 0.059561654925346375
- `rmse/max`: 0.059561654925346375
- `mae/mean`: 0.04043552279472351
- `mae/std`: 0.0
- `mae/min`: 0.04043552279472351
- `mae/max`: 0.04043552279472351
- `psnr_db/mean`: 24.50066566467285
- `psnr_db/std`: 0.0
- `psnr_db/min`: 24.50066566467285
- `psnr_db/max`: 24.50066566467285
- `ssim/mean`: 0.7053281664848328
- `ssim/std`: 0.0
- `ssim/min`: 0.7053281664848328
- `ssim/max`: 0.7053281664848328
- `max_abs_error/mean`: 0.6078431606292725
- `max_abs_error/std`: 0.0
- `max_abs_error/min`: 0.6078431606292725
- `max_abs_error/max`: 0.6078431606292725
- `mean_signed_error/mean`: -0.003147648647427559
- `mean_signed_error/std`: 0.0
- `mean_signed_error/min`: -0.003147648647427559
- `mean_signed_error/max`: -0.003147648647427559
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
- `plain_mse/mean`: 0.00286884605884552
- `plain_mse/std`: 0.0
- `plain_mse/min`: 0.00286884605884552
- `plain_mse/max`: 0.00286884605884552
- `plain_rmse/mean`: 0.053561609238386154
- `plain_rmse/std`: 0.0
- `plain_rmse/min`: 0.053561609238386154
- `plain_rmse/max`: 0.053561609238386154
- `plain_mae/mean`: 0.03618704155087471
- `plain_mae/std`: 0.0
- `plain_mae/min`: 0.03618704155087471
- `plain_mae/max`: 0.03618704155087471
- `plain_max_abs_error/mean`: 0.5333333611488342
- `plain_max_abs_error/std`: 0.0
- `plain_max_abs_error/min`: 0.5333333611488342
- `plain_max_abs_error/max`: 0.5333333611488342
- `plain_mean_signed_error/mean`: -0.006002890411764383
- `plain_mean_signed_error/std`: 0.0
- `plain_mean_signed_error/min`: -0.006002890411764383
- `plain_mean_signed_error/max`: -0.006002890411764383
- `non_plain_pixel_count/mean`: 240161.0
- `non_plain_pixel_count/std`: 0.0
- `non_plain_pixel_count/min`: 240161.0
- `non_plain_pixel_count/max`: 240161.0
- `non_plain_pixel_fraction/mean`: 0.2290353775024414
- `non_plain_pixel_fraction/std`: 0.0
- `non_plain_pixel_fraction/min`: 0.2290353775024414
- `non_plain_pixel_fraction/max`: 0.2290353775024414
- `non_plain_mse/mean`: 0.005832340102642775
- `non_plain_mse/std`: 0.0
- `non_plain_mse/min`: 0.005832340102642775
- `non_plain_mse/max`: 0.005832340102642775
- `non_plain_rmse/mean`: 0.0763697624206543
- `non_plain_rmse/std`: 0.0
- `non_plain_rmse/min`: 0.0763697624206543
- `non_plain_rmse/max`: 0.0763697624206543
- `non_plain_mae/mean`: 0.05473649129271507
- `non_plain_mae/std`: 0.0
- `non_plain_mae/min`: 0.05473649129271507
- `non_plain_mae/max`: 0.05473649129271507
- `non_plain_max_abs_error/mean`: 0.6078431606292725
- `non_plain_max_abs_error/std`: 0.0
- `non_plain_max_abs_error/min`: 0.6078431606292725
- `non_plain_max_abs_error/max`: 0.6078431606292725
- `non_plain_mean_signed_error/mean`: 0.006463491823524237
- `non_plain_mean_signed_error/std`: 0.0
- `non_plain_mean_signed_error/min`: 0.006463491823524237
- `non_plain_mean_signed_error/max`: 0.006463491823524237
- `lpips/mean`: 0.19006654620170593
- `lpips/std`: 0.0
- `lpips/min`: 0.19006654620170593
- `lpips/max`: 0.19006654620170593

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
- `inverted_shapiro_p_value/mean`: 0.21626624464988708
- `inverted_shapiro_p_value/std`: 0.0
- `inverted_shapiro_p_value/min`: 0.21626624464988708
- `inverted_shapiro_p_value/max`: 0.21626624464988708
- `gaussian_kl_initial_to_inverted/mean`: 0.004395671188831329
- `gaussian_kl_initial_to_inverted/std`: 0.0
- `gaussian_kl_initial_to_inverted/min`: 0.004395671188831329
- `gaussian_kl_initial_to_inverted/max`: 0.004395671188831329
- `gaussian_kl_inverted_to_initial/mean`: 0.004031822085380554
- `gaussian_kl_inverted_to_initial/std`: 0.0
- `gaussian_kl_inverted_to_initial/min`: 0.004031822085380554
- `gaussian_kl_inverted_to_initial/max`: 0.004031822085380554
- `gaussian_kl_symmetric/mean`: 0.004213746637105942
- `gaussian_kl_symmetric/std`: 0.0
- `gaussian_kl_symmetric/min`: 0.004213746637105942
- `gaussian_kl_symmetric/max`: 0.004213746637105942

### initial_vs_inverted_noise_by_plain_region

- `plain_abs_error/mean`: 0.5776560306549072
- `plain_abs_error/std`: 0.0
- `plain_abs_error/min`: 0.5776560306549072
- `plain_abs_error/max`: 0.5776560306549072
- `non_plain_abs_error/mean`: 0.5418490767478943
- `non_plain_abs_error/std`: 0.0
- `non_plain_abs_error/min`: 0.5418490767478943
- `non_plain_abs_error/max`: 0.5418490767478943
- `plain_initial_noise_std/mean`: 0.9824392199516296
- `plain_initial_noise_std/std`: 0.0
- `plain_initial_noise_std/min`: 0.9824392199516296
- `plain_initial_noise_std/max`: 0.9824392199516296
- `non_plain_initial_noise_std/mean`: 1.0324267148971558
- `non_plain_initial_noise_std/std`: 0.0
- `non_plain_initial_noise_std/min`: 1.0324267148971558
- `non_plain_initial_noise_std/max`: 1.0324267148971558
- `plain_inverted_noise_std/mean`: 0.9307163953781128
- `plain_inverted_noise_std/std`: 0.0
- `plain_inverted_noise_std/min`: 0.9307163953781128
- `plain_inverted_noise_std/max`: 0.9307163953781128
- `non_plain_inverted_noise_std/mean`: 0.9769560098648071
- `non_plain_inverted_noise_std/std`: 0.0
- `non_plain_inverted_noise_std/min`: 0.9769560098648071
- `non_plain_inverted_noise_std/max`: 0.9769560098648071
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
