# Evaluation Summary

- Input directory: `/tmp/sdxl-tada-metrics-84439/tmp/sdxl_tada_matched_metric_views/ddim`
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

- `mean`: 0.021003669127821922
- `std`: 0.9076108336448669
- `min`: -4.203125
- `p01`: -2.037109375
- `p05`: -1.45703125
- `p50`: 0.01342010498046875
- `p95`: 1.5283203125
- `p99`: 2.125
- `max`: 3.884765625
- `skew`: 0.032572366297245026
- `excess_kurtosis`: -0.09022378921508789
- `abs_mean_error_from_normal`: 0.021003669127821922
- `abs_std_error_from_normal`: 0.09239614009857178
- `normal_kl_from_standard`: 0.009040221571922302

### initial_vs_inverted_noise

- `mse`: 0.07776221632957458
- `rmse`: 0.2788587808609009
- `mae`: 0.21497797966003418
- `l2`: 71.38801574707031
- `cosine`: 0.9622006416320801
- `relative_rmse_to_reference_std`: 0.2781607508659363
- `candidate_to_reference_std_ratio`: 0.9053319692611694
- `mean_shift`: 0.01684667356312275

### inverted_noise_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### latent_normality

- `corr`: 0.0
- `corr_std`: 0.0
- `kl`: 0.01075301319360733
- `reverse_kl`: 0.009407766163349152
- `symmetric_kl`: 0.010080389678478241
- `per_location_kl`: 38881103872.0
- `per_location_reverse_kl`: 38881103872.0
- `per_location_symmetric_kl`: 38881103872.0

### initial_vs_inverted_noise_per_channel

- `channel_0/mse`: 0.11478342115879059
- `channel_0/rmse`: 0.33879703283309937
- `channel_0/mae`: 0.2638917565345764
- `channel_0/l2`: 43.366031646728516
- `channel_0/cosine`: 0.949950635433197
- `channel_0/relative_rmse_to_reference_std`: 0.33647090196609497
- `channel_0/candidate_to_reference_std_ratio`: 0.8247681856155396
- `channel_0/mean_shift`: 0.013525981456041336
- `channel_1/mse`: 0.0615655779838562
- `channel_1/rmse`: 0.2481241226196289
- `channel_1/mae`: 0.1940198689699173
- `channel_1/l2`: 31.759876251220703
- `channel_1/cosine`: 0.9697453379631042
- `channel_1/relative_rmse_to_reference_std`: 0.24616530537605286
- `channel_1/candidate_to_reference_std_ratio`: 0.9379896521568298
- `channel_1/mean_shift`: 0.00797894038259983
- `channel_2/mse`: 0.053584031760692596
- `channel_2/rmse`: 0.23148225247859955
- `channel_2/mae`: 0.17986252903938293
- `channel_2/l2`: 29.629711151123047
- `channel_2/cosine`: 0.9728034138679504
- `channel_2/relative_rmse_to_reference_std`: 0.23167850077152252
- `channel_2/candidate_to_reference_std_ratio`: 0.9681167602539062
- `channel_2/mean_shift`: 0.019868070259690285
- `channel_3/mse`: 0.08111581206321716
- `channel_3/rmse`: 0.28480836749076843
- `channel_3/mae`: 0.22213774919509888
- `channel_3/l2`: 36.45548629760742
- `channel_3/cosine`: 0.9613027572631836
- `channel_3/relative_rmse_to_reference_std`: 0.28597474098205566
- `channel_3/candidate_to_reference_std_ratio`: 0.884270966053009
- `channel_3/mean_shift`: 0.026013702154159546

### inversion_error_stats

- `signed_mean`: 0.01684667356312275
- `signed_std`: 0.2783494293689728
- `abs_mean`: 0.21497797966003418
- `abs_std`: 0.17761385440826416
- `abs_p50`: 0.173583984375
- `abs_p90`: 0.4521484375
- `abs_p95`: 0.5576171875
- `abs_p99`: 0.794921875
- `abs_max`: 1.548828125

### inversion_error_patch_topk_correlation

- `mean`: 0.0
- `std`: 0.0

### inversion_error_structure

- `signed_error_vs_final_latent_cosine`: 0.04633716121315956
- `abs_error_vs_abs_final_latent_cosine`: 0.6105996370315552

### final_vs_reconstructed_image

- `mse/mean`: 0.004241123329848051
- `mse/std`: 0.0
- `mse/min`: 0.004241123329848051
- `mse/max`: 0.004241123329848051
- `rmse/mean`: 0.06512390822172165
- `rmse/std`: 0.0
- `rmse/min`: 0.06512390822172165
- `rmse/max`: 0.06512390822172165
- `mae/mean`: 0.04252990707755089
- `mae/std`: 0.0
- `mae/min`: 0.04252990707755089
- `mae/max`: 0.04252990707755089
- `psnr_db/mean`: 23.725189208984375
- `psnr_db/std`: 0.0
- `psnr_db/min`: 23.725189208984375
- `psnr_db/max`: 23.725189208984375
- `ssim/mean`: 0.7192643284797668
- `ssim/std`: 0.0
- `ssim/min`: 0.7192643284797668
- `ssim/max`: 0.7192643284797668
- `max_abs_error/mean`: 0.5843137502670288
- `max_abs_error/std`: 0.0
- `max_abs_error/min`: 0.5843137502670288
- `max_abs_error/max`: 0.5843137502670288
- `mean_signed_error/mean`: -0.007709265220910311
- `mean_signed_error/std`: 0.0
- `mean_signed_error/min`: -0.007709265220910311
- `mean_signed_error/max`: -0.007709265220910311
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
- `plain_mse/mean`: 0.0034986247774213552
- `plain_mse/std`: 0.0
- `plain_mse/min`: 0.0034986247774213552
- `plain_mse/max`: 0.0034986247774213552
- `plain_rmse/mean`: 0.05914917215704918
- `plain_rmse/std`: 0.0
- `plain_rmse/min`: 0.05914917215704918
- `plain_rmse/max`: 0.05914917215704918
- `plain_mae/mean`: 0.038264770060777664
- `plain_mae/std`: 0.0
- `plain_mae/min`: 0.038264770060777664
- `plain_mae/max`: 0.038264770060777664
- `plain_max_abs_error/mean`: 0.5411764979362488
- `plain_max_abs_error/std`: 0.0
- `plain_max_abs_error/min`: 0.5411764979362488
- `plain_max_abs_error/max`: 0.5411764979362488
- `plain_mean_signed_error/mean`: -0.011021631769835949
- `plain_mean_signed_error/std`: 0.0
- `plain_mean_signed_error/min`: -0.011021631769835949
- `plain_mean_signed_error/max`: -0.011021631769835949
- `non_plain_pixel_count/mean`: 240161.0
- `non_plain_pixel_count/std`: 0.0
- `non_plain_pixel_count/min`: 240161.0
- `non_plain_pixel_count/max`: 240161.0
- `non_plain_pixel_fraction/mean`: 0.2290353775024414
- `non_plain_pixel_fraction/std`: 0.0
- `non_plain_pixel_fraction/min`: 0.2290353775024414
- `non_plain_pixel_fraction/max`: 0.2290353775024414
- `non_plain_mse/mean`: 0.006740476004779339
- `non_plain_mse/std`: 0.0
- `non_plain_mse/min`: 0.006740476004779339
- `non_plain_mse/max`: 0.006740476004779339
- `non_plain_rmse/mean`: 0.08210039883852005
- `non_plain_rmse/std`: 0.0
- `non_plain_rmse/min`: 0.08210039883852005
- `non_plain_rmse/max`: 0.08210039883852005
- `non_plain_mae/mean`: 0.0568869523704052
- `non_plain_mae/std`: 0.0
- `non_plain_mae/min`: 0.0568869523704052
- `non_plain_mae/max`: 0.0568869523704052
- `non_plain_max_abs_error/mean`: 0.5843137502670288
- `non_plain_max_abs_error/std`: 0.0
- `non_plain_max_abs_error/min`: 0.5843137502670288
- `non_plain_max_abs_error/max`: 0.5843137502670288
- `non_plain_mean_signed_error/mean`: 0.003440618049353361
- `non_plain_mean_signed_error/std`: 0.0
- `non_plain_mean_signed_error/min`: 0.003440618049353361
- `non_plain_mean_signed_error/max`: 0.003440618049353361
- `lpips/mean`: 0.20195342600345612
- `lpips/std`: 0.0
- `lpips/min`: 0.20195342600345612
- `lpips/max`: 0.20195342600345612

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
- `inverted_shapiro_p_value/mean`: 0.17146848142147064
- `inverted_shapiro_p_value/std`: 0.0
- `inverted_shapiro_p_value/min`: 0.17146848142147064
- `inverted_shapiro_p_value/max`: 0.17146848142147064
- `gaussian_kl_initial_to_inverted/mean`: 0.012076526880264282
- `gaussian_kl_initial_to_inverted/std`: 0.0
- `gaussian_kl_initial_to_inverted/min`: 0.012076526880264282
- `gaussian_kl_initial_to_inverted/max`: 0.012076526880264282
- `gaussian_kl_inverted_to_initial/mean`: 0.010484576225280762
- `gaussian_kl_inverted_to_initial/std`: 0.0
- `gaussian_kl_inverted_to_initial/min`: 0.010484576225280762
- `gaussian_kl_inverted_to_initial/max`: 0.010484576225280762
- `gaussian_kl_symmetric/mean`: 0.011280551552772522
- `gaussian_kl_symmetric/std`: 0.0
- `gaussian_kl_symmetric/min`: 0.011280551552772522
- `gaussian_kl_symmetric/max`: 0.011280551552772522

### initial_vs_inverted_noise_by_plain_region

- `plain_abs_error/mean`: 0.22653089463710785
- `plain_abs_error/std`: 0.0
- `plain_abs_error/min`: 0.22653089463710785
- `plain_abs_error/max`: 0.22653089463710785
- `non_plain_abs_error/mean`: 0.19732052087783813
- `non_plain_abs_error/std`: 0.0
- `non_plain_abs_error/min`: 0.19732052087783813
- `non_plain_abs_error/max`: 0.19732052087783813
- `plain_initial_noise_std/mean`: 0.9824392199516296
- `plain_initial_noise_std/std`: 0.0
- `plain_initial_noise_std/min`: 0.9824392199516296
- `plain_initial_noise_std/max`: 0.9824392199516296
- `non_plain_initial_noise_std/mean`: 1.0324267148971558
- `non_plain_initial_noise_std/std`: 0.0
- `non_plain_initial_noise_std/min`: 1.0324267148971558
- `non_plain_initial_noise_std/max`: 1.0324267148971558
- `plain_inverted_noise_std/mean`: 0.8759334683418274
- `plain_inverted_noise_std/std`: 0.0
- `plain_inverted_noise_std/min`: 0.8759334683418274
- `plain_inverted_noise_std/max`: 0.8759334683418274
- `non_plain_inverted_noise_std/mean`: 0.9539780616760254
- `non_plain_inverted_noise_std/std`: 0.0
- `non_plain_inverted_noise_std/min`: 0.9539780616760254
- `non_plain_inverted_noise_std/max`: 0.9539780616760254
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
