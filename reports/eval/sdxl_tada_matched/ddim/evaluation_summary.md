# Evaluation Summary

- Input directory: `/tmp/sdxl-tada-metrics-84444/tmp/sdxl_tada_matched_metric_views/ddim`
- Samples: 1522

- Noise comparison artifacts: `noise_comparisons/`

- Normality artifacts: `normality/`

- Image reconstruction artifacts: `image_comparisons/`

- Masked noise-region artifacts: `masked_noise_regions/`

## Aggregate

### initial_latent_stats

- `mean`: 0.00199610716663301
- `std`: 0.9967097640037537
- `min`: -4.6640625
- `p01`: -2.314453125
- `p05`: -1.638671875
- `p50`: 0.005596160888671875
- `p95`: 1.640625
- `p99`: 2.3203125
- `max`: 4.5546875
- `skew`: -0.0044959294609725475
- `excess_kurtosis`: -0.021062850952148438
- `abs_mean_error_from_normal`: 0.00199610716663301
- `abs_std_error_from_normal`: 0.003292679786682129
- `normal_kl_from_standard`: 1.2850388884544373e-05

### final_latent_stats

- `mean`: 0.06361760944128036
- `std`: 0.5870237946510315
- `min`: -2.369140625
- `p01`: -1.3017578125
- `p05`: -0.89453125
- `p50`: 0.059661865234375
- `p95`: 1.03125
- `p99`: 1.43359375
- `max`: 2.697265625
- `skew`: 0.018799640238285065
- `excess_kurtosis`: -0.0192413330078125
- `abs_mean_error_from_normal`: 0.06361760944128036
- `abs_std_error_from_normal`: 0.4129776358604431
- `normal_kl_from_standard`: 0.20701360702514648

### initial_patch_topk_correlation

- `mean`: 0.09424944221973419
- `std`: 0.0065282476134598255

### final_patch_topk_correlation

- `mean`: 0.6351078748703003
- `std`: 0.046941354870796204

### inverted_noise_stats

- `mean`: 0.017558472231030464
- `std`: 0.9190342426300049
- `min`: -4.56640625
- `p01`: -2.0703125
- `p05`: -1.482421875
- `p50`: 0.014102935791015625
- `p95`: 1.5341796875
- `p99`: 2.14453125
- `max`: 4.4296875
- `skew`: 0.02277977019548416
- `excess_kurtosis`: -0.0964808464050293
- `abs_mean_error_from_normal`: 0.017558472231030464
- `abs_std_error_from_normal`: 0.08096802234649658
- `normal_kl_from_standard`: 0.0068983882665634155

### initial_vs_inverted_noise

- `mse`: 0.06256267428398132
- `rmse`: 0.25012531876564026
- `mae`: 0.18948037922382355
- `l2`: 2467.009765625
- `cosine`: 0.9910526275634766
- `relative_rmse_to_reference_std`: 0.2501223385334015
- `candidate_to_reference_std_ratio`: 0.9219961762428284
- `mean_shift`: 0.01598251983523369

### inverted_noise_patch_topk_correlation

- `mean`: 0.16599006950855255
- `std`: 0.0180758498609066

### latent_normality

- `corr`: 0.16599006950855255
- `corr_std`: 0.0180758498609066
- `kl`: 0.007118105888366699
- `reverse_kl`: 0.0063803791999816895
- `symmetric_kl`: 0.006749242544174194
- `per_location_kl`: 0.009244385175406933
- `per_location_reverse_kl`: 0.007935167290270329
- `per_location_symmetric_kl`: 0.00858977623283863

### initial_vs_inverted_noise_per_channel

- `channel_0/mse`: 0.07826071232557297
- `channel_0/rmse`: 0.27975115180015564
- `channel_0/mae`: 0.20893655717372894
- `channel_0/l2`: 1394.83349609375
- `channel_0/cosine`: 0.9657914638519287
- `channel_0/relative_rmse_to_reference_std`: 0.2797131836414337
- `channel_0/candidate_to_reference_std_ratio`: 0.8882191181182861
- `channel_0/mean_shift`: 0.014567900449037552
- `channel_1/mse`: 0.05371075123548508
- `channel_1/rmse`: 0.2317558079957962
- `channel_1/mae`: 0.17773288488388062
- `channel_1/l2`: 1155.4102783203125
- `channel_1/cosine`: 0.9763098955154419
- `channel_1/relative_rmse_to_reference_std`: 0.2317582368850708
- `channel_1/candidate_to_reference_std_ratio`: 0.9361051321029663
- `channel_1/mean_shift`: 0.002931587165221572
- `channel_2/mse`: 0.045584067702293396
- `channel_2/rmse`: 0.21350425481796265
- `channel_2/mae`: 0.16344504058361053
- `channel_2/l2`: 1064.596435546875
- `channel_2/cosine`: 0.979797899723053
- `channel_2/relative_rmse_to_reference_std`: 0.21352529525756836
- `channel_2/candidate_to_reference_std_ratio`: 0.966734766960144
- `channel_2/mean_shift`: 0.01563970185816288
- `channel_3/mse`: 0.07269519567489624
- `channel_3/rmse`: 0.26962047815322876
- `channel_3/mae`: 0.2078070044517517
- `channel_3/l2`: 1344.4298095703125
- `channel_3/cosine`: 0.9683191776275635
- `channel_3/relative_rmse_to_reference_std`: 0.2696148157119751
- `channel_3/candidate_to_reference_std_ratio`: 0.8945072889328003
- `channel_3/mean_shift`: 0.030790887773036957

### inversion_error_stats

- `signed_mean`: 0.01598251610994339
- `signed_std`: 0.2496141791343689
- `abs_mean`: 0.1892547607421875
- `abs_std`: 0.16286571323871613
- `abs_p50`: 0.14817047119140625
- `abs_p90`: 0.405029296875
- `abs_p95`: 0.5078125
- `abs_p99`: 0.7412109375
- `abs_max`: 1.7744140625

### inversion_error_patch_topk_correlation

- `mean`: 0.25001823902130127
- `std`: 0.014858909882605076

### inversion_error_structure

- `signed_error_vs_final_latent_cosine`: 0.04686989635229111
- `abs_error_vs_abs_final_latent_cosine`: 0.6164788007736206

### final_vs_reconstructed_image

- `mse/mean`: 0.0018546768696978688
- `mse/std`: 0.0014616071712225676
- `mse/min`: 0.00020971731282770634
- `mse/max`: 0.014613124541938305
- `rmse/mean`: 0.04079771041870117
- `rmse/std`: 0.013792159967124462
- `rmse/min`: 0.01448161993175745
- `rmse/max`: 0.12088475376367569
- `mae/mean`: 0.027119670063257217
- `mae/std`: 0.009893917478621006
- `mae/min`: 0.01017795316874981
- `mae/max`: 0.09148144721984863
- `psnr_db/mean`: 28.21846580505371
- `psnr_db/std`: 2.682081699371338
- `psnr_db/min`: 18.352569580078125
- `psnr_db/max`: 36.78365707397461
- `ssim/mean`: 0.8060412406921387
- `ssim/std`: 0.07205023616552353
- `ssim/min`: 0.3450237214565277
- `ssim/max`: 0.9430671334266663
- `max_abs_error/mean`: 0.5542449951171875
- `max_abs_error/std`: 0.10835085809230804
- `max_abs_error/min`: 0.21568629145622253
- `max_abs_error/max`: 0.9843137264251709
- `mean_signed_error/mean`: -0.0015026883920654655
- `mean_signed_error/std`: 0.003576181596145034
- `mean_signed_error/min`: -0.032128505408763885
- `mean_signed_error/max`: 0.019420428201556206
- `plain_threshold/mean`: 0.02500000223517418
- `plain_threshold/std`: 1.862645149230957e-09
- `plain_threshold/min`: 0.02500000037252903
- `plain_threshold/max`: 0.02500000037252903
- `plain_pixel_count/mean`: 709561.25
- `plain_pixel_count/std`: 114920.2734375
- `plain_pixel_count/min`: 303396.0
- `plain_pixel_count/max`: 993602.0
- `plain_pixel_fraction/mean`: 0.6766903400421143
- `plain_pixel_fraction/std`: 0.10959651321172714
- `plain_pixel_fraction/min`: 0.2893409729003906
- `plain_pixel_fraction/max`: 0.9475727081298828
- `plain_mse/mean`: 0.0011315351584926248
- `plain_mse/std`: 0.001117889885790646
- `plain_mse/min`: 0.00016093437443487346
- `plain_mse/max`: 0.01205922570079565
- `plain_rmse/mean`: 0.031290214508771896
- `plain_rmse/std`: 0.01234736479818821
- `plain_rmse/min`: 0.012685990892350674
- `plain_rmse/max`: 0.10981450229883194
- `plain_mae/mean`: 0.021289704367518425
- `plain_mae/std`: 0.008814718574285507
- `plain_mae/min`: 0.0071779112331569195
- `plain_mae/max`: 0.08234957605600357
- `plain_max_abs_error/mean`: 0.45533743500709534
- `plain_max_abs_error/std`: 0.1052756980061531
- `plain_max_abs_error/min`: 0.18823528289794922
- `plain_max_abs_error/max`: 0.8392156958580017
- `plain_mean_signed_error/mean`: -0.0021667180117219687
- `plain_mean_signed_error/std`: 0.005063383840024471
- `plain_mean_signed_error/min`: -0.058053504675626755
- `plain_mean_signed_error/max`: 0.022237755358219147
- `non_plain_pixel_count/mean`: 339014.71875
- `non_plain_pixel_count/std`: 114920.2734375
- `non_plain_pixel_count/min`: 54974.0
- `non_plain_pixel_count/max`: 745180.0
- `non_plain_pixel_fraction/mean`: 0.32330963015556335
- `non_plain_pixel_fraction/std`: 0.10959651321172714
- `non_plain_pixel_fraction/min`: 0.05242729187011719
- `non_plain_pixel_fraction/max`: 0.7106590270996094
- `non_plain_mse/mean`: 0.003374455263838172
- `non_plain_mse/std`: 0.002256652805954218
- `non_plain_mse/min`: 0.0007053891895338893
- `non_plain_mse/max`: 0.02407275140285492
- `non_plain_rmse/mean`: 0.055825527757406235
- `non_plain_rmse/std`: 0.0160613264888525
- `non_plain_rmse/min`: 0.026559164747595787
- `non_plain_rmse/max`: 0.15515395998954773
- `non_plain_mae/mean`: 0.03954603523015976
- `non_plain_mae/std`: 0.011498977430164814
- `non_plain_mae/min`: 0.01943705603480339
- `non_plain_mae/max`: 0.11682238429784775
- `non_plain_max_abs_error/mean`: 0.5526268482208252
- `non_plain_max_abs_error/std`: 0.10851702094078064
- `non_plain_max_abs_error/min`: 0.21568626165390015
- `non_plain_max_abs_error/max`: 0.9843137264251709
- `non_plain_mean_signed_error/mean`: 2.5876840936689405e-06
- `non_plain_mean_signed_error/std`: 0.0042501273564994335
- `non_plain_mean_signed_error/min`: -0.021389158442616463
- `non_plain_mean_signed_error/max`: 0.03626075014472008
- `lpips/mean`: 0.10042189806699753
- `lpips/std`: 0.05850498750805855
- `lpips/min`: 0.022699685767292976
- `lpips/max`: 0.6541187763214111

### initial_vs_inverted_noise_normality

- `sample_size/mean`: 5000.0
- `sample_size/std`: 0.0
- `sample_size/min`: 5000.0
- `sample_size/max`: 5000.0
- `num_quantiles/mean`: 201.0
- `num_quantiles/std`: 0.0
- `num_quantiles/min`: 201.0
- `num_quantiles/max`: 201.0
- `initial_shapiro_p_value/mean`: 0.5172836780548096
- `initial_shapiro_p_value/std`: 0.27771735191345215
- `initial_shapiro_p_value/min`: 0.0006801433628425002
- `initial_shapiro_p_value/max`: 0.999314546585083
- `inverted_shapiro_p_value/mean`: 0.27470651268959045
- `inverted_shapiro_p_value/std`: 0.26601800322532654
- `inverted_shapiro_p_value/min`: 9.197891813528258e-06
- `inverted_shapiro_p_value/max`: 0.9987606406211853
- `gaussian_kl_initial_to_inverted/mean`: 0.007684766314923763
- `gaussian_kl_initial_to_inverted/std`: 0.004070598632097244
- `gaussian_kl_initial_to_inverted/min`: 0.0012333504855632782
- `gaussian_kl_initial_to_inverted/max`: 0.030625134706497192
- `gaussian_kl_inverted_to_initial/mean`: 0.006790827959775925
- `gaussian_kl_inverted_to_initial/std`: 0.003354295389726758
- `gaussian_kl_inverted_to_initial/min`: 0.0011776387691497803
- `gaussian_kl_inverted_to_initial/max`: 0.024563252925872803
- `gaussian_kl_symmetric/mean`: 0.007237797603011131
- `gaussian_kl_symmetric/std`: 0.003711982164531946
- `gaussian_kl_symmetric/min`: 0.0012054946273565292
- `gaussian_kl_symmetric/max`: 0.027594193816184998

### initial_vs_inverted_noise_by_plain_region

- `plain_abs_error/mean`: 0.2065671682357788
- `plain_abs_error/std`: 0.025919128209352493
- `plain_abs_error/min`: 0.12374776601791382
- `plain_abs_error/max`: 0.36172783374786377
- `non_plain_abs_error/mean`: 0.17688436806201935
- `non_plain_abs_error/std`: 0.01845691166818142
- `non_plain_abs_error/min`: 0.12296148389577866
- `non_plain_abs_error/max`: 0.2596331238746643
- `plain_initial_noise_std/mean`: 0.9687893986701965
- `plain_initial_noise_std/std`: 0.010329981334507465
- `plain_initial_noise_std/min`: 0.932158350944519
- `plain_initial_noise_std/max`: 0.9939183592796326
- `non_plain_initial_noise_std/mean`: 1.021799087524414
- `non_plain_initial_noise_std/std`: 0.008673234842717648
- `non_plain_initial_noise_std/min`: 0.9996885657310486
- `non_plain_initial_noise_std/max`: 1.069328784942627
- `plain_inverted_noise_std/mean`: 0.8734233379364014
- `plain_inverted_noise_std/std`: 0.02212083525955677
- `plain_inverted_noise_std/min`: 0.6631709337234497
- `plain_inverted_noise_std/max`: 0.9255744218826294
- `non_plain_inverted_noise_std/mean`: 0.9552851915359497
- `non_plain_inverted_noise_std/std`: 0.015079773031175137
- `non_plain_inverted_noise_std/min`: 0.8852312564849854
- `non_plain_inverted_noise_std/max`: 1.0102276802062988
- `plain_pixel_fraction/mean`: 0.4217347204685211
- `plain_pixel_fraction/std`: 0.1409408450126648
- `plain_pixel_fraction/min`: 0.07342529296875
- `plain_pixel_fraction/max`: 0.84228515625

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
