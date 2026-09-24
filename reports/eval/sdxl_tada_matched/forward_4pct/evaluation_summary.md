# Evaluation Summary

- Input directory: `/tmp/sdxl-tada-metrics-84444/tmp/sdxl_tada_matched_metric_views/forward_4pct`
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

- `mean`: 0.01290237344801426
- `std`: 0.9511927366256714
- `min`: -3.994140625
- `p01`: -2.171875
- `p05`: -1.5458984375
- `p50`: 0.0113677978515625
- `p95`: 1.576171875
- `p99`: 2.201202392578125
- `max`: 4.2265625
- `skew`: 0.00834033265709877
- `excess_kurtosis`: -0.09491491317749023
- `abs_mean_error_from_normal`: 0.01290237344801426
- `abs_std_error_from_normal`: 0.04880964756011963
- `normal_kl_from_standard`: 0.0025058574974536896

### initial_vs_inverted_noise

- `mse`: 0.44404664635658264
- `rmse`: 0.6663682460784912
- `mae`: 0.5208994746208191
- `l2`: 6576.8515625
- `cosine`: 0.7855448126792908
- `relative_rmse_to_reference_std`: 0.6663603186607361
- `candidate_to_reference_std_ratio`: 0.9533926844596863
- `mean_shift`: 0.01195303164422512

### inverted_noise_patch_topk_correlation

- `mean`: 0.11297877132892609
- `std`: 0.009599453769624233

### latent_normality

- `corr`: 0.11297877132892609
- `corr_std`: 0.009599453769624233
- `kl`: 0.0024308599531650543
- `reverse_kl`: 0.0022786445915699005
- `symmetric_kl`: 0.0023547522723674774
- `per_location_kl`: 0.0033786180429160595
- `per_location_reverse_kl`: 0.0030732890591025352
- `per_location_symmetric_kl`: 0.0032259535510092974

### initial_vs_inverted_noise_per_channel

- `channel_0/mse`: 0.561840295791626
- `channel_0/rmse`: 0.7495600581169128
- `channel_0/mae`: 0.5886129140853882
- `channel_0/l2`: 3737.75341796875
- `channel_0/cosine`: 0.7021776437759399
- `channel_0/relative_rmse_to_reference_std`: 0.7494583129882812
- `channel_0/candidate_to_reference_std_ratio`: 0.9277678728103638
- `channel_0/mean_shift`: 0.011082658544182777
- `channel_1/mse`: 0.41590699553489685
- `channel_1/rmse`: 0.6449084877967834
- `channel_1/mae`: 0.5077088475227356
- `channel_1/l2`: 3215.51220703125
- `channel_1/cosine`: 0.7862862944602966
- `channel_1/relative_rmse_to_reference_std`: 0.6449152827262878
- `channel_1/candidate_to_reference_std_ratio`: 0.9588368535041809
- `channel_1/mean_shift`: 0.0017341842176392674
- `channel_2/mse`: 0.32652151584625244
- `channel_2/rmse`: 0.5714206099510193
- `channel_2/mae`: 0.4457207918167114
- `channel_2/l2`: 2849.72412109375
- `channel_2/cosine`: 0.8363897204399109
- `channel_2/relative_rmse_to_reference_std`: 0.571476936340332
- `channel_2/candidate_to_reference_std_ratio`: 0.9825531840324402
- `channel_2/mean_shift`: 0.011335690505802631
- `channel_3/mse`: 0.47191765904426575
- `channel_3/rmse`: 0.6869626045227051
- `channel_3/mae`: 0.5415552258491516
- `channel_3/l2`: 3425.2470703125
- `channel_3/cosine`: 0.7538484930992126
- `channel_3/relative_rmse_to_reference_std`: 0.6869481801986694
- `channel_3/candidate_to_reference_std_ratio`: 0.9434385895729065
- `channel_3/mean_shift`: 0.023659594357013702

### inversion_error_stats

- `signed_mean`: 0.01195303350687027
- `signed_std`: 0.6662610173225403
- `abs_mean`: 0.5206989645957947
- `abs_std`: 0.41453731060028076
- `abs_p50`: 0.4261474609375
- `abs_p90`: 1.09033203125
- `abs_p95`: 1.328125
- `abs_p99`: 1.8272414207458496
- `abs_max`: 3.6259765625

### inversion_error_patch_topk_correlation

- `mean`: 0.1414078176021576
- `std`: 0.012372483499348164

### inversion_error_structure

- `signed_error_vs_final_latent_cosine`: -0.051194123923778534
- `abs_error_vs_abs_final_latent_cosine`: 0.6340053081512451

### final_vs_reconstructed_image

- `mse/mean`: 0.0029179605189710855
- `mse/std`: 0.0014847200363874435
- `mse/min`: 0.0004673281218856573
- `mse/max`: 0.013461374677717686
- `rmse/mean`: 0.05250292643904686
- `rmse/std`: 0.012704446911811829
- `rmse/min`: 0.021617773920297623
- `rmse/max`: 0.11602316796779633
- `mae/mean`: 0.035650670528411865
- `mae/std`: 0.00886287447065115
- `mae/min`: 0.014663944952189922
- `mae/max`: 0.0827173963189125
- `psnr_db/mean`: 25.84473991394043
- `psnr_db/std`: 2.0842692852020264
- `psnr_db/min`: 18.7091064453125
- `psnr_db/max`: 33.30377960205078
- `ssim/mean`: 0.7013041377067566
- `ssim/std`: 0.07735123485326767
- `ssim/min`: 0.4276444911956787
- `ssim/max`: 0.8995958566665649
- `max_abs_error/mean`: 0.598953902721405
- `max_abs_error/std`: 0.09971998631954193
- `max_abs_error/min`: 0.3176470696926117
- `max_abs_error/max`: 0.9019607901573181
- `mean_signed_error/mean`: -0.00103060994297266
- `mean_signed_error/std`: 0.002842117100954056
- `mean_signed_error/min`: -0.023484257981181145
- `mean_signed_error/max`: 0.01897156424820423
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
- `plain_mse/mean`: 0.0017013616161420941
- `plain_mse/std`: 0.0010114052565768361
- `plain_mse/min`: 0.00034187364508397877
- `plain_mse/max`: 0.01286480762064457
- `plain_rmse/mean`: 0.039934393018484116
- `plain_rmse/std`: 0.010325007140636444
- `plain_rmse/min`: 0.01848982460796833
- `plain_rmse/max`: 0.11342313885688782
- `plain_mae/mean`: 0.02759314328432083
- `plain_mae/std`: 0.007077259477227926
- `plain_mae/min`: 0.009985200129449368
- `plain_mae/max`: 0.07924821227788925
- `plain_max_abs_error/mean`: 0.5062044262886047
- `plain_max_abs_error/std`: 0.09212836623191833
- `plain_max_abs_error/min`: 0.2666666805744171
- `plain_max_abs_error/max`: 0.8784313797950745
- `plain_mean_signed_error/mean`: -0.0016319070709869266
- `plain_mean_signed_error/std`: 0.003431473160162568
- `plain_mean_signed_error/min`: -0.030713219195604324
- `plain_mean_signed_error/max`: 0.021607162430882454
- `non_plain_pixel_count/mean`: 339014.71875
- `non_plain_pixel_count/std`: 114920.2734375
- `non_plain_pixel_count/min`: 54974.0
- `non_plain_pixel_count/max`: 745180.0
- `non_plain_pixel_fraction/mean`: 0.32330963015556335
- `non_plain_pixel_fraction/std`: 0.10959651321172714
- `non_plain_pixel_fraction/min`: 0.05242729187011719
- `non_plain_pixel_fraction/max`: 0.7106590270996094
- `non_plain_mse/mean`: 0.00546275032684207
- `non_plain_mse/std`: 0.002386081498116255
- `non_plain_mse/min`: 0.001475098542869091
- `non_plain_mse/max`: 0.0223594568669796
- `non_plain_rmse/mean`: 0.07240407168865204
- `non_plain_rmse/std`: 0.014845922589302063
- `non_plain_rmse/min`: 0.038407012820243835
- `non_plain_rmse/max`: 0.1495307832956314
- `non_plain_mae/mean`: 0.05281316861510277
- `non_plain_mae/std`: 0.010569942183792591
- `non_plain_mae/min`: 0.029245752841234207
- `non_plain_mae/max`: 0.10236622393131256
- `non_plain_max_abs_error/mean`: 0.5971580743789673
- `non_plain_max_abs_error/std`: 0.10005410015583038
- `non_plain_max_abs_error/min`: 0.3176470696926117
- `non_plain_max_abs_error/max`: 0.9019607901573181
- `non_plain_mean_signed_error/mean`: 0.0003947805962525308
- `non_plain_mean_signed_error/std`: 0.004416801035404205
- `non_plain_mean_signed_error/min`: -0.01656799204647541
- `non_plain_mean_signed_error/max`: 0.039201948791742325
- `lpips/mean`: 0.13544262945652008
- `lpips/std`: 0.04073378071188927
- `lpips/min`: 0.03768504783511162
- `lpips/max`: 0.388813316822052

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
- `inverted_shapiro_p_value/mean`: 0.3696523606777191
- `inverted_shapiro_p_value/std`: 0.2784488797187805
- `inverted_shapiro_p_value/min`: 2.3965578293427825e-05
- `inverted_shapiro_p_value/max`: 0.9996281862258911
- `gaussian_kl_initial_to_inverted/mean`: 0.0026364459190517664
- `gaussian_kl_initial_to_inverted/std`: 0.0013251376803964376
- `gaussian_kl_initial_to_inverted/min`: 0.00023140013217926025
- `gaussian_kl_initial_to_inverted/max`: 0.009152516722679138
- `gaussian_kl_inverted_to_initial/mean`: 0.002450813539326191
- `gaussian_kl_inverted_to_initial/std`: 0.0011877829674631357
- `gaussian_kl_inverted_to_initial/min`: 0.00022676587104797363
- `gaussian_kl_inverted_to_initial/max`: 0.008088082075119019
- `gaussian_kl_symmetric/mean`: 0.002543630078434944
- `gaussian_kl_symmetric/std`: 0.0012564110802486539
- `gaussian_kl_symmetric/min`: 0.00022908300161361694
- `gaussian_kl_symmetric/max`: 0.008620299398899078

### initial_vs_inverted_noise_by_plain_region

- `plain_abs_error/mean`: 0.5445988774299622
- `plain_abs_error/std`: 0.03345469757914543
- `plain_abs_error/min`: 0.43279215693473816
- `plain_abs_error/max`: 0.7370790243148804
- `non_plain_abs_error/mean`: 0.5030322074890137
- `non_plain_abs_error/std`: 0.027157872915267944
- `non_plain_abs_error/min`: 0.4253152310848236
- `non_plain_abs_error/max`: 0.6177776455879211
- `plain_initial_noise_std/mean`: 0.9687893986701965
- `plain_initial_noise_std/std`: 0.010329981334507465
- `plain_initial_noise_std/min`: 0.932158350944519
- `plain_initial_noise_std/max`: 0.9939183592796326
- `non_plain_initial_noise_std/mean`: 1.021799087524414
- `non_plain_initial_noise_std/std`: 0.008673234842717648
- `non_plain_initial_noise_std/min`: 0.9996885657310486
- `non_plain_initial_noise_std/max`: 1.069328784942627
- `plain_inverted_noise_std/mean`: 0.9236220121383667
- `plain_inverted_noise_std/std`: 0.009548005647957325
- `plain_inverted_noise_std/min`: 0.8538335561752319
- `plain_inverted_noise_std/max`: 0.9494707584381104
- `non_plain_inverted_noise_std/mean`: 0.9740960597991943
- `non_plain_inverted_noise_std/std`: 0.008473852649331093
- `non_plain_inverted_noise_std/min`: 0.9451539516448975
- `non_plain_inverted_noise_std/max`: 1.015419363975525
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
