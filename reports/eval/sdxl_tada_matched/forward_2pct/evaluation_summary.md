# Evaluation Summary

- Input directory: `/tmp/sdxl-tada-metrics-84444/tmp/sdxl_tada_matched_metric_views/forward_2pct`
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

- `mean`: 0.01644197478890419
- `std`: 0.929076075553894
- `min`: -4.1796875
- `p01`: -2.1054883003234863
- `p05`: -1.5009765625
- `p50`: 0.013195037841796875
- `p95`: 1.5478515625
- `p99`: 2.16015625
- `max`: 4.39453125
- `skew`: 0.01748482696712017
- `excess_kurtosis`: -0.0956430435180664
- `abs_mean_error_from_normal`: 0.01644197478890419
- `abs_std_error_from_normal`: 0.0709262490272522
- `normal_kl_from_standard`: 0.005291357636451721

### initial_vs_inverted_noise

- `mse`: 0.13410095870494843
- `rmse`: 0.36619797348976135
- `mae`: 0.2816331684589386
- `l2`: 3614.687744140625
- `cosine`: 0.9514164328575134
- `relative_rmse_to_reference_std`: 0.36619362235069275
- `candidate_to_reference_std_ratio`: 0.9317490458488464
- `mean_shift`: 0.015128236263990402

### inverted_noise_patch_topk_correlation

- `mean`: 0.14873583614826202
- `std`: 0.016883576288819313

### latent_normality

- `corr`: 0.14873583614826202
- `corr_std`: 0.016883576288819313
- `kl`: 0.0053731948137283325
- `reverse_kl`: 0.004884347319602966
- `symmetric_kl`: 0.005128771066665649
- `per_location_kl`: 0.006978671997785568
- `per_location_reverse_kl`: 0.006108437664806843
- `per_location_symmetric_kl`: 0.0065435548312962055

### initial_vs_inverted_noise_per_channel

- `channel_0/mse`: 0.16477996110916138
- `channel_0/rmse`: 0.4059309661388397
- `channel_0/mae`: 0.31078729033470154
- `channel_0/l2`: 2024.3095703125
- `channel_0/cosine`: 0.9166750311851501
- `channel_0/relative_rmse_to_reference_std`: 0.4058758616447449
- `channel_0/candidate_to_reference_std_ratio`: 0.9030246138572693
- `channel_0/mean_shift`: 0.013772777281701565
- `channel_1/mse`: 0.12076831609010696
- `channel_1/rmse`: 0.34751737117767334
- `channel_1/mae`: 0.26972487568855286
- `channel_1/l2`: 1732.65478515625
- `channel_1/cosine`: 0.9404085278511047
- `channel_1/relative_rmse_to_reference_std`: 0.34752100706100464
- `channel_1/candidate_to_reference_std_ratio`: 0.9436290264129639
- `channel_1/mean_shift`: 0.00268277944996953
- `channel_2/mse`: 0.10016493499279022
- `channel_2/rmse`: 0.31648844480514526
- `channel_2/mae`: 0.24377679824829102
- `channel_2/l2`: 1577.903076171875
- `channel_2/cosine`: 0.9515753388404846
- `channel_2/relative_rmse_to_reference_std`: 0.3165196180343628
- `channel_2/candidate_to_reference_std_ratio`: 0.9707802534103394
- `channel_2/mean_shift`: 0.014809105545282364
- `channel_3/mse`: 0.15069067478179932
- `channel_3/rmse`: 0.3881889581680298
- `channel_3/mae`: 0.3022436797618866
- `channel_3/l2`: 1935.7864990234375
- `channel_3/cosine`: 0.9243453741073608
- `channel_3/relative_rmse_to_reference_std`: 0.38818082213401794
- `channel_3/candidate_to_reference_std_ratio`: 0.9077391624450684
- `channel_3/mean_shift`: 0.029248284175992012

### inversion_error_stats

- `signed_mean`: 0.015128237195312977
- `signed_std`: 0.3658853769302368
- `abs_mean`: 0.281595379114151
- `abs_std`: 0.23339243233203888
- `abs_p50`: 0.2249603271484375
- `abs_p90`: 0.5966796875
- `abs_p95`: 0.7348051071166992
- `abs_p99`: 1.046875
- `abs_max`: 2.22607421875

### inversion_error_patch_topk_correlation

- `mean`: 0.1972484290599823
- `std`: 0.01324532087892294

### inversion_error_structure

- `signed_error_vs_final_latent_cosine`: 0.008496797643601894
- `abs_error_vs_abs_final_latent_cosine`: 0.6219521760940552

### final_vs_reconstructed_image

- `mse/mean`: 0.0020154330413788557
- `mse/std`: 0.001367409946396947
- `mse/min`: 0.00024227560788858682
- `mse/max`: 0.013281879015266895
- `rmse/mean`: 0.04298194870352745
- `rmse/std`: 0.012960923835635185
- `rmse/min`: 0.015565205365419388
- `rmse/max`: 0.11524703353643417
- `mae/mean`: 0.02858298271894455
- `mae/std`: 0.009053651243448257
- `mae/min`: 0.011045433580875397
- `mae/max`: 0.08961251378059387
- `psnr_db/mean`: 27.692373275756836
- `psnr_db/std`: 2.4659175872802734
- `psnr_db/min`: 18.767404556274414
- `psnr_db/max`: 36.15690231323242
- `ssim/mean`: 0.7864910960197449
- `ssim/std`: 0.06685923039913177
- `ssim/min`: 0.3973065912723541
- `ssim/max`: 0.9340652227401733
- `max_abs_error/mean`: 0.5682075023651123
- `max_abs_error/std`: 0.10404852777719498
- `max_abs_error/min`: 0.23529410362243652
- `max_abs_error/max`: 0.9607843160629272
- `mean_signed_error/mean`: -0.0013396897120401263
- `mean_signed_error/std`: 0.0031414490658789873
- `mean_signed_error/min`: -0.02791498601436615
- `mean_signed_error/max`: 0.019565848633646965
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
- `plain_mse/mean`: 0.001187321962788701
- `plain_mse/std`: 0.0010040898341685534
- `plain_mse/min`: 0.0001824879291234538
- `plain_mse/max`: 0.01176920160651207
- `plain_rmse/mean`: 0.032563790678977966
- `plain_rmse/std`: 0.011265929788351059
- `plain_rmse/min`: 0.013508809730410576
- `plain_rmse/max`: 0.10848595201969147
- `plain_mae/mean`: 0.02214144356548786
- `plain_mae/std`: 0.007754568476229906
- `plain_mae/min`: 0.007718851324170828
- `plain_mae/max`: 0.07908578217029572
- `plain_max_abs_error/mean`: 0.46517738699913025
- `plain_max_abs_error/std`: 0.1027083545923233
- `plain_max_abs_error/min`: 0.20784315466880798
- `plain_max_abs_error/max`: 0.8549019694328308
- `plain_mean_signed_error/mean`: -0.0019158850191161036
- `plain_mean_signed_error/std`: 0.004273546393960714
- `plain_mean_signed_error/min`: -0.05649782344698906
- `plain_mean_signed_error/max`: 0.02270856499671936
- `non_plain_pixel_count/mean`: 339014.71875
- `non_plain_pixel_count/std`: 114920.2734375
- `non_plain_pixel_count/min`: 54974.0
- `non_plain_pixel_count/max`: 745180.0
- `non_plain_pixel_fraction/mean`: 0.32330963015556335
- `non_plain_pixel_fraction/std`: 0.10959651321172714
- `non_plain_pixel_fraction/min`: 0.05242729187011719
- `non_plain_pixel_fraction/max`: 0.7106590270996094
- `non_plain_mse/mean`: 0.0037461123429238796
- `non_plain_mse/std`: 0.0021627633832395077
- `non_plain_mse/min`: 0.0008390596485696733
- `non_plain_mse/max`: 0.01767372339963913
- `non_plain_rmse/mean`: 0.059297826141119
- `non_plain_rmse/std`: 0.01516181044280529
- `non_plain_rmse/min`: 0.028966525569558144
- `non_plain_rmse/max`: 0.1329425573348999
- `non_plain_mae/mean`: 0.04227026551961899
- `non_plain_mae/std`: 0.010677367448806763
- `non_plain_mae/min`: 0.021198254078626633
- `non_plain_mae/max`: 0.10095024853944778
- `non_plain_max_abs_error/mean`: 0.5667001008987427
- `non_plain_max_abs_error/std`: 0.10441344976425171
- `non_plain_max_abs_error/min`: 0.23529410362243652
- `non_plain_max_abs_error/max`: 0.9607843160629272
- `non_plain_mean_signed_error/mean`: -2.705307815631386e-05
- `non_plain_mean_signed_error/std`: 0.004072518553584814
- `non_plain_mean_signed_error/min`: -0.013717086054384708
- `non_plain_mean_signed_error/max`: 0.03724321722984314
- `lpips/mean`: 0.10162593424320221
- `lpips/std`: 0.04672714322805405
- `lpips/min`: 0.023946944624185562
- `lpips/max`: 0.609410285949707

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
- `inverted_shapiro_p_value/mean`: 0.31785717606544495
- `inverted_shapiro_p_value/std`: 0.272918701171875
- `inverted_shapiro_p_value/min`: 4.785989585798234e-05
- `inverted_shapiro_p_value/max`: 0.9896875023841858
- `gaussian_kl_initial_to_inverted/mean`: 0.005813620053231716
- `gaussian_kl_initial_to_inverted/std`: 0.003120318753644824
- `gaussian_kl_initial_to_inverted/min`: 0.0010473895817995071
- `gaussian_kl_initial_to_inverted/max`: 0.02475789189338684
- `gaussian_kl_inverted_to_initial/mean`: 0.0052167014218866825
- `gaussian_kl_inverted_to_initial/std`: 0.002636832185089588
- `gaussian_kl_inverted_to_initial/min`: 0.0010036937892436981
- `gaussian_kl_inverted_to_initial/max`: 0.02028198540210724
- `gaussian_kl_symmetric/mean`: 0.005515160039067268
- `gaussian_kl_symmetric/std`: 0.0028782906010746956
- `gaussian_kl_symmetric/min`: 0.0010255416855216026
- `gaussian_kl_symmetric/max`: 0.02251993864774704

### initial_vs_inverted_noise_by_plain_region

- `plain_abs_error/mean`: 0.3031919300556183
- `plain_abs_error/std`: 0.03015168383717537
- `plain_abs_error/min`: 0.20410500466823578
- `plain_abs_error/max`: 0.4787694215774536
- `non_plain_abs_error/mean`: 0.2655455470085144
- `non_plain_abs_error/std`: 0.022476647049188614
- `non_plain_abs_error/min`: 0.20125453174114227
- `non_plain_abs_error/max`: 0.3630324900150299
- `plain_initial_noise_std/mean`: 0.9687893986701965
- `plain_initial_noise_std/std`: 0.010329981334507465
- `plain_initial_noise_std/min`: 0.932158350944519
- `plain_initial_noise_std/max`: 0.9939183592796326
- `non_plain_initial_noise_std/mean`: 1.021799087524414
- `non_plain_initial_noise_std/std`: 0.008673234842717648
- `non_plain_initial_noise_std/min`: 0.9996885657310486
- `non_plain_initial_noise_std/max`: 1.069328784942627
- `plain_inverted_noise_std/mean`: 0.8869733214378357
- `plain_inverted_noise_std/std`: 0.019499482586979866
- `plain_inverted_noise_std/min`: 0.709938645362854
- `plain_inverted_noise_std/max`: 0.9333882331848145
- `non_plain_inverted_noise_std/mean`: 0.9625563621520996
- `non_plain_inverted_noise_std/std`: 0.013580940663814545
- `non_plain_inverted_noise_std/min`: 0.9032817482948303
- `non_plain_inverted_noise_std/max`: 1.0146417617797852
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
