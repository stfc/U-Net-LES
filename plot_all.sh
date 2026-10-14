#!/bin/bash
dnsres="256 512"

eq="HW"
for res in $dnsres; do
	python plot_unet_validation.py BOUT_runs/${eq}_DNS$res BOUT_runs/${eq}_DNS${res}_LES$((res/2))/results_U-Net/fields/ BOUT_runs/${eq}_DNS${res}_LES$((res/4))/results_U-Net/fields/  BOUT_runs/${eq}_DNS${res}_LES$((res/8))/results_U-Net/fields/ --dpi 300 --output-dir validationJW/${eq}_N${res}
done

eq="mHW"
for res in $dnsres; do
	python plot_unet_validation.py BOUT_runs/${eq}_DNS$res BOUT_runs/${eq}_DNS${res}_LES$((res/2))/results_U-Net/fields/ BOUT_runs/${eq}_DNS${res}_LES$((res/4))/results_U-Net/fields/  BOUT_runs/${eq}_DNS${res}_LES$((res/8))/results_U-Net/fields/ --dpi 300 --output-dir validationJW/${eq}_N${res} --final-time 100
done

python plot_unet_validation.py BOUT_runs/HW_DNS256/ BOUT_runs/HW_DNS256_LES64/ BOUT_runs/HW_DNS256_LES64_CENTORI BOUT_runs/HW_DNS256_LES64_SMAGORINSKY/ BOUT_runs/HW_DNS256_LES64_IMPLICIT --labels fDNS "UNet-LES" CENTORI Smagorinsky "No model" --filter-dns --output-dir validationJW/LES_MODELS -cutoff 64 --ysize 8.5
python plot_unet_validation.py BOUT_runs/HW_DNS256/ BOUT_runs/HW_DNS256_LES64/results_U-Net/fields/ BOUT_runs/HW_DNS256_LES64_CENTORI BOUT_runs/HW_DNS256_LES64_SMAGORINSKY/ BOUT_runs/HW_DNS256_LES64_IMPLICIT --labels DNS "UNet-DNS" CENTORI Smagorinsky "No model" --output-dir validationJW/LES_MODELS_DNS -cutoff 64 --ysize 8.5

