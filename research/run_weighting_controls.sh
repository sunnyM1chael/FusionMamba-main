#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/FusionMamba-SACAFM-14e9a93
commit="$1"
for mode in equal learned; do
    /root/miniconda3/bin/python -u research/launch_msrs.py \
        --commit "$commit" --weighting_mode "$mode" --wait \
        --output "/root/autodl-fs/research_protocol/v1/runs/msrs_sacafm_${mode}_seed42_controls_v1"
done
