#!/bin/zsh
set -euo pipefail

EP1_SRC='/Volumes/ACL_DEMO/STEP BY STEP ｜ Vivianne Miedema & Beth Mead ｜ Football Was My Happy Place ｜ Episode One [vxHrH2nCqR8].mp4'
EP2_SRC='/Volumes/ACL_DEMO/STEP BY STEP ｜ Vivianne Miedema & Beth Mead ｜ Viv runs for the first time! ｜ Episode Two [Y97tn7HBKrg].mp4'
OUT_DIR='/Volumes/ACL_DEMO/demo_assets/beth_viv_step_by_step_rehab_cuts'

mkdir -p "$OUT_DIR/beth_mead" "$OUT_DIR/vivianne_miedema"

cut_clip() {
  local source_path="$1"
  local start_time="$2"
  local duration="$3"
  local output_path="$4"

  ffmpeg -hide_banner -loglevel error -y \
    -ss "$start_time" -i "$source_path" -t "$duration" \
    -map 0:v:0 -map '0:a:0?' \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
    -c:a aac -b:a 192k \
    -movflags +faststart -sn -dn -map_metadata -1 \
    "$output_path"
}

# Beth Mead — Episode One
cut_clip "$EP1_SRC" 1010.40 9.20  "$OUT_DIR/beth_mead/01_beth_118_days_since_surgery_gym.mp4"
cut_clip "$EP1_SRC" 1019.60 11.24 "$OUT_DIR/beth_mead/02_beth_indoor_running_progression.mp4"
cut_clip "$EP1_SRC" 1036.32 10.32 "$OUT_DIR/beth_mead/03_beth_relearning_running_and_strength.mp4"
cut_clip "$EP1_SRC" 1061.80 8.28  "$OUT_DIR/beth_mead/04_beth_controlled_knee_strength_work.mp4"
cut_clip "$EP1_SRC" 1097.24 14.32 "$OUT_DIR/beth_mead/05_beth_single_leg_hops_and_landings.mp4"
cut_clip "$EP1_SRC" 1315.60 8.68  "$OUT_DIR/beth_mead/06_beth_first_pitch_session.mp4"
cut_clip "$EP1_SRC" 1324.28 8.60  "$OUT_DIR/beth_mead/07_beth_running_on_grass.mp4"
cut_clip "$EP1_SRC" 1332.88 10.24 "$OUT_DIR/beth_mead/08_beth_ball_work_with_team.mp4"

# Vivianne Miedema — Episode Two
cut_clip "$EP2_SRC" 355.28  9.04  "$OUT_DIR/vivianne_miedema/01_viv_resistance_band_knee_work.mp4"
cut_clip "$EP2_SRC" 393.12  14.48 "$OUT_DIR/vivianne_miedema/02_viv_single_leg_hop_drill.mp4"
cut_clip "$EP2_SRC" 437.60  11.76 "$OUT_DIR/vivianne_miedema/03_viv_antigravity_treadmill_setup.mp4"
cut_clip "$EP2_SRC" 449.36  10.00 "$OUT_DIR/vivianne_miedema/04_viv_reduced_bodyweight_calibration.mp4"
cut_clip "$EP2_SRC" 493.08  14.88 "$OUT_DIR/vivianne_miedema/05_viv_first_run_at_reduced_bodyweight.mp4"
cut_clip "$EP2_SRC" 507.96  15.88 "$OUT_DIR/vivianne_miedema/06_viv_running_with_support.mp4"
cut_clip "$EP2_SRC" 530.04  10.00 "$OUT_DIR/vivianne_miedema/07_viv_final_running_effort.mp4"
cut_clip "$EP2_SRC" 551.60  13.40 "$OUT_DIR/vivianne_miedema/08_viv_post_run_relief_and_celebration.mp4"

cp 'tmp/BETH_VIV_REHAB_CLIP_INDEX.md' "$OUT_DIR/CLIP_INDEX.md"

