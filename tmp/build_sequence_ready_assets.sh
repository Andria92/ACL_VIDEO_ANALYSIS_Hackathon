#!/bin/zsh
set -euo pipefail

DEST="/Volumes/ACL_DEMO/demo_assets/intro_sequence_ready_round1"
AUDIO="$DEST/audio_only"
mkdir -p "$AUDIO"

copy_video() {
  cp "$1" "$DEST/$2"
}

cut_video() {
  local start="$1"
  local duration="$2"
  local source="$3"
  local output="$4"

  ffmpeg -hide_banner -loglevel error -y \
    -ss "$start" -i "$source" -t "$duration" \
    -map 0:v:0 -map '0:a:0?' \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
    -c:a aac -b:a 192k -movflags +faststart \
    -sn -dn -map_metadata -1 \
    "$DEST/$output"
}

extract_audio() {
  ffmpeg -hide_banner -loglevel error -y \
    -i "$DEST/$1" -vn -c:a aac -b:a 192k -map_metadata -1 \
    "$AUDIO/$2"
}

copy_video "/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/02_viv_one_more_acl.mp4" "01_viv_one_more_acl.mp4"
copy_video "/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/05_ellie_carpenter_down_in_pain.mp4" "02_ellie_carpenter_down_in_pain.mp4"

cut_video 324.800 7.000 \
  "/Volumes/ACL_DEMO/STEP BY STEP ｜ Vivianne Miedema & Beth Mead ｜ Football Was My Happy Place ｜ Episode One [vxHrH2nCqR8].mp4" \
  "03_beth_movement_done_a_thousand_times.mp4"

copy_video "/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/04_vivianne_miedema_stretcher.mp4" "04_vivianne_miedema_on_stretcher.mp4"
copy_video "/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/10_beth_returning_to_training.mp4" "05_beth_returning_to_training.mp4"
copy_video "/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/06_doctor_examining_knee_mri.mp4" "06_doctor_examining_knee_mri.mp4"
copy_video "/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/07_animated_knee_acl_labelled.mp4" "07_animated_knee_acl_labelled.mp4"
copy_video "/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/09_biomechanics_motion_capture_lab.mp4" "08_motion_capture_laboratory.mp4"

cut_video 25.650 2.600 \
  "/Volumes/ACL_DEMO/YTDown.com_YouTube_Emma-Hayes-calls-for-more-research-to-be_Media_N2bCmQndTdw_001_1080p.mp4" \
  "09_emma_multifactorial_no_simple_answer.mp4"

extract_audio "01_viv_one_more_acl.mp4" "01_viv_one_more_acl.m4a"
extract_audio "03_beth_movement_done_a_thousand_times.mp4" "03_beth_movement_done_a_thousand_times.m4a"
extract_audio "09_emma_multifactorial_no_simple_answer.mp4" "09_emma_multifactorial_no_simple_answer.m4a"

cp "/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/SEQUENCE_READY_CLIP_INDEX.md" "$DEST/CLIP_INDEX.md"

echo "Created sequence-ready assets in $DEST"
