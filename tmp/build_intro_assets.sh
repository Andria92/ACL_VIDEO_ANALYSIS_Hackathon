#!/bin/zsh
set -euo pipefail

DEST="/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1"
AUDIO="$DEST/audio_only"

mkdir -p "$AUDIO"

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
    "$DEST/$output.mp4"
}

extract_audio() {
  local input="$1"
  local output="$2"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$DEST/$input.mp4" -vn \
    -c:a aac -b:a 192k -map_metadata -1 \
    "$AUDIO/$output.m4a"
}

ITV="/Volumes/ACL_DEMO/ACL injuries： An epidemic in Women's Football ｜ ITV Sport [sEGitLFTCHA].mp4"
ARSENAL_TWO="/Volumes/ACL_DEMO/STEP BY STEP ｜ Vivianne Miedema & Beth Mead ｜ Viv runs for the first time! ｜ Episode Two [Y97tn7HBKrg].mp4"
LEAH="/Volumes/ACL_DEMO/How Leah Williamson Bounced Back From A Footballer's WORST NIGHTMARE [SFf9BKDLsec].mp4"
MIDGE="/Volumes/ACL_DEMO/Breaking News： Midge Purce Tears ACL I Attacking Third [I5GD0iXIKEA].mp4"
PBS="/Volumes/ACL_DEMO/Why ACL injuries are more common in female athletes than male counterparts [z9_KKYOWGYQ].mp4"
LAB="/Volumes/ACL_DEMO/YTDown.com_YouTube_Why-are-female-footballers-more-suscepti_Media_EK0NRSXcC6A_001_1080p.mp4"
BETH_RETURN="/Volumes/ACL_DEMO/WSL： Beth Mead returns to Arsenal training following ACL injury [pZy3REWoMeE].mp4"
ALEXIA_RETURN="/Volumes/ACL_DEMO/YTDown.com_YouTube_Alexia-Putellas-starts-training-with-the_Media_d2h_2spypck_001_720p.mp4"

cut_video 0.000 8.600 "$ITV" "01_itv_epidemic_up_to_30_players"
cut_video 218.500 3.100 "$ARSENAL_TWO" "02_viv_one_more_acl"
cut_video 36.800 10.100 "$ITV" "03_chloe_kelly_dark_days_lonely"
cut_video 491.400 2.600 "$LEAH" "04_leah_months_of_hell"
cut_video 79.000 6.400 "$MIDGE" "05_midge_legitimate_epidemic"

extract_audio "01_itv_epidemic_up_to_30_players" "01_itv_epidemic_up_to_30_players"
extract_audio "02_viv_one_more_acl" "02_viv_one_more_acl"
extract_audio "03_chloe_kelly_dark_days_lonely" "03_chloe_kelly_dark_days_lonely"
extract_audio "04_leah_months_of_hell" "04_leah_months_of_hell"
extract_audio "05_midge_legitimate_epidemic" "05_midge_legitimate_epidemic"

cut_video 226.000 8.000 "$PBS" "06_doctor_examining_knee_mri"
cut_video 246.000 8.000 "$PBS" "07_animated_knee_acl_labelled"
cut_video 262.000 8.000 "$PBS" "08_rehabilitation_strength_testing"
cut_video 65.700 12.300 "$LAB" "09_biomechanics_motion_capture_lab"
cut_video 41.000 5.500 "$BETH_RETURN" "10_beth_returning_to_training"
cut_video 8.200 6.600 "$ALEXIA_RETURN" "11_alexia_returning_to_training"

cp "/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/INTRO_CLIP_INDEX.md" "$DEST/CLIP_INDEX.md"

echo "Created intro assets in $DEST"
