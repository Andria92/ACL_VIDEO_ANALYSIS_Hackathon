#!/bin/zsh
set -euo pipefail

BUILD_DIR='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/acl_intro_fixed_build'
LOCAL_OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_WITH_BETH.mov'

LEAH='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/01_leah_williamson_down_commentary.mp4'
PRESS='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/02_christen_press_helped_off.mp4'
ELLIE='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/05_ellie_carpenter_down_in_pain.mp4'
MIEDEMA='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/04_vivianne_miedema_stretcher.mp4'
BETH_INJURY='/Volumes/ACL_DEMO/YTDown.com_YouTube_Beth-Mead-nasty-knee-injury-ACL-Arsenal-_Media_Kxz1vhDq94Y_001_1080p.mp4'
CHLOE='/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/03_chloe_kelly_dark_days_lonely.mp4'
ALEXIA='/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/11_alexia_returning_to_training.mp4'
BETH_GRASS='/Volumes/ACL_DEMO/demo_assets/beth_viv_step_by_step_rehab_cuts/beth_mead/07_beth_running_on_grass.mp4'
BETH_HOPS='/Volumes/ACL_DEMO/demo_assets/beth_viv_step_by_step_rehab_cuts/beth_mead/05_beth_single_leg_hops_and_landings.mp4'
VIV_RUN='/Volumes/ACL_DEMO/demo_assets/beth_viv_step_by_step_rehab_cuts/vivianne_miedema/07_viv_final_running_effort.mp4'

ITV_AUDIO='/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/audio_only/01_itv_epidemic_up_to_30_players.m4a'
VIV_AUDIO='/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/audio_only/02_viv_one_more_acl_fixed.m4a'
BETH_AUDIO='/Volumes/ACL_DEMO/demo_assets/intro_sequence_ready_round1/audio_only/03_beth_movement_done_a_thousand_times.m4a'
CHLOE_AUDIO='/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/audio_only/03_chloe_kelly_dark_days_lonely.m4a'
EMMA_AUDIO='/Volumes/ACL_DEMO/demo_assets/intro_sequence_ready_round1/audio_only/09_emma_multifactorial_no_simple_answer.m4a'

mkdir -p "$BUILD_DIR"

make_video() {
  local source_path="$1"
  local start_time="$2"
  local duration="$3"
  local output_path="$4"
  local extra_filter="${5:-null}"

  ffmpeg -hide_banner -loglevel error -y \
    -ss "$start_time" -i "$source_path" -t "$duration" \
    -an -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=30,setsar=1,$extra_filter,format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 18 -movflags +faststart \
    "$output_path"
}

make_video "$LEAH"       0.00 8.10  "$BUILD_DIR/01_leah.mp4"
make_video "$PRESS"      0.20 4.40  "$BUILD_DIR/02_press.mp4"
make_video "$ELLIE"      0.50 2.50  "$BUILD_DIR/03_ellie.mp4"
make_video "$MIEDEMA"    0.00 5.55  "$BUILD_DIR/04_miedema.mp4" 'tpad=stop_mode=clone:stop_duration=0.55'
make_video "$BETH_INJURY" 3.00 7.00 "$BUILD_DIR/05_beth_injury.mp4"

# Chloe appears briefly to establish the speaker; her voice then continues over rehab.
make_video "$CHLOE"      4.80 1.60  "$BUILD_DIR/06_chloe.mp4"
make_video "$ALEXIA"     2.80 2.80  "$BUILD_DIR/07_alexia.mp4"
make_video "$BETH_GRASS" 0.00 2.20  "$BUILD_DIR/08_beth_grass.mp4"
make_video "$BETH_HOPS"  3.50 1.80  "$BUILD_DIR/09_beth_hops.mp4"
make_video "$VIV_RUN"    0.00 1.72  "$BUILD_DIR/10_viv_run_chloe_audio.mp4"
make_video "$VIV_RUN"    1.72 2.60  "$BUILD_DIR/11_viv_run_emma_audio.mp4" 'fade=t=out:st=2.20:d=0.40'

ffmpeg -hide_banner -loglevel error -y \
  -i "$LEAH" \
  -i "$ITV_AUDIO" \
  -i "$VIV_AUDIO" \
  -i "$BETH_AUDIO" \
  -i "$CHLOE_AUDIO" \
  -i "$EMMA_AUDIO" \
  -filter_complex \
  "[0:a]atrim=duration=8.10,asetpts=N/SR/TB,loudnorm=I=-20:TP=-3:LRA=7,afade=t=out:st=7.95:d=0.15[a0]; \
   [1:a]atrim=duration=8.60,asetpts=N/SR/TB,loudnorm=I=-20:TP=-3:LRA=7,afade=t=in:d=0.15,afade=t=out:st=8.45:d=0.15[a1]; \
   [2:a]atrim=duration=3.85,asetpts=N/SR/TB,loudnorm=I=-20:TP=-3:LRA=7,afade=t=in:d=0.15,afade=t=out:st=3.70:d=0.15[a2]; \
   [3:a]atrim=duration=7.00,asetpts=N/SR/TB,loudnorm=I=-20:TP=-3:LRA=7,afade=t=in:d=0.15,afade=t=out:st=6.85:d=0.15[a3]; \
   [4:a]atrim=duration=10.12,asetpts=N/SR/TB,loudnorm=I=-20:TP=-3:LRA=7,afade=t=in:d=0.15,afade=t=out:st=9.97:d=0.15[a4]; \
   [5:a]atrim=duration=2.60,asetpts=N/SR/TB,loudnorm=I=-20:TP=-3:LRA=7,afade=t=in:d=0.15,afade=t=out:st=2.20:d=0.40[a5]; \
   [a0][a1][a2][a3][a4][a5]concat=n=6:v=0:a=1,aresample=48000,alimiter=limit=0.95[aout]" \
  -map '[aout]' -c:a pcm_s16le "$BUILD_DIR/final_audio.wav"

ffmpeg -hide_banner -loglevel error -y \
  -f concat -safe 0 -i '/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/acl_intro_fixed_concat.txt' \
  -an -c copy "$BUILD_DIR/final_video.mp4"

ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD_DIR/final_video.mp4" -i "$BUILD_DIR/final_audio.wav" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 256k \
  -movflags +faststart -shortest -metadata title='ACL Intro — Revised with Beth Mead' \
  "$LOCAL_OUTPUT"

