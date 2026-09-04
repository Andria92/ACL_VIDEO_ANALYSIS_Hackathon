#!/bin/zsh
set -euo pipefail

BUILD_DIR='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/acl_intro_v2_build'
LOCAL_OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V2_ALL_AUDIO_WITH_BETH.mov'

ORIGINAL='/Users/andriagryffinpro/Desktop/ACL_INTRO.mov'
LEAH='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/01_leah_williamson_down_commentary.mp4'
PRESS='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/02_christen_press_helped_off.mp4'
ELLIE='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/05_ellie_carpenter_down_in_pain.mp4'
MIEDEMA='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/04_vivianne_miedema_stretcher.mp4'
BETH_INJURY='/Volumes/ACL_DEMO/YTDown.com_YouTube_Beth-Mead-nasty-knee-injury-ACL-Arsenal-_Media_Kxz1vhDq94Y_001_1080p.mp4'
BETH_AUDIO='/Volumes/ACL_DEMO/demo_assets/intro_sequence_ready_round1/audio_only/03_beth_movement_done_a_thousand_times.m4a'

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

# End Leah at the final pitch-side frame; the following coach reaction is excluded.
make_video "$LEAH"        0.00  5.70  "$BUILD_DIR/01_leah_no_coaches.mp4"
make_video "$PRESS"       0.20  4.50  "$BUILD_DIR/02_press.mp4"
make_video "$ELLIE"       0.50  2.50  "$BUILD_DIR/03_ellie.mp4"
make_video "$MIEDEMA"     0.00  2.80  "$BUILD_DIR/04_miedema.mp4"

# Beth's movement and fall are timed so “within milliseconds” lands on contact.
make_video "$BETH_INJURY" 4.00  7.00  "$BUILD_DIR/05_beth_injury.mp4"

# Resume the user's original recovery/science visual edit. The short tail keeps
# the rebuilt picture exactly aligned to the untouched 59.53-second soundtrack.
make_video "$ORIGINAL"   23.40 37.033333 "$BUILD_DIR/06_original_rehab_and_science.mp4" \
  'tpad=stop_mode=clone:stop_duration=0.90,fade=t=out:st=36.633333:d=0.40'

ffmpeg -hide_banner -loglevel error -y \
  -f concat -safe 0 -i '/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/acl_intro_v2_concat.txt' \
  -an -c copy "$BUILD_DIR/final_video.mp4"

# Preserve the complete soundtrack from ACL_INTRO.mov. Only the non-explanatory
# injury ambience from 15.50–22.50 is ducked underneath Beth's own seven-second
# quotation; every scientific/mechanical excerpt after 22.50 remains unchanged.
ffmpeg -hide_banner -loglevel error -y \
  -i "$ORIGINAL" -i "$BETH_AUDIO" \
  -filter_complex \
  "[0:a]atrim=duration=59.533333,asetpts=N/SR/TB,volume='if(between(t,15.50,22.49),0.14,1)':eval=frame[original]; \
   [1:a]atrim=duration=7.00,asetpts=N/SR/TB,loudnorm=I=-20:TP=-3:LRA=7,afade=t=in:d=0.12,afade=t=out:st=6.85:d=0.15,adelay=15500|15500[beth]; \
   [original][beth]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95,aresample=48000[aout]" \
  -map '[aout]' -c:a pcm_s16le "$BUILD_DIR/final_audio.wav"

ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD_DIR/final_video.mp4" -i "$BUILD_DIR/final_audio.wav" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 256k \
  -movflags +faststart -shortest \
  -metadata title='ACL Intro — Original Audio Preserved, Beth Mead Added' \
  "$LOCAL_OUTPUT"
