#!/bin/zsh
set -euo pipefail

BUILD_DIR='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/acl_intro_v6_build'
INPUT='/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V5_FULL_EMMA_TO_Q_ANGLES.mov'
LEAH='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/01_leah_williamson_down_commentary.mp4'
PRESS='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/02_christen_press_helped_off.mp4'
ELLIE='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/05_ellie_carpenter_down_in_pain.mp4'
MIEDEMA='/Volumes/ACL_DEMO/demo_assets/acl_injury_cuts_round1/04_vivianne_miedema_stretcher.mp4'
BETH_INJURY='/Volumes/ACL_DEMO/YTDown.com_YouTube_Beth-Mead-nasty-knee-injury-ACL-Arsenal-_Media_Kxz1vhDq94Y_001_1080p.mp4'
BETH_REHAB='/Volumes/ACL_DEMO/demo_assets/intro_sequence_ready_round1/05_beth_returning_to_training.mp4'
BETH_QUOTE='/Volumes/ACL_DEMO/demo_assets/intro_sequence_ready_round1/audio_only/03_beth_movement_done_a_thousand_times.m4a'
OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V6_SHORTER_INJURIES_NO_9_MONTH_QUOTE.mov'

mkdir -p "$BUILD_DIR"

make_video() {
  local source_path="$1"
  local start_time="$2"
  local duration="$3"
  local output_path="$4"

  ffmpeg -hide_banner -loglevel error -y \
    -ss "$start_time" -i "$source_path" -t "$duration" \
    -an -vf 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=30,setsar=1,format=yuv420p' \
    -c:v libx264 -preset veryfast -crf 18 -movflags +faststart \
    "$output_path"
}

# A faster injury montage: every player remains identifiable and the defining
# moment stays visible, but setup and lingering frames are removed.
make_video "$LEAH"        0.00 4.500 "$BUILD_DIR/01_leah.mp4"
make_video "$PRESS"       0.70 3.500 "$BUILD_DIR/02_press.mp4"
make_video "$ELLIE"       0.60 2.000 "$BUILD_DIR/03_ellie.mp4"
make_video "$MIEDEMA"     0.00 2.938 "$BUILD_DIR/04_miedema.mp4"
make_video "$BETH_INJURY" 4.00 5.400 "$BUILD_DIR/05_beth_injury.mp4"
make_video "$BETH_REHAB"  0.40 1.600 "$BUILD_DIR/06_beth_rehab_bridge.mp4"
make_video "$INPUT"      22.505 36.861 "$BUILD_DIR/07_original_science_tail.mp4"

ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD_DIR/01_leah.mp4" \
  -i "$BUILD_DIR/02_press.mp4" \
  -i "$BUILD_DIR/03_ellie.mp4" \
  -i "$BUILD_DIR/04_miedema.mp4" \
  -i "$BUILD_DIR/05_beth_injury.mp4" \
  -i "$BUILD_DIR/06_beth_rehab_bridge.mp4" \
  -i "$BUILD_DIR/07_original_science_tail.mp4" \
  -filter_complex '[0:v][1:v][2:v][3:v][4:v][5:v][6:v]concat=n=7:v=1:a=0,tpad=stop_mode=clone:stop_duration=0.033334[vout]' \
  -map '[vout]' -an -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -movflags +faststart \
  "$BUILD_DIR/final_video.mp4"

# Keep the epidemic opening, replace the complete unwanted “nine...” section
# with Beth's clean seven-second quote, then resume the user's science soundtrack.
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT" -i "$BETH_QUOTE" \
  -filter_complex \
  "[0:a]atrim=start=0:end=12.938,asetpts=PTS-STARTPTS[a1]; \
   [1:a]atrim=start=0:end=7.000,asetpts=PTS-STARTPTS,loudnorm=I=-20:TP=-3:LRA=7,afade=t=in:d=0.12,afade=t=out:st=6.85:d=0.15,aresample=48000[a2]; \
   [0:a]atrim=start=22.505:end=59.394,asetpts=PTS-STARTPTS[a3]; \
   [a1][a2][a3]concat=n=3:v=0:a=1,alimiter=limit=0.95,aresample=48000[aout]" \
  -map '[aout]' -c:a pcm_s16le "$BUILD_DIR/final_audio.wav"

ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD_DIR/final_video.mp4" -i "$BUILD_DIR/final_audio.wav" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 256k -shortest -movflags +faststart \
  -metadata title='ACL Intro — Shorter Injuries, Nine-Month Quote Removed' \
  "$OUTPUT"
