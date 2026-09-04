#!/bin/zsh
set -euo pipefail

BUILD_DIR='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/acl_intro_v3_build'
INPUT='/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V2_ALL_AUDIO_WITH_BETH.mov'
LAB='/Volumes/ACL_DEMO/demo_assets/intro_hooks_and_visuals_round1/09_biomechanics_motion_capture_lab.mp4'
OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V3_EMMA_Q_ANGLES_CUT.mov'

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

# Emma remains on screen through the complete phrase “we've got Q angles.”
# The following two seconds use movement-capture footage while her explanation continues.
make_video "$INPUT" 0.00 36.70 "$BUILD_DIR/01_through_q_angles.mp4"
make_video "$LAB"   0.30  2.00 "$BUILD_DIR/02_q_angle_lab_bridge.mp4"
make_video "$INPUT" 38.70 20.80 "$BUILD_DIR/03_original_tail.mp4"

ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD_DIR/01_through_q_angles.mp4" \
  -i "$BUILD_DIR/02_q_angle_lab_bridge.mp4" \
  -i "$BUILD_DIR/03_original_tail.mp4" \
  -filter_complex '[0:v][1:v][2:v]concat=n=3:v=1:a=0,tpad=stop_mode=clone:stop_duration=0.033334[vout]' \
  -map '[vout]' -an -c:v libx264 -preset veryfast -crf 18 -movflags +faststart \
  "$BUILD_DIR/final_video.mp4"

# Reuse the complete V2 soundtrack unchanged so every injury, scientific and
# mechanical quote stays at its established timing.
ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD_DIR/final_video.mp4" -i "$INPUT" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -movflags +faststart \
  -metadata title='ACL Intro — Emma Cut at Q Angles' \
  "$OUTPUT"
