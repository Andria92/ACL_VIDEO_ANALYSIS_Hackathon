#!/bin/zsh
set -euo pipefail

INPUT='/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V7_EXACT_EMMA_Q_ANGLES_CUT.mov'
BETH_STRENGTH='/Volumes/ACL_DEMO/demo_assets/beth_viv_step_by_step_rehab_cuts/beth_mead/04_beth_controlled_knee_strength_work.mp4'
BUILD_DIR='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/acl_intro_v8_build'
OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V8_NO_DUPLICATE_ARSENAL_RUNNING.mov'

mkdir -p "$BUILD_DIR"

# Replace only the first copy of the Arsenal running shot (18.338–19.938)
# with Beth's controlled indoor knee-strength work. Timing remains unchanged.
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT" -i "$BETH_STRENGTH" \
  -filter_complex \
  "[0:v]trim=start=0:end=18.338,setpts=PTS-STARTPTS,fps=30[v1]; \
   [1:v]trim=start=0.800:end=2.400,setpts=PTS-STARTPTS,fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p[v2]; \
   [0:v]trim=start=19.938:end=55.467,setpts=PTS-STARTPTS,fps=30[v3]; \
   [v1][v2][v3]concat=n=3:v=1:a=0[vout]" \
  -map '[vout]' -an -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -movflags +faststart \
  "$BUILD_DIR/final_video.mp4"

# Reuse the V7 audio stream without altering any narration or timing.
ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD_DIR/final_video.mp4" -i "$INPUT" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -movflags +faststart \
  -metadata title='ACL Intro — Duplicate Arsenal Running Removed' \
  "$OUTPUT"
