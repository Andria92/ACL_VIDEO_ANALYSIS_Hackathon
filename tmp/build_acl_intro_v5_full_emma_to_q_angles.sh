#!/bin/zsh
set -euo pipefail

INPUT='/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V3_EMMA_Q_ANGLES_CUT.mov'
EMMA='/Volumes/ACL_DEMO/demo_assets/voice_quote_cuts_all_20/06_multifactorial_no_simple_answer.mp4'
OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V5_FULL_EMMA_TO_Q_ANGLES.mov'

# The Emma excerpt in the soundtrack begins at 35.538. “Q angles” finishes at
# 39.467. Her source video is used throughout that passage so picture and voice
# stay together. Only the short remainder before the next quote at 39.638 is cut.
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT" -i "$EMMA" \
  -filter_complex \
  "[0:v]trim=start=0:end=35.538,setpts=PTS-STARTPTS,fps=30[v1]; \
   [0:a]atrim=start=0:end=35.538,asetpts=PTS-STARTPTS[a1]; \
   [1:v]trim=start=0:end=3.929,setpts=PTS-STARTPTS,fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p[v2]; \
   [0:a]atrim=start=35.538:end=39.467,asetpts=PTS-STARTPTS[a2]; \
   [0:v]trim=start=39.638:end=59.50,setpts=PTS-STARTPTS,fps=30[v3]; \
   [0:a]atrim=start=39.638:end=59.50,asetpts=PTS-STARTPTS[a3]; \
   [v1][a1][v2][a2][v3][a3]concat=n=3:v=1:a=1[vout][aout]" \
  -map '[vout]' -map '[aout]' \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 256k -movflags +faststart \
  -metadata title='ACL Intro — Full Emma Statement Through Q Angles' \
  "$OUTPUT"
