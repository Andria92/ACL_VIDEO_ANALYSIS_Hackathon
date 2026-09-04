#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
BUILD="$ROOT/tmp/judges_cut_v5_features"
SOURCE="$BUILD/ACL_DEMO_FINAL_V5_SMOOTH_FEATURE_COMPLETE.mov"
FINAL="$BUILD/ACL_DEMO_FINAL_V7_NO_3M02S_PAUSE.mov"

# Remove the long hesitation between “in no capacity” and “we want to justify”.
ffmpeg -hide_banner -loglevel error -y -i "$SOURCE" -filter_complex \
  "[0:v]trim=start=0:end=181.08,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=181.08,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=182.68:end=230.40,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=182.68:end=230.40,asetpts=PTS-STARTPTS[a1];
   [v0][a0][v1][a1]concat=n=2:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Final Feature-Complete Judges Cut" \
  "$FINAL"

echo "$FINAL"
