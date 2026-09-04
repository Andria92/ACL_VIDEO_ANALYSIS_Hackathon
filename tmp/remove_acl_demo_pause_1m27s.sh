#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
BUILD="$ROOT/tmp/judges_cut_v5_features"
SOURCE="$BUILD/ACL_DEMO_FINAL_V7_NO_3M02S_PAUSE.mov"
FINAL="$BUILD/ACL_DEMO_FINAL_V8_NO_1M27S_PAUSE.mov"

# Close the two adjacent low-energy gaps around 01:27 without touching speech.
ffmpeg -hide_banner -loglevel error -y -i "$SOURCE" -filter_complex \
  "[0:v]trim=start=0:end=86.05,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=86.05,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=86.50:end=86.925,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=86.50:end=86.925,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=87.075:end=228.80,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=87.075:end=228.80,asetpts=PTS-STARTPTS[a2];
   [v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Final Smooth Feature-Complete Judges Cut" \
  "$FINAL"

echo "$FINAL"
