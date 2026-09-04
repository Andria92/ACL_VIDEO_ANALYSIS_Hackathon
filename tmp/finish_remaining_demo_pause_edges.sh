#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
BUILD="$ROOT/tmp/judges_cut_under_4min_v2"
SOURCE="$BUILD/ACL_DEMO_UNDER_4_MIN_V3_NO_AWKWARD_PAUSES.mov"
FINAL="$BUILD/ACL_DEMO_UNDER_4_MIN_V4_FINAL_SMOOTH.mov"

# Close the two small residual gaps left at the edges of the requested repairs.
ffmpeg -hide_banner -loglevel error -y -i "$SOURCE" -filter_complex \
  "[0:v]trim=start=0:end=89.56,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=89.56,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=89.84:end=199.52,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=89.84:end=199.52,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=199.92:end=229.333333,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=199.92:end=229.333333,asetpts=PTS-STARTPTS[a2];
   [v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Final Smooth Four-Minute Judges Cut" \
  "$FINAL"

echo "$FINAL"
