#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
BUILD="$ROOT/tmp/judges_cut_under_4min_v2"
SOURCE="$BUILD/ACL_DEMO_UNDER_4_MIN_V2_REFINED_WITH_PHASE_STORY.mov"
FINAL="$BUILD/ACL_DEMO_UNDER_4_MIN_V3_NO_AWKWARD_PAUSES.mov"

# Tighten the hesitation around 01:28 and the dead-air cluster around 03:20.
# Every cut begins and ends inside low-energy audio, away from spoken words.
ffmpeg -hide_banner -loglevel error -y -i "$SOURCE" -filter_complex \
  "[0:v]trim=start=0:end=88.12,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=88.12,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=88.52:end=197.85,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=88.52:end=197.85,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=199.15:end=201.00,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=199.15:end=201.00,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=201.45:end=202.42,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=201.45:end=202.42,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=203.25:end=204.62,setpts=PTS-STARTPTS[v4];
   [0:a]atrim=start=203.25:end=204.62,asetpts=PTS-STARTPTS[a4];
   [0:v]trim=start=205.20:end=232.866667,setpts=PTS-STARTPTS[v5];
   [0:a]atrim=start=205.20:end=232.866667,asetpts=PTS-STARTPTS[a5];
   [v0][a0][v1][a1][v2][a2][v3][a3][v4][a4][v5][a5]
   concat=n=6:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Final Four-Minute Judges Cut" \
  "$FINAL"

echo "$FINAL"
