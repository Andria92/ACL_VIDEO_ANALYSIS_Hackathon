#!/bin/zsh
set -euo pipefail

INPUT='/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V8_NO_DUPLICATE_ARSENAL_RUNNING.mov'
OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V9_TIGHTER_52_SECONDS.mov'

# Tighten only verified silent pauses in the final scientific/recovery section.
# Every spoken word and all previously approved injury/Emma edits remain intact.
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT" \
  -filter_complex \
  "[0:v]trim=start=0:end=39.660,setpts=PTS-STARTPTS[v1]; \
   [0:a]atrim=start=0:end=39.660,asetpts=PTS-STARTPTS[a1]; \
   [0:v]trim=start=40.060:end=40.820,setpts=PTS-STARTPTS[v2]; \
   [0:a]atrim=start=40.060:end=40.820,asetpts=PTS-STARTPTS[a2]; \
   [0:v]trim=start=41.900:end=44.280,setpts=PTS-STARTPTS[v3]; \
   [0:a]atrim=start=41.900:end=44.280,asetpts=PTS-STARTPTS[a3]; \
   [0:v]trim=start=45.320:end=47.300,setpts=PTS-STARTPTS[v4]; \
   [0:a]atrim=start=45.320:end=47.300,asetpts=PTS-STARTPTS[a4]; \
   [0:v]trim=start=47.600:end=53.400,setpts=PTS-STARTPTS[v5]; \
   [0:a]atrim=start=47.600:end=53.400,asetpts=PTS-STARTPTS[a5]; \
   [0:v]trim=start=53.700:end=54.560,setpts=PTS-STARTPTS[v6]; \
   [0:a]atrim=start=53.700:end=54.560,asetpts=PTS-STARTPTS[a6]; \
   [0:v]trim=start=54.700:end=55.467,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=0.020[v7]; \
   [0:a]atrim=start=54.700:end=55.485,asetpts=PTS-STARTPTS[a7]; \
   [v1][a1][v2][a2][v3][a3][v4][a4][v5][a5][v6][a6][v7][a7]concat=n=7:v=1:a=1[vout][aout]" \
  -map '[vout]' -map '[aout]' \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 256k -movflags +faststart \
  -metadata title='ACL Intro — Tighter 52-Second Cut' \
  "$OUTPUT"
