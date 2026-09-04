#!/bin/zsh
set -euo pipefail

INPUT='/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V3_EMMA_Q_ANGLES_CUT.mov'
OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V4_EMMA_SHORTENED.mov'

# Keep Emma through “we've got Q angles” at 36.70, then jump directly to the
# next mechanical quote at 39.63. Both picture and sound in between are removed.
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT" \
  -filter_complex \
  "[0:v]trim=start=0:end=36.70,setpts=PTS-STARTPTS[v1]; \
   [0:a]atrim=start=0:end=36.70,asetpts=PTS-STARTPTS[a1]; \
   [0:v]trim=start=39.63:end=59.50,setpts=PTS-STARTPTS[v2]; \
   [0:a]atrim=start=39.63:end=59.50,asetpts=PTS-STARTPTS[a2]; \
   [v1][a1][v2][a2]concat=n=2:v=1:a=1[vout][aout]" \
  -map '[vout]' -map '[aout]' \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 256k -movflags +faststart \
  -metadata title='ACL Intro — Emma Ends at Q Angles' \
  "$OUTPUT"
