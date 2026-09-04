#!/bin/zsh
set -euo pipefail

INPUT='/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V6_SHORTER_INJURIES_NO_9_MONTH_QUOTE.mov'
OUTPUT='/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS/tmp/ACL_INTRO_FIXED_V7_EXACT_EMMA_Q_ANGLES_CUT.mov'

# Emma's retained line ends at 35.65:
# “We're women, we're built differently. We've got Q angles.”
# Delete “we've got hormones, we've got...” and the remainder before the next
# mechanical quote, which begins at 36.971.
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT" \
  -filter_complex \
  "[0:v]trim=start=0:end=35.650,setpts=PTS-STARTPTS[v1]; \
   [0:a]atrim=start=0:end=35.650,asetpts=PTS-STARTPTS[a1]; \
   [0:v]trim=start=36.971:end=56.800,setpts=PTS-STARTPTS[v2]; \
   [0:a]atrim=start=36.971:end=56.800,asetpts=PTS-STARTPTS[a2]; \
   [v1][a1][v2][a2]concat=n=2:v=1:a=1[vout][aout]" \
  -map '[vout]' -map '[aout]' \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 256k -movflags +faststart \
  -metadata title='ACL Intro — Exact Emma Q Angles Cut' \
  "$OUTPUT"
