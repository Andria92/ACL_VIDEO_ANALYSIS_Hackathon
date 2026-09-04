#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
V1="$ROOT/tmp/judges_cut_under_4min_v1"
BUILD="$ROOT/tmp/judges_cut_v5_features"
SIM_SOURCE="$BUILD/06_similarity_four_measures_stats.mov"
SIM_FINAL="$BUILD/06_similarity_four_measures_stats_final_smooth.mov"
FINAL="$BUILD/ACL_DEMO_FINAL_V5_SMOOTH_FEATURE_COMPLETE.mov"

# Remove the remaining long hesitation before “the correlations”.
ffmpeg -hide_banner -loglevel error -y -i "$SIM_SOURCE" -filter_complex \
  "[0:v]trim=start=0:end=36.62,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=36.62,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=38.34:end=43.267,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=38.34:end=43.267,asetpts=PTS-STARTPTS[a1];
   [v0][a0][v1][a1]concat=n=2:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]highpass=f=70,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$SIM_FINAL"

ffmpeg -hide_banner -loglevel error -y \
  -i "$V1/01_hook_36s.mov" \
  -i "$V1/02_app_intro_12s.mov" \
  -i "$V1/03_method_36s.mov" \
  -i "$BUILD/04_annotation_final_smooth.mov" \
  -i "$BUILD/05_results_phase_story_responsible_ai.mov" \
  -i "$SIM_FINAL" \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a]concat=n=6:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Smooth Feature-Complete Judges Cut" \
  "$FINAL"

echo "$FINAL"
