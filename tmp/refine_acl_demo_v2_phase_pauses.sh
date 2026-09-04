#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
V1="$ROOT/tmp/judges_cut_under_4min_v1"
BUILD="$ROOT/tmp/judges_cut_under_4min_v2"
SOURCE="$BUILD/05_results_evidence_phase_story_smoother.mov"
REFINED="$BUILD/05_results_evidence_phase_story_refined.mov"
FINAL="$BUILD/ACL_DEMO_UNDER_4_MIN_V2_REFINED_WITH_PHASE_STORY.mov"

# Remove only detected silence, preserving every spoken sentence and its screen context.
ffmpeg -hide_banner -loglevel error -y -i "$SOURCE" -filter_complex \
  "[0:v]trim=start=0:end=9.60,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=9.60,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=9.90:end=17.30,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=9.90:end=17.30,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=17.88:end=38.34,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=17.88:end=38.34,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=39.44:end=46.08,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=39.44:end=46.08,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=46.56:end=49.42,setpts=PTS-STARTPTS[v4];
   [0:a]atrim=start=46.56:end=49.42,asetpts=PTS-STARTPTS[a4];
   [0:v]trim=start=50.06:end=51.84,setpts=PTS-STARTPTS[v5];
   [0:a]atrim=start=50.06:end=51.84,asetpts=PTS-STARTPTS[a5];
   [0:v]trim=start=52.30:end=68.78,setpts=PTS-STARTPTS[v6];
   [0:a]atrim=start=52.30:end=68.78,asetpts=PTS-STARTPTS[a6];
   [0:v]trim=start=69.42:end=72.65,setpts=PTS-STARTPTS[v7];
   [0:a]atrim=start=69.42:end=72.65,asetpts=PTS-STARTPTS[a7];
   [0:v]trim=start=73.10:end=81.20,setpts=PTS-STARTPTS[v8];
   [0:a]atrim=start=73.10:end=81.20,asetpts=PTS-STARTPTS[a8];
   [v0][a0][v1][a1][v2][a2][v3][a3][v4][a4][v5][a5][v6][a6][v7][a7][v8][a8]
   concat=n=9:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]highpass=f=70,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$REFINED"

ffmpeg -hide_banner -loglevel error -y \
  -i "$V1/01_hook_36s.mov" \
  -i "$V1/02_app_intro_12s.mov" \
  -i "$V1/03_method_36s.mov" \
  -i "$BUILD/04_annotation_smoother_32s.mov" \
  -i "$REFINED" \
  -i "$BUILD/06_similarity_stats_smoother_39s.mov" \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a]concat=n=6:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Refined Four-Minute Cut with Movement Story" \
  "$FINAL"

echo "$FINAL"
