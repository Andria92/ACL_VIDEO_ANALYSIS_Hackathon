#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
V1="$ROOT/tmp/judges_cut_under_4min_v1"
V2="$ROOT/tmp/judges_cut_under_4min_v2"
BUILD="$ROOT/tmp/judges_cut_v5_features"
ANNOTATION_SOURCE="$V2/04_annotation_smoother_32s.mov"
RESULTS_SOURCE="$V2/05_results_evidence_phase_story_refined.mov"
CURRENT_FINAL="$V2/ACL_DEMO_UNDER_4_MIN_V4_FINAL_SMOOTH.mov"
RESULTS_RAW="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 11.31.01 PM.mov"
SIMILARITY_RAW="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 11.40.34 PM.mov"

mkdir -p "$BUILD"

# Annotation: remove the remaining hesitation near 01:34 as well as the smaller
# pauses already identified in the opening annotation sentence.
ffmpeg -hide_banner -loglevel error -y -i "$ANNOTATION_SOURCE" -filter_complex \
  "[0:v]trim=start=0:end=3.453,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=3.453,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=3.853:end=5.293,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=3.853:end=5.293,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=5.573:end=8.250,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=5.573:end=8.250,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=9.700:end=11.950,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=9.700:end=11.950,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=13.800:end=32.233,setpts=PTS-STARTPTS[v4];
   [0:a]atrim=start=13.800:end=32.233,asetpts=PTS-STARTPTS[a4];
   [v0][a0][v1][a1][v2][a2][v3][a3][v4][a4]concat=n=5:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]highpass=f=70,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/04_annotation_final_smooth.mov"

# Responsible AI: keep the explanation of meaning and the explicit no-prediction
# boundary, while removing the long hesitations inside the original delivery.
ffmpeg -hide_banner -loglevel error -y -i "$RESULTS_RAW" -filter_complex \
  "[0:v]trim=start=313.22:end=314.15,setpts=PTS-STARTPTS[r0v];
   [0:a]atrim=start=313.22:end=314.15,asetpts=PTS-STARTPTS[r0a];
   [0:v]trim=start=315.35:end=320.25,setpts=PTS-STARTPTS[r1v];
   [0:a]atrim=start=315.35:end=320.25,asetpts=PTS-STARTPTS[r1a];
   [0:v]trim=start=321.25:end=323.00,setpts=PTS-STARTPTS[r2v];
   [0:a]atrim=start=321.25:end=323.00,asetpts=PTS-STARTPTS[r2a];
   [0:v]trim=start=323.55:end=332.95,setpts=PTS-STARTPTS[r3v];
   [0:a]atrim=start=323.55:end=332.95,asetpts=PTS-STARTPTS[r3a];
   [0:v]trim=start=336.85:end=338.85,setpts=PTS-STARTPTS[r4v];
   [0:a]atrim=start=336.85:end=338.85,asetpts=PTS-STARTPTS[r4a];
   [r0v][r1v][r2v][r3v][r4v]concat=n=5:v=1:a=0,
     setpts=PTS/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30,format=yuv420p[v];
   [r0a][r1a][r2a][r3a][r4a]concat=n=5:v=0:a=1,
     atempo=1.20,pan=stereo|c0=c0|c1=c0,highpass=f=70,
     loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/responsible_ai_compact.mov"

# Results: preserve Movement Story and deterministic phase reasoning, remove the
# slow unsupported-points passage containing the 02:52/02:55 pauses, retain the
# change/statistics view, and finish with the responsible-AI boundary.
ffmpeg -hide_banner -loglevel error -y -i "$RESULTS_SOURCE" -i "$BUILD/responsible_ai_compact.mov" -filter_complex \
  "[0:v]trim=start=0:end=51.62,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=51.62,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=68.34:end=76.655,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=68.34:end=76.655,asetpts=PTS-STARTPTS[a1];
   [v0][a0][v1][a1][1:v][1:a]concat=n=3:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]highpass=f=70,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/05_results_phase_story_responsible_ai.mov"

# Distance-measure insert: visibly show the dropdown and name all four measures.
ffmpeg -hide_banner -loglevel error -y -i "$SIMILARITY_RAW" -filter_complex \
  "[0:v]trim=start=36.55:end=39.84,setpts=PTS-STARTPTS[d0v];
   [0:a]atrim=start=36.55:end=39.84,asetpts=PTS-STARTPTS[d0a];
   [0:v]trim=start=47.45:end=49.40,setpts=PTS-STARTPTS[d1v];
   [0:a]atrim=start=47.45:end=49.40,asetpts=PTS-STARTPTS[d1a];
   [0:v]trim=start=50.10:end=51.20,setpts=PTS-STARTPTS[d2v];
   [0:a]atrim=start=50.10:end=51.20,asetpts=PTS-STARTPTS[d2a];
   [0:v]trim=start=51.80:end=54.65,setpts=PTS-STARTPTS[d3v];
   [0:a]atrim=start=51.80:end=54.65,asetpts=PTS-STARTPTS[d3a];
   [d0v][d1v][d2v][d3v]concat=n=4:v=1:a=0,
     setpts=PTS/1.15,crop=3114:1752:171:0,scale=1920:1080,fps=30,format=yuv420p[v];
   [d0a][d1a][d2a][d3a]concat=n=4:v=0:a=1,
     atempo=1.15,pan=stereo|c0=c0|c1=c0,highpass=f=70,
     loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/four_distance_measures.mov"

# Similarity: purpose, the four measures, then supporting evidence and statistics.
ffmpeg -hide_banner -loglevel error -y -i "$CURRENT_FINAL" -i "$BUILD/four_distance_measures.mov" -filter_complex \
  "[0:v]trim=start=192.90:end=206.32,setpts=PTS-STARTPTS[s0v];
   [0:a]atrim=start=192.90:end=206.32,asetpts=PTS-STARTPTS[s0a];
   [0:v]trim=start=206.88:end=228.633,setpts=PTS-STARTPTS[s1v];
   [0:a]atrim=start=206.88:end=228.633,asetpts=PTS-STARTPTS[s1a];
   [s0v][s0a][1:v][1:a][s1v][s1a]concat=n=3:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]highpass=f=70,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/06_similarity_four_measures_stats.mov"

ffmpeg -hide_banner -loglevel error -y \
  -i "$V1/01_hook_36s.mov" \
  -i "$V1/02_app_intro_12s.mov" \
  -i "$V1/03_method_36s.mov" \
  -i "$BUILD/04_annotation_final_smooth.mov" \
  -i "$BUILD/05_results_phase_story_responsible_ai.mov" \
  -i "$BUILD/06_similarity_four_measures_stats.mov" \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a]concat=n=6:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Judges Cut with Movement Story, Responsible AI, and Four Distance Measures" \
  "$BUILD/ACL_DEMO_FINAL_WITH_RESPONSIBLE_AI_AND_FOUR_MEASURES.mov"

echo "$BUILD"
