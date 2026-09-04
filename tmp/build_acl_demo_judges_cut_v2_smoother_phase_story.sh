#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
V1="$ROOT/tmp/judges_cut_under_4min_v1"
BUILD="$ROOT/tmp/judges_cut_under_4min_v2"
ANNOTATION_V1="$V1/04_annotation_54s.mov"
RESULTS_V1="$V1/05_results_48s.mov"
RESULTS_RAW="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 11.31.01 PM.mov"
SIMILARITY_RAW="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 11.40.34 PM.mov"

mkdir -p "$BUILD"

# Annotation: retain only the prepared clip, athlete tracking, final movement frame,
# save/validate, and generate. Remove the demonstration's remaining hesitant pauses.
ffmpeg -hide_banner -loglevel error -y -i "$ANNOTATION_V1" -filter_complex \
  "[0:v]trim=start=3.46:end=11.96,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=3.46:end=11.96,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=21.82:end=33.98,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=21.82:end=33.98,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=36.02:end=40.66,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=36.02:end=40.66,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=44.24:end=48.36,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=44.24:end=48.36,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=49.14:end=53.60,setpts=PTS-STARTPTS[v4];
   [0:a]atrim=start=49.14:end=53.60,asetpts=PTS-STARTPTS[a4];
   [v0][v1][v2][v3][v4]concat=n=5:v=1:a=0,setpts=PTS/1.05,fps=30,format=yuv420p[v];
   [a0][a1][a2][a3][a4]concat=n=5:v=0:a=1,atempo=1.05,highpass=f=70,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 32.267 -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/04_annotation_smoother_32s.mov"

# Results: tighten all pauses in the existing evidence overview, then restore the
# signature Movement Story and its deterministic, phase-by-phase explanation.
ffmpeg -hide_banner -loglevel error -y -i "$RESULTS_V1" -i "$RESULTS_RAW" -filter_complex \
  "[0:v]trim=start=0:end=2.2969,setpts=PTS-STARTPTS[c0v];
   [0:a]atrim=start=0:end=2.2969,asetpts=PTS-STARTPTS[c0a];
   [0:v]trim=start=2.5787:end=4.0969,setpts=PTS-STARTPTS[c1v];
   [0:a]atrim=start=2.5787:end=4.0969,asetpts=PTS-STARTPTS[c1a];
   [0:v]trim=start=5.3149:end=6.608,setpts=PTS-STARTPTS[c2v];
   [0:a]atrim=start=5.3149:end=6.608,asetpts=PTS-STARTPTS[c2a];
   [0:v]trim=start=6.861:end=7.891,setpts=PTS-STARTPTS[c3v];
   [0:a]atrim=start=6.861:end=7.891,asetpts=PTS-STARTPTS[c3a];
   [0:v]trim=start=8.139:end=8.780,setpts=PTS-STARTPTS[c4v];
   [0:a]atrim=start=8.139:end=8.780,asetpts=PTS-STARTPTS[c4a];
   [0:v]trim=start=9.056:end=12.150,setpts=PTS-STARTPTS[c5v];
   [0:a]atrim=start=9.056:end=12.150,asetpts=PTS-STARTPTS[c5a];
   [0:v]trim=start=12.400:end=15.315,setpts=PTS-STARTPTS[c6v];
   [0:a]atrim=start=12.400:end=15.315,asetpts=PTS-STARTPTS[c6a];
   [0:v]trim=start=15.563:end=18.380,setpts=PTS-STARTPTS[c7v];
   [0:a]atrim=start=15.563:end=18.380,asetpts=PTS-STARTPTS[c7a];

   [1:v]trim=start=54.35:end=68.64,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[p0v];
   [1:a]atrim=start=54.35:end=68.64,asetpts=PTS-STARTPTS,atempo=1.20,pan=stereo|c0=c0|c1=c0[p0a];
   [1:v]trim=start=69.75:end=79.40,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[p1v];
   [1:a]atrim=start=69.75:end=79.40,asetpts=PTS-STARTPTS,atempo=1.20,pan=stereo|c0=c0|c1=c0[p1a];
   [1:v]trim=start=285.95:end=292.50,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[p2v];
   [1:a]atrim=start=285.95:end=292.50,asetpts=PTS-STARTPTS,atempo=1.20,pan=stereo|c0=c0|c1=c0[p2a];
   [1:v]trim=start=292.95:end=309.95,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[p3v];
   [1:a]atrim=start=292.95:end=309.95,asetpts=PTS-STARTPTS,atempo=1.20,pan=stereo|c0=c0|c1=c0[p3a];

   [0:v]trim=start=18.520:end=32.544,setpts=PTS-STARTPTS[d0v];
   [0:a]atrim=start=18.520:end=32.544,asetpts=PTS-STARTPTS[d0a];
   [0:v]trim=start=32.794:end=34.512,setpts=PTS-STARTPTS[d1v];
   [0:a]atrim=start=32.794:end=34.512,asetpts=PTS-STARTPTS[d1a];
   [0:v]trim=start=35.093:end=38.715,setpts=PTS-STARTPTS[d2v];
   [0:a]atrim=start=35.093:end=38.715,asetpts=PTS-STARTPTS[d2a];
   [0:v]trim=start=39.012:end=40.115,setpts=PTS-STARTPTS[d3v];
   [0:a]atrim=start=39.012:end=40.115,asetpts=PTS-STARTPTS[d3a];
   [0:v]trim=start=40.366:end=40.849,setpts=PTS-STARTPTS[d4v];
   [0:a]atrim=start=40.366:end=40.849,asetpts=PTS-STARTPTS[d4a];
   [0:v]trim=start=41.477:end=42.928,setpts=PTS-STARTPTS[d5v];
   [0:a]atrim=start=41.477:end=42.928,asetpts=PTS-STARTPTS[d5a];
   [0:v]trim=start=43.178:end=43.946,setpts=PTS-STARTPTS[d6v];
   [0:a]atrim=start=43.178:end=43.946,asetpts=PTS-STARTPTS[d6a];
   [0:v]trim=start=44.599:end=45.788,setpts=PTS-STARTPTS[d7v];
   [0:a]atrim=start=44.599:end=45.788,asetpts=PTS-STARTPTS[d7a];
   [0:v]trim=start=46.103:end=47.617,setpts=PTS-STARTPTS[d8v];
   [0:a]atrim=start=46.103:end=47.617,asetpts=PTS-STARTPTS[d8a];

   [c0v][c0a][c1v][c1a][c2v][c2a][c3v][c3a][c4v][c4a][c5v][c5a][c6v][c6a][c7v][c7a]
   [p0v][p0a][p1v][p1a][p2v][p2a][p3v][p3a]
   [d0v][d0a][d1v][d1a][d2v][d2a][d3v][d3a][d4v][d4a][d5v][d5a][d6v][d6a][d7v][d7a][d8v][d8a]
   concat=n=21:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]highpass=f=70,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/05_results_evidence_phase_story_smoother.mov"

# Similarity/statistics: retain the purpose, direct supporting evidence, case-level
# statistical comparison, correlations, and group readiness; remove navigation pauses.
ffmpeg -hide_banner -loglevel error -y -i "$SIMILARITY_RAW" -filter_complex \
  "[0:v]trim=start=0:end=20.60,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=20.60,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=99.10:end=107.20,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=99.10:end=107.20,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=197.15:end=214.20,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=197.15:end=214.20,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=217.70:end=219.10,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=217.70:end=219.10,asetpts=PTS-STARTPTS[a3];
   [v0][v1][v2][v3]concat=n=4:v=1:a=0,setpts=PTS/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30,fade=t=out:st=39.042:d=0.25,format=yuv420p[v];
   [a0][a1][a2][a3]concat=n=4:v=0:a=1,atempo=1.20,highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,afade=t=out:st=39.172:d=0.12,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 39.292 -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/06_similarity_stats_smoother_39s.mov"

# Assemble with the unchanged hook, app introduction, and technical method from V1.
ffmpeg -hide_banner -loglevel error -y \
  -i "$V1/01_hook_36s.mov" \
  -i "$V1/02_app_intro_12s.mov" \
  -i "$V1/03_method_36s.mov" \
  -i "$BUILD/04_annotation_smoother_32s.mov" \
  -i "$BUILD/05_results_evidence_phase_story_smoother.mov" \
  -i "$BUILD/06_similarity_stats_smoother_39s.mov" \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a]concat=n=6:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Smoother Four-Minute Cut with Movement Story" \
  "$BUILD/ACL_DEMO_UNDER_4_MIN_V2_SMOOTHER_WITH_PHASE_STORY.mov"

echo "$BUILD"
