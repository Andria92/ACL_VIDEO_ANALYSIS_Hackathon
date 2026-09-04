#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
BUILD="$ROOT/tmp/judges_cut_under_4min_v1"
INTRO="/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V9_TIGHTER_52_SECONDS.mov"
APP="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 10.34.39 PM.mov"
TECH="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 10.37.48 PM.mov"
ANNOTATION="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 10.43.24 PM.mov"
RESULTS="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 11.31.01 PM.mov"
SIMILARITY="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 11.40.34 PM.mov"
V1_FRAMES="$ROOT/tmp/complete_presentation_v1/frames"

mkdir -p "$BUILD"

# 1. Human impact and scientific question: 52 seconds -> about 36 seconds.
# Keep the injuries/epidemic, Beth's description, uncertainty, biomechanics, Q angles,
# and the closing risk-reduction message.
ffmpeg -hide_banner -loglevel error -y -i "$INTRO" -filter_complex \
  "[0:v]trim=start=0:end=19.60,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=19.60,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=20.06:end=23.74,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=20.06:end=23.74,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=27.24:end=30.46,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=27.24:end=30.46,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=33.10:end=35.58,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=33.10:end=35.58,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=45.30:end=52.266667,setpts=PTS-STARTPTS[v4];
   [0:a]atrim=start=45.30:end=52.266667,asetpts=PTS-STARTPTS[a4];
   [v0][a0][v1][a1][v2][a2][v3][a3][v4][a4]concat=n=5:v=1:a=1[vcat][acat];
   [vcat]fade=t=out:st=35.746667:d=0.20,format=yuv420p[v];
   [acat]aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/01_hook_36s.mov"

# 2. Concise app introduction, gently accelerated while retaining the natural delivery.
ffmpeg -hide_banner -loglevel error -y -i "$APP" -filter_complex \
  "[0:v]trim=start=0:end=12.78,setpts=(PTS-STARTPTS)/1.05,crop=3114:1752:171:0,scale=1920:1080,fps=30,tpad=start_mode=clone:start_duration=0.25,fade=t=in:st=0:d=0.25,format=yuv420p[v];
   [0:a]atrim=start=0:end=12.78,asetpts=PTS-STARTPTS,atempo=1.05,highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,adelay=250:all=1,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 12.421 -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/02_app_intro_12s.mov"

# 3. Essential technical method. Relevant interface views replace the static landing page.
ffmpeg -hide_banner -loglevel error -y \
  -i "$TECH" \
  -loop 1 -framerate 30 -i "$V1_FRAMES/home.jpg" \
  -loop 1 -framerate 30 -i "$V1_FRAMES/player_tracking.jpg" \
  -loop 1 -framerate 30 -i "$V1_FRAMES/workspace.jpg" \
  -loop 1 -framerate 30 -i "$V1_FRAMES/evidence_gate.jpg" \
  -loop 1 -framerate 30 -i "$V1_FRAMES/home.jpg" \
  -filter_complex \
  "[1:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00010,1.03)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,trim=duration=6.291667,setpts=PTS-STARTPTS[v0];
   [2:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00009,1.03)':x='(iw-iw/zoom)*0.42':y='(ih-ih/zoom)*0.48':d=1:s=1920x1080:fps=30,trim=duration=10.416667,setpts=PTS-STARTPTS[v1];
   [3:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00009,1.03)':x='(iw-iw/zoom)*0.52':y='(ih-ih/zoom)*0.44':d=1:s=1920x1080:fps=30,trim=duration=9.208333,setpts=PTS-STARTPTS[v2];
   [4:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00009,1.03)':x='(iw-iw/zoom)*0.58':y='(ih-ih/zoom)*0.52':d=1:s=1920x1080:fps=30,trim=duration=6.983333,setpts=PTS-STARTPTS[v3];
   [5:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00010,1.03)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,trim=duration=3.125,setpts=PTS-STARTPTS[v4];
   [v0][v1][v2][v3][v4]concat=n=5:v=1:a=0,tpad=stop_mode=clone:stop_duration=0.20,scale=1920:1080:in_range=full:out_range=limited,format=yuv420p,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709[v];
   [0:a]atrim=start=0:end=7.55,asetpts=PTS-STARTPTS[a0];
   [0:a]atrim=start=7.85:end=20.35,asetpts=PTS-STARTPTS[a1];
   [0:a]atrim=start=20.65:end=31.70,asetpts=PTS-STARTPTS[a2];
   [0:a]atrim=start=57.00:end=65.38,asetpts=PTS-STARTPTS[a3];
   [0:a]atrim=start=94.00:end=97.75,asetpts=PTS-STARTPTS[a4];
   [a0][a1][a2][a3][a4]concat=n=5:v=0:a=1,atempo=1.20,highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,apad=pad_dur=0.20,atrim=duration=36.225,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 36.225 -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/03_method_36s.mov"

# 4. Annotation in one minute: preserve the explanation, accelerate narration slightly,
# and compress the repetitive correction of boxes across frames to 2.5 seconds.
ffmpeg -hide_banner -loglevel error -y -i "$ANNOTATION" -filter_complex \
  "[0:v]trim=start=0:end=4.10,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[v0];
   [0:a]atrim=start=0:end=4.10,asetpts=PTS-STARTPTS,atempo=1.20[a0];
   [0:v]trim=start=15.85:end=26.05,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[v1];
   [0:a]atrim=start=15.85:end=26.05,asetpts=PTS-STARTPTS,atempo=1.20[a1];
   [0:v]trim=start=49.10:end=60.00,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[v2];
   [0:a]atrim=start=49.10:end=60.00,asetpts=PTS-STARTPTS,atempo=1.20[a2];
   [0:v]trim=start=60.70:end=75.75,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[v3];
   [0:a]atrim=start=60.70:end=75.75,asetpts=PTS-STARTPTS,atempo=1.20[a3];
   [0:v]trim=start=85.65:end=228.15,setpts=(PTS-STARTPTS)/57,crop=3114:1752:171:0,scale=1920:1080,fps=30[v4];
   anullsrc=r=48000:cl=mono,atrim=duration=2.50,asetpts=PTS-STARTPTS[a4];
   [0:v]trim=start=228.15:end=249.30,setpts=(PTS-STARTPTS)/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30[v5];
   [0:a]atrim=start=228.15:end=249.30,asetpts=PTS-STARTPTS,atempo=1.20[a5];
   [v0][a0][v1][a1][v2][a2][v3][a3][v4][a4][v5][a5]concat=n=6:v=1:a=1[vcat][acat];
   [vcat]format=yuv420p[v];
   [acat]highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 53.667 -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/04_annotation_54s.mov"

# 5. Results: show the video beside the analysis, supported phases, missing-evidence
# handling, change measurements, and descriptive statistics.
ffmpeg -hide_banner -loglevel error -y -i "$RESULTS" -filter_complex \
  "[0:v]trim=start=0:end=12.56,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=12.56,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=79.64:end=88.92,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=79.64:end=88.92,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=182.92:end=192.10,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=182.92:end=192.10,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=193.08:end=206.60,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=193.08:end=206.60,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=208.20:end=218.10,setpts=PTS-STARTPTS[v4];
   [0:a]atrim=start=208.20:end=218.10,asetpts=PTS-STARTPTS[a4];
   [0:v]trim=start=224.00:end=226.70,setpts=PTS-STARTPTS[v5];
   [0:a]atrim=start=224.00:end=226.70,asetpts=PTS-STARTPTS[a5];
   [v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0,setpts=PTS/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30,format=yuv420p[v];
   [a0][a1][a2][a3][a4][a5]concat=n=6:v=0:a=1,atempo=1.20,highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 47.617 -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/05_results_48s.mov"

# 6. Similarity and statistics: retain the purpose, supporting evidence, explorer,
# correlations, and group-readiness conclusion.
ffmpeg -hide_banner -loglevel error -y -i "$SIMILARITY" -filter_complex \
  "[0:v]trim=start=0:end=20.60,setpts=PTS-STARTPTS[v0];
   [0:a]atrim=start=0:end=20.60,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=99.10:end=113.40,setpts=PTS-STARTPTS[v1];
   [0:a]atrim=start=99.10:end=113.40,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=177.00:end=185.90,setpts=PTS-STARTPTS[v2];
   [0:a]atrim=start=177.00:end=185.90,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=197.15:end=214.20,setpts=PTS-STARTPTS[v3];
   [0:a]atrim=start=197.15:end=214.20,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=217.70:end=219.10,setpts=PTS-STARTPTS[v4];
   [0:a]atrim=start=217.70:end=219.10,asetpts=PTS-STARTPTS[a4];
   [v0][v1][v2][v3][v4]concat=n=5:v=1:a=0,setpts=PTS/1.20,crop=3114:1752:171:0,scale=1920:1080,fps=30,fade=t=out:st=51.625:d=0.25,format=yuv420p[v];
   [a0][a1][a2][a3][a4]concat=n=5:v=0:a=1,atempo=1.20,highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,afade=t=out:st=51.755:d=0.12,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 51.875 -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart "$BUILD/06_similarity_stats_52s.mov"

# Final judges' cut. The expected duration is about 3:58.4.
ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD/01_hook_36s.mov" \
  -i "$BUILD/02_app_intro_12s.mov" \
  -i "$BUILD/03_method_36s.mov" \
  -i "$BUILD/04_annotation_54s.mov" \
  -i "$BUILD/05_results_48s.mov" \
  -i "$BUILD/06_similarity_stats_52s.mov" \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a]concat=n=6:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 \
  -c:a aac -b:a 192k -movflags +faststart \
  -metadata title="ACL Movement Analytics Lab — Four-Minute Judges' Cut" \
  "$BUILD/ACL_DEMO_JUDGES_CUT_UNDER_4_MIN_V1.mov"

echo "$BUILD"
