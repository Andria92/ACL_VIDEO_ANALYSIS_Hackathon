#!/bin/zsh
set -euo pipefail

ROOT="/Users/andriagryffinpro/Documents/ChatGPT/HACKATHON ACL VIDEO ANALYSIS"
BUILD="$ROOT/tmp/complete_presentation_v1"
MONTAGE="/Volumes/ACL_DEMO/demo_assets/ACL_INTRO_FIXED_V9_TIGHTER_52_SECONDS.mov"
APP_INTRO="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 10.34.39 PM.mov"
TECHNICAL="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 10.37.48 PM.mov"
ANNOTATION="/Users/andriagryffinpro/Desktop/Screen Recording 2026-09-04 at 10.43.24 PM.mov"

mkdir -p "$BUILD/frames"

# Full-resolution interface frames used as relevant B-roll under the technical narration.
ffmpeg -hide_banner -loglevel error -y -ss 0 -i "$ANNOTATION" -frames:v 1 -q:v 2 "$BUILD/frames/home.jpg"
ffmpeg -hide_banner -loglevel error -y -ss 45 -i "$ANNOTATION" -frames:v 1 -q:v 2 "$BUILD/frames/workspace.jpg"
ffmpeg -hide_banner -loglevel error -y -ss 90 -i "$ANNOTATION" -frames:v 1 -q:v 2 "$BUILD/frames/player_tracking.jpg"
ffmpeg -hide_banner -loglevel error -y -ss 225 -i "$ANNOTATION" -frames:v 1 -q:v 2 "$BUILD/frames/movement_end.jpg"
ffmpeg -hide_banner -loglevel error -y -ss 270 -i "$ANNOTATION" -frames:v 1 -q:v 2 "$BUILD/frames/evidence_gate.jpg"

# Preserve the completed montage, adding only a short visual fade into the app section.
ffmpeg -hide_banner -loglevel error -y -i "$MONTAGE" \
  -vf "fade=t=out:st=52.066667:d=0.20,format=yuv420p" \
  -af "aresample=48000,pan=stereo|c0=c0|c1=c1" \
  -r 30 -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -movflags +faststart \
  "$BUILD/01_acl_montage.mov"

# Keep only the greeting and high-level purpose. Add breathing room around the cut and
# bring the quiet microphone recording into line with the montage.
ffmpeg -hide_banner -loglevel error -y -i "$APP_INTRO" -filter_complex \
  "[0:v]trim=start=0:end=12.78,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30,tpad=start_mode=clone:start_duration=0.30:stop_mode=clone:stop_duration=0.30,fade=t=in:st=0:d=0.30,format=yuv420p[v];
   [0:a]atrim=start=0:end=12.78,asetpts=PTS-STARTPTS,highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,adelay=300:all=1,apad=pad_dur=0.30,atrim=duration=13.38,aresample=48000[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -movflags +faststart \
  "$BUILD/02_app_introduction_short.mov"

# Technical narration: remove the larger pauses and the repetitive 65.2-85.6 second
# output list. Match each retained idea to an appropriate interface view.
ffmpeg -hide_banner -loglevel error -y \
  -i "$TECHNICAL" \
  -loop 1 -framerate 30 -i "$BUILD/frames/home.jpg" \
  -loop 1 -framerate 30 -i "$BUILD/frames/player_tracking.jpg" \
  -loop 1 -framerate 30 -i "$BUILD/frames/workspace.jpg" \
  -loop 1 -framerate 30 -i "$BUILD/frames/movement_end.jpg" \
  -loop 1 -framerate 30 -i "$BUILD/frames/evidence_gate.jpg" \
  -loop 1 -framerate 30 -i "$BUILD/frames/home.jpg" \
  -filter_complex \
  "[1:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00008,1.035)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,trim=duration=7.55,setpts=PTS-STARTPTS[v0];
   [2:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00007,1.03)':x='(iw-iw/zoom)*0.42':y='(ih-ih/zoom)*0.48':d=1:s=1920x1080:fps=30,trim=duration=12.50,setpts=PTS-STARTPTS[v1];
   [3:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00007,1.03)':x='(iw-iw/zoom)*0.52':y='(ih-ih/zoom)*0.44':d=1:s=1920x1080:fps=30,trim=duration=11.05,setpts=PTS-STARTPTS[v2];
   [4:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00007,1.03)':x='(iw-iw/zoom)*0.40':y='(ih-ih/zoom)*0.56':d=1:s=1920x1080:fps=30,trim=duration=11.35,setpts=PTS-STARTPTS[v3];
   [5:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00005,1.035)':x='(iw-iw/zoom)*0.58':y='(ih-ih/zoom)*0.52':d=1:s=1920x1080:fps=30,trim=duration=21.88,setpts=PTS-STARTPTS[v4];
   [6:v]crop=3114:1752:171:0,zoompan=z='min(zoom+0.00007,1.03)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,trim=duration=12.60,setpts=PTS-STARTPTS[v5];
   [v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0,tpad=stop_mode=clone:stop_duration=0.30,scale=1920:1080:in_range=full:out_range=limited,format=yuv420p,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709[v];
   [0:a]atrim=start=0:end=7.55,asetpts=PTS-STARTPTS[a0];
   [0:a]atrim=start=7.85:end=20.35,asetpts=PTS-STARTPTS[a1];
   [0:a]atrim=start=20.65:end=31.70,asetpts=PTS-STARTPTS[a2];
   [0:a]atrim=start=32.00:end=43.35,asetpts=PTS-STARTPTS[a3];
   [0:a]atrim=start=43.50:end=65.38,asetpts=PTS-STARTPTS[a4];
   [0:a]atrim=start=85.15:end=97.75,asetpts=PTS-STARTPTS[a5];
   [a0][a1][a2][a3][a4][a5]concat=n=6:v=0:a=1,highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,apad=pad_dur=0.30,atrim=duration=77.23,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 77.23 -r 30 -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -movflags +faststart \
  "$BUILD/03_technical_explanation_tight.mov"

# Annotation walkthrough: retain the narration, remove loading/waiting, compress the
# repetitive frame-by-frame box correction to six seconds, and finish after Generate.
ffmpeg -hide_banner -loglevel error -y -i "$ANNOTATION" -filter_complex \
  "[0:v]trim=start=0:end=42.55,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30[v0];
   [0:a]atrim=start=0:end=42.55,asetpts=PTS-STARTPTS[a0];
   [0:v]trim=start=44.70:end=46.20,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30[v1];
   [0:a]atrim=start=44.70:end=46.20,asetpts=PTS-STARTPTS[a1];
   [0:v]trim=start=49.15:end=60.00,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30[v2];
   [0:a]atrim=start=49.15:end=60.00,asetpts=PTS-STARTPTS[a2];
   [0:v]trim=start=60.70:end=65.90,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30[v3];
   [0:a]atrim=start=60.70:end=65.90,asetpts=PTS-STARTPTS[a3];
   [0:v]trim=start=67.85:end=75.75,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30[v4];
   [0:a]atrim=start=67.85:end=75.75,asetpts=PTS-STARTPTS[a4];
   [0:v]trim=start=80.95:end=85.60,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30[v5];
   [0:a]atrim=start=80.95:end=85.60,asetpts=PTS-STARTPTS[a5];
   [0:v]trim=start=85.60:end=228.15,setpts=(PTS-STARTPTS)/23.758333,crop=3114:1752:171:0,scale=1920:1080,fps=30[v6];
   anullsrc=r=48000:cl=mono,atrim=duration=6.00,asetpts=PTS-STARTPTS[a6];
   [0:v]trim=start=228.15:end=249.30,setpts=PTS-STARTPTS,crop=3114:1752:171:0,scale=1920:1080,fps=30[v7];
   [0:a]atrim=start=228.15:end=249.30,asetpts=PTS-STARTPTS[a7];
   [v0][a0][v1][a1][v2][a2][v3][a3][v4][a4][v5][a5][v6][a6][v7][a7]concat=n=8:v=1:a=1[vcat][acat];
   [vcat]fade=t=out:st=99.50:d=0.30,format=yuv420p[v];
   [acat]highpass=f=70,pan=stereo|c0=c0|c1=c0,loudnorm=I=-23:TP=-1.5:LRA=9,afade=t=out:st=99.50:d=0.30,aresample=48000[a]" \
  -map "[v]" -map "[a]" -t 99.80 -r 30 -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -movflags +faststart \
  "$BUILD/04_annotation_walkthrough_tight.mov"

# Current complete presentation assembly. Each component is also retained separately
# so later demo sections can be inserted or rearranged in CapCut.
ffmpeg -hide_banner -loglevel error -y \
  -i "$BUILD/01_acl_montage.mov" \
  -i "$BUILD/02_app_introduction_short.mov" \
  -i "$BUILD/03_technical_explanation_tight.mov" \
  -i "$BUILD/04_annotation_walkthrough_tight.mov" \
  -filter_complex \
  "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a]concat=n=4:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -movflags +faststart \
  "$BUILD/ACL_DEMO_PRESENTATION_ASSEMBLY_V1.mov"

echo "$BUILD"
