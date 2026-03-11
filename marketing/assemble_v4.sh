#!/bin/bash
# V4: 4-clip assembly with synced voiceovers + real icon + branding
# Usage: bash assemble_v4.sh <script_num> <hook_text> <bad_text> <turn_text> <cta_text>

S=$1
HOOK="$2"
BAD="$3"
TURN="$4"
CTA="$5"
DIR="/app/marketing/production"
VO="/app/marketing/voiceovers/split"
LOGO="/app/marketing/real_icon_watermark.png"
FINAL="/app/marketing/FINAL_${S}.mp4"

echo "Assembling Script ${S} (v4 - 4 clips + synced VO + real icon)..."

# Step 1: Add voiceover to each clip
for x in a b c d; do
    if [ -f "${DIR}/${S}${x}.mp4" ]; then
        ffmpeg -y -i ${DIR}/${S}${x}.mp4 -i ${VO}/${S}${x}.mp3 -c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0 -shortest ${DIR}/voiced_${S}${x}.mp4 2>/dev/null
    fi
done

# Step 2: Concat
> ${DIR}/concat_${S}.txt
for x in a b c d; do
    if [ -f "${DIR}/voiced_${S}${x}.mp4" ]; then
        echo "file 'voiced_${S}${x}.mp4'" >> ${DIR}/concat_${S}.txt
    fi
done
ffmpeg -y -f concat -safe 0 -i ${DIR}/concat_${S}.txt -c copy ${DIR}/merged_${S}.mp4 2>/dev/null

# Get duration and quarters
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 ${DIR}/merged_${S}.mp4 2>/dev/null)
Q1=$(python3 -c "print(round($DUR/4, 2))")
Q2=$(python3 -c "print(round($DUR/2, 2))")
Q3=$(python3 -c "print(round($DUR*3/4, 2))")
LAST_4=$(python3 -c "print(round($DUR - 4, 2))")

# Step 3: Logo + text overlays timed to each quarter
ffmpeg -y \
  -i ${DIR}/merged_${S}.mp4 \
  -i ${LOGO} \
  -filter_complex "\
    [1:v]scale=80:80[logo];\
    [0:v][logo]overlay=W-100:20[v1];\
    [v1]drawtext=text='You Are What You Eat':fontcolor=white@0.7:fontsize=18:borderw=1:bordercolor=black@0.5:x=W-260:y=108[v2];\
    [v2]drawtext=text='${HOOK}':fontcolor=white:fontsize=40:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-90:enable='between(t,0.3,${Q1})'[v3];\
    [v3]drawtext=text='${BAD}':fontcolor=#FF5252:fontsize=46:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-90:enable='between(t,${Q1},${Q2})'[v4];\
    [v4]drawtext=text='${TURN}':fontcolor=white:fontsize=40:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-90:enable='between(t,${Q2},${Q3})'[v5];\
    [v5]drawtext=text='${CTA}':fontcolor=#00E676:fontsize=40:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-130:enable='between(t,${Q3},${DUR})'[v6];\
    [v6]drawtext=text='Free on Google Play':fontcolor=white:fontsize=28:borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-70:enable='between(t,${LAST_4},${DUR})'[vout]" \
  -map "[vout]" -map 0:a -c:a copy \
  ${FINAL} 2>/dev/null

if [ -f "$FINAL" ] && [ -s "$FINAL" ]; then
    SIZE=$(ls -lh $FINAL | awk '{print $5}')
    FDUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $FINAL 2>/dev/null)
    echo "DONE: ${FINAL} (${SIZE}, ${FDUR}s)"
else
    echo "FAILED"
fi

rm -f ${DIR}/voiced_${S}a.mp4 ${DIR}/voiced_${S}b.mp4 ${DIR}/voiced_${S}c.mp4 ${DIR}/voiced_${S}d.mp4 ${DIR}/concat_${S}.txt ${DIR}/merged_${S}.mp4
