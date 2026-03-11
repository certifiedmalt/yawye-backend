#!/bin/bash
# Assemble with split voiceovers - each line synced to its clip
# Usage: bash assemble_v2.sh <script_num> <hook_text> <bad_text> <cta_text>

S=$1
HOOK="$2"
BAD="$3"
CTA="$4"
DIR="/app/marketing/production"
VO="/app/marketing/voiceovers/split"
FINAL="/app/marketing/FINAL_${S}.mp4"

echo "Assembling Script ${S} (v2 - synced voiceovers)..."

# Step 1: Add voiceover to each individual clip
for x in a b c; do
    ffmpeg -y \
      -i ${DIR}/${S}${x}.mp4 \
      -i ${VO}/${S}${x}.mp3 \
      -c:v copy -c:a aac -b:a 192k \
      -map 0:v:0 -map 1:a:0 \
      -shortest \
      ${DIR}/voiced_${S}${x}.mp4 2>/dev/null
done

# Step 2: Concat the 3 voiced clips
echo "file 'voiced_${S}a.mp4'" > ${DIR}/concat_${S}.txt
echo "file 'voiced_${S}b.mp4'" >> ${DIR}/concat_${S}.txt
echo "file 'voiced_${S}c.mp4'" >> ${DIR}/concat_${S}.txt

ffmpeg -y -f concat -safe 0 -i ${DIR}/concat_${S}.txt -c copy ${DIR}/merged_${S}.mp4 2>/dev/null

# Get total duration for text timing
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 ${DIR}/merged_${S}.mp4 2>/dev/null)
THIRD=$(python3 -c "print(round($DUR/3, 2))")
TWO_THIRD=$(python3 -c "print(round($DUR*2/3, 2))")

# Step 3: Add text overlays timed to each third
ffmpeg -y \
  -i ${DIR}/merged_${S}.mp4 \
  -vf "\
    drawtext=text='${HOOK}':fontcolor=white:fontsize=42:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-90:enable='between(t,0.5,${THIRD})',\
    drawtext=text='${BAD}':fontcolor=#FF5252:fontsize=48:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-90:enable='between(t,${THIRD},${TWO_THIRD})',\
    drawtext=text='${CTA}':fontcolor=#4CAF50:fontsize=44:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-130:enable='between(t,${TWO_THIRD},${DUR})',\
    drawtext=text='You Are What You Eat':fontcolor=white:fontsize=32:borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-70:enable='between(t,${TWO_THIRD},${DUR})'" \
  -c:a copy \
  ${FINAL} 2>/dev/null

if [ -f "$FINAL" ] && [ -s "$FINAL" ]; then
    SIZE=$(ls -lh $FINAL | awk '{print $5}')
    FDUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $FINAL 2>/dev/null)
    echo "DONE: ${FINAL} (${SIZE}, ${FDUR}s)"
else
    echo "FAILED"
fi

# Cleanup
rm -f ${DIR}/voiced_${S}a.mp4 ${DIR}/voiced_${S}b.mp4 ${DIR}/voiced_${S}c.mp4 ${DIR}/concat_${S}.txt ${DIR}/merged_${S}.mp4
