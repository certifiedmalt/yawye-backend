#!/bin/bash
# Assemble a 3-part script into a final video with voiceover + text overlays
# Usage: bash assemble.sh <script_num> <voiceover_file> <hook_text> <mid_text> <cta_text>

SCRIPT=$1
VO_FILE=$2
HOOK="$3"
MID="$4"
CTA="$5"

DIR="/app/marketing/production"
FINAL="/app/marketing/FINAL_${SCRIPT}.mp4"

echo "Assembling Script ${SCRIPT}..."

# Concat the 3 clips
echo "file '${DIR}/${SCRIPT}a.mp4'" > ${DIR}/concat_${SCRIPT}.txt
echo "file '${DIR}/${SCRIPT}b.mp4'" >> ${DIR}/concat_${SCRIPT}.txt
echo "file '${DIR}/${SCRIPT}c.mp4'" >> ${DIR}/concat_${SCRIPT}.txt

ffmpeg -y -f concat -safe 0 -i ${DIR}/concat_${SCRIPT}.txt -c copy ${DIR}/merged_${SCRIPT}.mp4 2>/dev/null

# Add voiceover
ffmpeg -y \
  -i ${DIR}/merged_${SCRIPT}.mp4 \
  -i ${VO_FILE} \
  -c:v copy -c:a aac -b:a 192k \
  -map 0:v:0 -map 1:a:0 \
  -shortest \
  ${DIR}/audio_${SCRIPT}.mp4 2>/dev/null

# Add text overlays with crossfade-style timing
ffmpeg -y \
  -i ${DIR}/audio_${SCRIPT}.mp4 \
  -vf "\
    drawtext=text='${HOOK}':fontcolor=white:fontsize=42:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-100:enable='between(t,0,3.5)',\
    drawtext=text='ULTRA PROCESSED':fontcolor=#FF5252:fontsize=52:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h/2+60:enable='between(t,5,8)',\
    drawtext=text='${MID}':fontcolor=white:fontsize=40:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-100:enable='between(t,9,14)',\
    drawtext=text='MINIMALLY PROCESSED':fontcolor=#4CAF50:fontsize=48:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h/2+60:enable='between(t,16,20)',\
    drawtext=text='${CTA}':fontcolor=white:fontsize=38:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-140:enable='between(t,20,24)',\
    drawtext=text='You Are What You Eat':fontcolor=#4CAF50:fontsize=36:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-80:enable='between(t,21,24)'" \
  -c:a copy \
  ${FINAL} 2>/dev/null

if [ -f "$FINAL" ]; then
    SIZE=$(ls -lh $FINAL | awk '{print $5}')
    DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $FINAL 2>/dev/null)
    echo "DONE: ${FINAL} (${SIZE}, ${DUR}s)"
else
    echo "FAILED to assemble ${SCRIPT}"
fi

# Cleanup temp files
rm -f ${DIR}/concat_${SCRIPT}.txt ${DIR}/merged_${SCRIPT}.mp4 ${DIR}/audio_${SCRIPT}.mp4
