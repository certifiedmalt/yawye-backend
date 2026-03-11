#!/bin/bash
# Enhance all FINAL videos with a 3.5s QR code end-card
# Creates endcard video, then appends it to each FINAL video

DIR="/app/marketing"
PROD="/app/marketing/production"
ENDCARD_IMG="${PROD}/endcard.png"
ENDCARD_VID="${PROD}/endcard_clip.mp4"

echo "=== Creating end-card video clip (3.5s) ==="
# Create a 3.5s video from the end card image with silent audio
ffmpeg -y -loop 1 -i ${ENDCARD_IMG} -f lavfi -i anullsrc=r=44100:cl=stereo \
  -c:v libx264 -t 3.5 -pix_fmt yuv420p -r 24 -s 1280x720 \
  -c:a aac -b:a 128k -shortest \
  ${ENDCARD_VID} 2>/dev/null

if [ ! -f "$ENDCARD_VID" ]; then
    echo "FAILED to create endcard video"
    exit 1
fi
echo "End-card clip ready ($(ls -lh $ENDCARD_VID | awk '{print $5}'))"

echo ""
echo "=== Enhancing videos ==="

for VIDEO in ${DIR}/FINAL_*.mp4; do
    BASENAME=$(basename "$VIDEO")
    TEMP="${PROD}/temp_${BASENAME}"
    
    echo -n "Processing ${BASENAME}... "
    
    # Re-encode original to ensure compatible format, then concat with endcard
    ffmpeg -y \
      -i "$VIDEO" \
      -i "$ENDCARD_VID" \
      -filter_complex "\
        [0:v]scale=1280:720,setsar=1,fps=24[v0];\
        [1:v]scale=1280:720,setsar=1,fps=24[v1];\
        [0:a]aresample=44100[a0];\
        [1:a]aresample=44100[a1];\
        [v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]" \
      -map "[outv]" -map "[outa]" \
      -c:v libx264 -crf 23 -preset fast -c:a aac -b:a 192k \
      "$TEMP" 2>/dev/null
    
    if [ -f "$TEMP" ] && [ -s "$TEMP" ]; then
        mv "$TEMP" "$VIDEO"
        SIZE=$(ls -lh "$VIDEO" | awk '{print $5}')
        DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO" 2>/dev/null)
        echo "DONE (${SIZE}, ${DUR}s)"
    else
        echo "FAILED"
        rm -f "$TEMP"
    fi
done

echo ""
echo "=== All videos enhanced ==="
rm -f ${ENDCARD_VID}
