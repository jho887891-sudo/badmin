#!/usr/bin/env bash
echo "=== remote hard-negative pools ==="
for d in $(find /home/T7/dgut/robot_sim -maxdepth 4 -type d -name "hard_negatives*" 2>/dev/null); do
  n=$(find "$d" -type f ( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" ) | wc -l)
  echo "$d  images=$n"
  find "$d" -maxdepth 2 -type f ! -iname "*.jpg" ! -iname "*.jpeg" ! -iname "*.png" -printf "    doc %p %s\n" 2>/dev/null | head -4
done
echo "=== any hn2_* anywhere ==="
find /home/T7/dgut/robot_sim -name "hn2_*" -type f 2>/dev/null | head -3
echo "=== sha256 of one pool file for cross-check ==="
find /home/T7/dgut/robot_sim -path "*hard_negatives/raw/hn_002.jpg" 2>/dev/null | head -2 | while read f; do echo "$f $(sha256sum "$f" | cut -c1-32)"; done
