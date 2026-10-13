#!/bin/zsh
# Train the per-sector models on the v2 (relative) inputs. Run from model_v2/.
for s in 10 15 20 25 30 35 40 45 50 55 60; do
  echo "=== sector$s $(date +%H:%M)"
  python ../fundamental_run_model.py -sector_name sector$s -tic_column tic \
    -fundamental ./prepared/final_ratios.csv -sector ./prepared/sector$s.xlsx 2>&1 | grep -E "Time Spent|Error|error" | grep -v resource_tracker
done
echo "=== done $(date +%H:%M)"
