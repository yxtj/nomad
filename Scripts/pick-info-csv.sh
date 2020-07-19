
LOG_LIST=$(ls 10*)
OUT_FOLDER=score

for f in $LOG_LIST; do 
  grep 'termination check' $f | awk 'BEGIN{OFS=""} {if(NF == 15) print $9,$11,$13,$15;}' | sed 's/\:/,/g' > $OUT_FOLDER/$f-rmse.csv;
  grep 'finish checkpoint' $f | awk 'BEGIN{OFS=","} { if(NF==12) print $8,$10,$12; }' > $OUT_FOLDER/$f-cp.csv;
done
