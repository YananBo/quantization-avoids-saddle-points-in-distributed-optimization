#!/bin/bash
for ((s=0; s <= 1; s++))
do
  for ((a=0; a <= 7; a++))
  do
    for ((b=1; b <= 3; b++))
    do
      if [ $s -eq 0 ]; then
        python main.py -t $a -r $b -s
      else
        python main.py -t $a -r $b
      fi
    done
  done
done
