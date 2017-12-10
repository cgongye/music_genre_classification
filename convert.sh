#! /bin/bash
files=`find . -name "*.au"`
for f in $files
do
    ffmpeg -i $f $f.wav
done
rm $files