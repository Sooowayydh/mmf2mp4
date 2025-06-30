#!/bin/bash

echo "============================================================"
echo "MMF to MP4 Converter"
echo "============================================================"
echo
echo "This will convert your MMF files to MP4 format."
echo
read -p "Press Enter to continue..."
echo

python3 mmf_decoder.py

echo
echo "============================================================"
echo "Conversion finished!"
echo "============================================================"
echo
read -p "Press Enter to exit..." 