#!/bin/bash

files=( convolutions autoencoders applications deadline-miss-rate )

for name in "${files[@]}"; do
  pdflatex "$name"
  bibtex "$name"
  pdflatex "$name"
done
