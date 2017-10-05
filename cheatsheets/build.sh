#!/bin/bash

files=( convolutions autoencoders applications )

for name in "${files[@]}"; do
  pdflatex "$name"
  bibtex "$name"
  pdflatex "$name"
done
