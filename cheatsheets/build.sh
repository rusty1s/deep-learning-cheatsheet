#!/bin/bash

files=( convolutions autoencoders )

for name in "${files[@]}"; do
  pdflatex "$name"
  bibtex "$name"
  pdflatex "$name"
done
