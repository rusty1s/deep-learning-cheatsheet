#!/bin/sh

name="graphs"

pdflatex "$name"
bibtex "$name"
pdflatex "$name"
