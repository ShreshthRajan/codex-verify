#!/bin/sh
# Build the arXiv replacement source package (non-anonymous preprint) and test that it compiles.
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error preprint.tex >/dev/null
D=$(mktemp -d)
mkdir -p "$D/tables" "$D/figures"
cp preprint.tex "$D/main.tex"
cp preprint.bbl "$D/main.bbl"
cp body.tex appendix.tex references.bib aaai2027.sty aaai2027.bst "$D/"
cp tables/*.tex "$D/tables/"
cp figures/roc.pdf figures/pr.pdf "$D/figures/"
(cd "$D" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && grep -c "undefined" main.log || true)
rm -f "$D"/main.aux "$D"/main.log "$D"/main.out "$D"/main.pdf
tar -czf arxiv_v2.tar.gz -C "$D" .
echo "wrote $(pwd)/arxiv_v2.tar.gz"; tar -tzf arxiv_v2.tar.gz
