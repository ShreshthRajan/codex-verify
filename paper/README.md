# Paper Source

LaTeX source for "CodeX-Verify: Multi-Agent Verification of LLM-Generated Code"

## Main File

- `main.tex` - Complete paper (ready for Overleaf/ArXiv submission)

## Compiling

### In Overleaf:
1. Upload `main.tex`
2. Create `references.bib` (extract from comment block in main.tex)
3. Compile: pdflatex → bibtex → pdflatex → pdflatex

### Locally:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Archive

`archive/` contains previous versions and source files:
- `paper_title_abstract.tex` - Original development file
- `COMPLETE_PAPER.tex` - Template version
- Others - iteration history

## Current Stats

- **Pages:** ~24-28 (two-column format)
- **Sections:** 8 main sections + appendix
- **Tables:** 7
- **Figures:** 3 (ASCII diagrams)
- **References:** 40

## Submission Status

- [ ] ArXiv (cs.SE, cs.LG)
- [ ] ICML 2026
- [ ] ICSE 2026

## Contact

Shreshth Rajan - shreshthrajan@college.harvard.edu
