# How to Compile Your Paper in Overleaf

## Quick Start (3 Steps)

### Step 1: Create New Overleaf Project
1. Go to https://overleaf.com
2. Click "New Project" → "Blank Project"
3. Name it: "CodeX-Verify Paper"

### Step 2: Upload Files
You need 2 files:

**File 1: main.tex**
- Use `COMPLETE_PAPER.tex` as template
- Open `paper_title_abstract.tex`
- Copy content from line 56 (Section 1: Introduction) through line 2310 (End of Appendix)
- Paste into main.tex where it says `[CONTENT GOES HERE]`

**File 2: references.bib**
- Open `paper_title_abstract.tex`
- Find the `\begin{comment}` section (around line 1244)
- Copy all BibTeX entries (lines 1249-1703)
- Paste into new file `references.bib` in Overleaf

### Step 3: Compile
1. Click "Recompile" in Overleaf
2. If you get errors about missing packages, Overleaf will install them automatically
3. Compile sequence: pdflatex → bibtex → pdflatex → pdflatex
4. You should see an 18-19 page PDF

## What You Should See

**Page 1:** Title, Abstract, Introduction starts
**Pages 2-3:** Introduction + Related Work
**Pages 4-7:** Theoretical Framework (5 theorems)
**Pages 8-10:** System Design
**Pages 11-13:** Experimental Evaluation
**Pages 14-16:** Results (with tables and figures)
**Pages 17-18:** Discussion + Conclusion
**Page 19+:** References + Appendix

## File Structure in Overleaf

```
CodeX-Verify Paper/
├── main.tex          (COMPLETE_PAPER.tex + content from paper_title_abstract.tex)
└── references.bib    (BibTeX entries from paper_title_abstract.tex)
```

## Troubleshooting

**Problem:** References show as [?]
**Solution:** Compile sequence must be: pdflatex → bibtex → pdflatex → pdflatex

**Problem:** Missing package errors
**Solution:** Overleaf will auto-install. If not, add to preamble: `\usepackage{packagename}`

**Problem:** Figures don't show
**Solution:** ASCII figures in \begin{verbatim}...\end{verbatim} should render as monospace text

**Problem:** Math doesn't compile
**Solution:** Check all $ symbols are paired, and use \backslash for LaTeX commands

## Customization

**Change author:** Edit line 50 in COMPLETE_PAPER.tex
**Change affiliation:** Edit line 51
**Add funding:** Edit Acknowledgments section (around line 1711 in paper_title_abstract.tex)
**Add figures:** Replace ASCII diagrams with actual TikZ/includegraphics

## Next Steps After Compilation

1. **Review PDF:** Check all sections, tables, figures render correctly
2. **Update author info:** Add your name, affiliation, email
3. **Add acknowledgments:** Funding sources, collaborators
4. **For ArXiv:** Submit as-is (18-19 pages is fine)
5. **For ICML:** Trim to 8 pages (keep theory, compress system/eval)
6. **For ICSE:** Trim to 10 pages (keep practical, compress theory)

## Files Provided

- `COMPLETE_PAPER.tex` - LaTeX template with preamble
- `paper_title_abstract.tex` - All content (Introduction through Appendix)
- `OVERLEAF_INSTRUCTIONS.md` - This file

## Contact

If you have issues, the paper content is all in `paper_title_abstract.tex` starting from line 56.
