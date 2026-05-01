# Paper C build — rebuild the manuscript PDF and all figures
#
# Usage:
#   make            # build paper_c.pdf (default)
#   make pdf        # same
#   make figures    # rebuild all six figures (PNG + PDF)
#   make all        # figures + paper PDF
#   make clean      # remove build artefacts (paper_c.pdf only)
#   make distclean  # also wipe rendered figure PDF/PNG
#
# Style is matched to paper/nsn.pdf (the modeq Paper B preprint):
# Times New Roman body, Menlo monospace, 0.9 in margins, no TOC.
#
# Requirements:
#   pandoc      (Homebrew: brew install pandoc)
#   xelatex     (TinyTex: ~/Library/TinyTeX/bin/universal-darwin/xelatex,
#                or full TeX Live)
#   Times New Roman + Menlo (preinstalled on macOS)
#   python3 + matplotlib + numpy   (for figures, see figures_c/Makefile)

PANDOC      := pandoc
PDF_ENGINE  := xelatex
MAINFONT    := Times New Roman
MONOFONT    := Menlo
MARGIN      := 0.9in

PAPER_MD    := paper_c.md
PAPER_PDF   := paper_c.pdf
FIG_DIR     := figures_c

.PHONY: all pdf figures clean distclean

all: figures pdf

pdf: $(PAPER_PDF)

$(PAPER_PDF): $(PAPER_MD) $(wildcard $(FIG_DIR)/*.pdf)
	$(PANDOC) $(PAPER_MD) -o $(PAPER_PDF) \
	    --pdf-engine=$(PDF_ENGINE) \
	    --resource-path=. \
	    -V geometry:margin=$(MARGIN) \
	    -V mainfont="$(MAINFONT)" \
	    -V monofont="$(MONOFONT)"

figures:
	$(MAKE) -C $(FIG_DIR) all

clean:
	rm -f $(PAPER_PDF)

distclean: clean
	$(MAKE) -C $(FIG_DIR) clean
