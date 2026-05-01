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
# Requirements:
#   pandoc      (Homebrew: brew install pandoc)
#   xelatex     (TinyTex: ~/Library/TinyTeX/bin/universal-darwin/xelatex,
#                or full TeX Live)
#   STIX Two Text + STIX Two Math (preinstalled on macOS Sonoma+ /
#                Ventura, otherwise install via Font Book)
#   python3 + matplotlib + numpy   (for figures, see figures_c/Makefile)

PANDOC      := pandoc
PDF_ENGINE  := xelatex
MAINFONT    := STIX Two Text
MATHFONT    := STIX Two Math
MONOFONT    := Menlo

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
	    -V geometry:margin=1in \
	    -V mainfont="$(MAINFONT)" \
	    -V mathfont="$(MATHFONT)" \
	    -V monofont="$(MONOFONT)" \
	    --toc

figures:
	$(MAKE) -C $(FIG_DIR) all

clean:
	rm -f $(PAPER_PDF)

distclean: clean
	$(MAKE) -C $(FIG_DIR) clean
