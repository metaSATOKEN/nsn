# Paper C figures — regenerate from results_c JSONs
#
# Usage:
#   make all     # rebuild every figure (PDF + PNG)
#   make fig04   # rebuild a single figure
#   make clean   # remove all output PDF/PNG
#
# Each fig*.py is self-contained: it loads the relevant JSON from
# paper_c/results_c/ via the _style.load_results helper, applies
# the shared matplotlib style, and writes both .pdf and .png next
# to itself.

PY := python3

FIGS := fig01_R_distribution \
        fig02_sigma_scaling \
        fig03_matched_mean_5sizes \
        fig04_factorisation \
        fig05_long_context \
        fig06_state_norm_trajectory

PDFS := $(addsuffix .pdf,$(FIGS))
PNGS := $(addsuffix .png,$(FIGS))

.PHONY: all clean $(FIGS)

all: $(PDFS)

# Each figure target rebuilds via its standalone .py
%.pdf %.png: %.py _style.py
	$(PY) $<

$(FIGS): %: %.pdf

clean:
	rm -f $(PDFS) $(PNGS)
