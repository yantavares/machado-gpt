# Artigo IEEE (duas colunas) do projeto Machado de Assis.
#
#   make           compila artigo.pdf (via latexmk)
#   make figuras   regenera paper/figuras/ a partir de results/
#   make paginas   mostra o número de páginas (limite: 6)
#   make clean     remove auxiliares, mantém o PDF
#   make distclean remove auxiliares e o PDF

MAIN := artigo
export TEXINPUTS := ../conference-latex-template/IEEE-conference-template-062824//:$(TEXINPUTS)
export BSTINPUTS := ../IEEEtranBST2//:$(BSTINPUTS)
export BIBINPUTS := ../IEEEtranBST2//:$(BIBINPUTS)

SOURCES := $(MAIN).tex referencias.bib $(wildcard figuras/*)

.PHONY: all figuras paginas clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	latexmk $(MAIN).tex

figuras:
	cd .. && python -m machado_gpt.graficos

paginas: $(MAIN).pdf
	@pdfinfo $(MAIN).pdf | grep Pages

clean:
	latexmk -c $(MAIN).tex

distclean:
	latexmk -C $(MAIN).tex
