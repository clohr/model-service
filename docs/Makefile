SOURCES := $(wildcard *.gv)
PNGS := $(SOURCES:.gv=.png)
SVGS := $(SOURCES:.gv=.svg)

all: $(PNGS) $(SVGS)

%.png : %.gv
	dot -Tpng $< -o $@

%.svg : %.gv
	dot -Tsvg $< -o $@

clean:
	rm -f *.svg *.png
