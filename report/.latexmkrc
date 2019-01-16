# Use xelatex instead of pdflatex
$pdf_mode = 5;
$dvi_mode = 0;
$postscript_mode = 0;
# minted requires -shell-escape
# do not ask the user on errors -interaction=nonstopmode
# make synctex files -synctex=1
$xelatex = "xelatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S";
@default_files = ('src/main.tex');

