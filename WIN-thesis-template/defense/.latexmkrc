# Use XeLaTeX instead of pdflatex
$pdf_mode = 5;  # 5 means use xelatex

# Set XeLaTeX command
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Enable shell escape if needed (uncomment if you use minted or other packages requiring it)
# $xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -shell-escape %O %S';

