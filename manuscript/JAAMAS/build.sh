#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v pdflatex >/dev/null 2>&1; then
  source /home/ez/DelayedAdviceAudit/code/outputs/jaamas/local_tex_env.sh
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_current_pass1.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_current_pass2.log 2>&1
cp main.pdf Delayed_Advice_Audit_manuscript.pdf
(
  cd supplement
  export TEXINPUTS="..:${TEXINPUTS:-}"
  pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > supplement_pass1.log 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > supplement_pass2.log 2>&1
  cp supplement.pdf Delayed_Advice_Audit_Online_Resource_1.pdf
)
(
  cd submission
  pdflatex -interaction=nonstopmode -halt-on-error information_sheet.tex > information_build.log 2>&1
  cp information_sheet.pdf Delayed_Advice_Audit_Information_Sheet.pdf
  pdflatex -interaction=nonstopmode -halt-on-error cover_letter.tex > cover_build.log 2>&1
  cp cover_letter.pdf Delayed_Advice_Audit_Cover_Letter.pdf
)
