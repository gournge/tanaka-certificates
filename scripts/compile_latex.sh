#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

compile_log() {
  mkdir -p docs/research/pdf
  latexmk -pdf -cd -interaction=nonstopmode -halt-on-error \
    -outdir=../pdf -jobname=research-log docs/research/log/main.tex
}

compile_poster() {
  latexmk -pdf -cd -interaction=nonstopmode -halt-on-error \
    docs/research/poster/poster.tex
}

compile_weekly_reports() {
  mkdir -p docs/research/pdf/weekly-reports
  local report
  for report in docs/research/weekly-reports/*.tex; do
    latexmk -pdf -cd -interaction=nonstopmode -halt-on-error \
      -outdir=../pdf/weekly-reports "$report"
  done
}

compile_target() {
  case "$1" in
    all)
      compile_log
      compile_poster
      compile_weekly_reports
      ;;
    log)
      compile_log
      ;;
    poster)
      compile_poster
      ;;
    weekly)
      compile_weekly_reports
      ;;
    *)
      echo "Usage: $0 [all|log|poster|weekly] ..." >&2
      return 2
      ;;
  esac
}

if (($# == 0)); then
  set -- all
fi

for target in "$@"; do
  compile_target "$target"
done
