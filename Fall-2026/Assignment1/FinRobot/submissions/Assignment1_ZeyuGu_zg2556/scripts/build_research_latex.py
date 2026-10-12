"""Convert the existing research manuscript to a self-contained, editable LaTeX file.

No financial/model calls. Original Markdown, PDF and experiment outputs are read only.
Figures are redrawn as vector graphics from the exact saved measured CSV data.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/latex'
SOURCE = ROOT / 'docs/research_report.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(text):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
               '\u2013': '-', '\u2014': ' -- ', '\u2019': "'",
               '\u00d7': r'\ensuremath{\times}', '\u00b1': r'\ensuremath{\pm}'}
    return ''.join(mapping.get(c, c) for c in text)


def code(text):
    # Break long paths/hashes without changing their visible characters.
    pieces = re.split(r'(\s+)', text)
    result = []
    for word in pieces:
        if word.isspace():
            result.append(' ')
        else:
            result.append(r'\allowbreak{}'.join(escape(word[i:i+12]) for i in range(0, len(word), 12)))
    return r'\texttt{' + ''.join(result) + '}'


INLINE = re.compile(r'`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|\*([^*]+)\*')


def inline(text):
    out = []; last = 0
    for match in INLINE.finditer(text):
        out.append(escape(text[last:match.start()]))
        if match[1] is not None:
            out.append(code(match[1]))
        elif match[2] is not None:
            out.append(r'\href{' + match[3].replace('%', r'\%').replace('#', r'\#') + '}{' + escape(match[2]) + '}')
        elif match[4] is not None:
            out.append(r'\textbf{' + escape(match[4]) + '}')
        else:
            out.append(r'\emph{' + escape(match[5]) + '}')
        last = match.end()
    out.append(escape(text[last:]))
    return ''.join(out)


PREAMBLE = r'''% Standalone LaTeX: all chart coordinates are embedded below.
% Compile using pdfLaTeX (twice), XeLaTeX, LuaLaTeX, or Tectonic.
% No API key, local absolute path, external image, or bibliography file is required.
\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,left=20mm,right=20mm,top=22mm,bottom=22mm,headheight=14pt]{geometry}
\usepackage{microtype}
\usepackage{amsmath,array,longtable,booktabs}
\usepackage[table]{xcolor}
\usepackage{graphicx,caption,float,needspace}
\usepackage{tikz,pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\usepackage{fancyhdr}
\usepackage{hyperref}
\definecolor{ink}{HTML}{17344C}
\definecolor{linkblue}{HTML}{225B80}
\definecolor{tablehead}{HTML}{EAF0F5}
\definecolor{beforeblue}{HTML}{315E85}
\definecolor{afterteal}{HTML}{138C88}
\hypersetup{colorlinks=true,linkcolor=linkblue,urlcolor=linkblue,
 pdftitle={Can FinRobot support portfolio research?},pdfauthor={Zeyu Gu (zg2556)}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily FINROBOT / EVIDENCE AUDIT}
\fancyhead[R]{\small\sffamily Zeyu Gu | zg2556}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\linespread{1.09}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4pt}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\renewcommand{\arraystretch}{1.20}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}
\begin{document}
'''


def table(block, number):
    rows = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in block]
    align = rows[1]; rows = [rows[0]] + rows[2:]
    n = len(rows[0]); assert all(len(row) == n for row in rows)
    widths = {2: [0.23, 0.77], 3: [0.60, 0.20, 0.20],
              4: [0.24, 0.25, 0.255, 0.255], 5: [0.24, 0.19, 0.19, 0.19, 0.19]}[n]
    if number == 2: widths = [0.10, 0.15, 0.20, 0.55]
    if number == 13: widths = [0.27, 0.19, 0.18, 0.18, 0.18]
    if number == 17: widths = [0.35, 0.18, 0.15, 0.13, 0.19]
    cols = ''.join(('R' if a.endswith(':') else 'L') + '{\dimexpr ' + str(w) + r'\textwidth-' + str(2*n*w) + r'\tabcolsep\relax}' for w, a in zip(widths, align))
    header = r'\rowcolor{tablehead} ' + ' & '.join(r'\textbf{' + inline(c) + '}' for c in rows[0]) + r' \\'
    body = [r'\needspace{7\baselineskip}', r'{\fontsize{8.5}{10.5}\selectfont',
            '% Table '+str(number)+': exact original Markdown cells',
            r'\begin{longtable}{'+cols+'}', r'\toprule', header, r'\midrule', r'\endfirsthead',
            r'\toprule', header, r'\midrule', r'\endhead',
            r'\midrule \multicolumn{'+str(n)+r'}{r}{\footnotesize Continued on next page} \\',
            r'\endfoot', r'\bottomrule', r'\endlastfoot']
    for row in rows[1:]: body.append(' & '.join(inline(c) for c in row) + r' \\')
    body += [r'\end{longtable}', '}']
    return '\n'.join(body), rows


def figures():
    # The figure builder is part of this conversion, not a new experiment.
    from latex_result_figures import build_figures
    return build_figures(ROOT)


def main():
    text = SOURCE.read_text(encoding='utf-8-sig'); lines = text.splitlines()
    chart_tex, chart_inputs = figures()
    out = [PREAMBLE]; mapped = []; tables = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        start = i + 1
        heading = re.match(r'^(#{1,3})\s+(.*)', line)
        if heading:
            level, title = len(heading[1]), heading[2]
            if level == 1:
                rendered = r'{\color{ink}\sffamily\bfseries\fontsize{23}{27}\selectfont ' + inline(title) + r'\par}\vspace{6pt}'
            else:
                command = 'section' if level == 2 else 'subsection'
                rendered = '\\needspace{6\\baselineskip}\\'+command+'*{'+inline(title)+'}'
            kind='heading'; original=line; i += 1
        elif line.startswith('|'):
            block=[]
            while i < len(lines) and lines[i].strip().startswith('|'): block.append(lines[i]); i+=1
            rendered, rows = table(block, len(tables)+1); tables.append(rows)
            kind='table'; original='\n'.join(block)
        elif line.startswith('!['):
            match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)', line); assert match
            rendered=chart_tex['grounding']; kind='figure'; original=line; i+=1
        else:
            block=[]
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[)', lines[i].strip()):
                block.append(lines[i]); i+=1
            rendered=''
            for j, rawline in enumerate(block):
                rendered += inline(rawline.strip())
                if j < len(block)-1: rendered += (' ' + chr(92)*2 + '\n' if rawline.endswith('  ') else ' ')
            kind='paragraph'; original='\n'.join(block)
        entry={'kind':kind,'source_lines':[start,i],'source_text':original,
               'latex_sha256':hashlib.sha256(rendered.encode()).hexdigest()}
        mapped.append(entry)
        out += [f'% SOURCE-BLOCK {len(mapped)} | {kind} | Markdown lines {start}-{i}',rendered,'']
    out += [r'\clearpage\section*{Result figures}',
            'The following figures are generated from saved experiment results. These are scenario analyses and measured audits; no probability distribution or unperformed result is implied.',
            chart_tex['nvda'],r'\clearpage',chart_tex['avgo'],r'\end{document}']
    DEST.mkdir(parents=True,exist_ok=True)
    tex=DEST/'research_report.tex'; tex.write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n')
    manifest={'source':'docs/research_report.md','source_sha256':sha(SOURCE),
              'tex':'docs/latex/research_report.tex','tex_sha256':sha(tex),
              'converter_sha256':sha(Path(__file__)),'figure_builder_sha256':sha(Path(__file__).with_name('latex_result_figures.py')),
              'source_blocks':mapped,'table_count':len(tables),'table_body_rows':sum(len(t)-1 for t in tables),
              'table_cells':sum(sum(len(row) for row in t) for t in tables),
              'figure_count':3,'figure_inputs':chart_inputs,
              'original_report_pdf_sha256':sha(ROOT/'output/pdf/research_report.pdf'),
              'financial_or_model_calls':0,'scope':'Format conversion only; unchanged Markdown and experiments. Figures redrawn from saved CSV values. Compilation/visual review recorded separately.'}
    (DEST/'conversion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_blocks','figure_inputs']},indent=2))


if __name__=='__main__': main()
