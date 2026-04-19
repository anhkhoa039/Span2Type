"""
Generate a LaTeX report of GT labels vs Span2Type-LLM (MMR) cluster names
for all 5 CrossNER domains, parsed from BERT_Report/bertscore_report.tex.
"""

from collections import defaultdict

# ── Ground-truth entity types per CrossNER domain ─────────────────────────────

GT_TYPES = {
    "AI":         ["algorithm", "conference", "country", "field", "location",
                   "metrics", "misc", "organisation", "person", "product",
                   "programlang", "researcher", "task", "university"],
    "Literature": ["award", "book", "country", "event", "literarygenre",
                   "location", "magazine", "misc", "organisation", "person",
                   "poem", "writer"],
    "Music":      ["album", "award", "band", "country", "event", "location",
                   "misc", "musicalartist", "musicalinstrument", "musicgenre",
                   "organisation", "person", "song"],
    "Politics":   ["country", "election", "event", "location", "misc",
                   "organisation", "person", "politicalparty", "politician"],
    "Science":    ["academicjournal", "astronomicalobject", "award",
                   "chemicalcompound", "chemicalelement", "country",
                   "discipline", "enzyme", "event", "location", "misc",
                   "organisation", "person", "protein", "scientist",
                   "theory", "university"],
}

# ── Data extracted from BERT_Report/bertscore_report.tex ──────────────────────

DOMAINS = {
    "AI": {
        "mean": 0.9583,
        "clusters": [
            (0,  "algorithms",                "algorithm",      0.456, 0.9999),
            (1,  "fields",                    "product",        0.221, 0.9994),
            (2,  "organization",              "organisation",   0.606, 0.9994),
            (3,  "computer scientist",        "researcher",     0.699, 0.8693),
            (4,  "country",                   "country",        0.566, 1.0000),
            (5,  "programming language",      "product",        0.305, 0.8724),
            (6,  "conference",                "misc",           0.487, 0.9994),
            (7,  "information retrieval",     "task",           0.672, 0.8588),
            (8,  "metrics",                   "metrics",        0.770, 1.0000),
            (9,  "university",                "location",       0.474, 0.9803),
            (10, "metric",                    "metrics",        0.735, 0.9990),
            (11, "artificial intelligence",   "product",        0.227, 0.8390),
            (12, "software",                  "product",        0.560, 0.9994),
            (13, "field",                     "field",          0.747, 1.0000),
        ],
    },
    "Literature": {
        "mean": 0.9787,
        "clusters": [
            (0,  "book",          "book",           0.295, 1.0000),
            (1,  "author",        "writer",         0.754, 0.9983),
            (2,  "language",      "misc",           0.825, 0.9986),
            (3,  "theatre",       "location",       0.724, 0.9789),
            (4,  "academy",       "organisation",   0.589, 0.9982),
            (5,  "mythology",     "book",           0.364, 0.9993),
            (6,  "novel",         "book",           0.308, 0.9992),
            (7,  "novels",        "book",           0.610, 0.9995),
            (8,  "film festival", "event",          0.733, 0.8766),
            (9,  "country",       "country",        0.515, 1.0000),
            (10, "author",        "writer",         0.760, 0.9983),
            (11, "award",         "award",          0.945, 1.0000),
            (12, "novel",         "book",           0.686, 0.9992),
            (13, "literary work", "literarygenre",  0.748, 0.8552),
        ],
    },
    "Music": {
        "mean": 0.9391,
        "clusters": [
            (0,  "organization",  "organisation",   0.821, 0.9994),
            (1,  "albums",        "album",          0.701, 0.9996),
            (2,  "musician",      "musicalartist",  0.787, 0.8673),
            (3,  "country",       "country",        0.570, 1.0000),
            (4,  "music genres",  "musicgenre",     0.773, 0.8693),
            (5,  "award",         "award",          0.981, 1.0000),
            (6,  "musician",      "musicalartist",  0.921, 0.8673),
            (7,  "music",         "musicgenre",     0.905, 0.8617),
            (8,  "theatres",      "location",       0.953, 0.8413),
            (9,  "record label",  "organisation",   0.509, 0.8457),
            (10, "band",          "band",           0.702, 1.0000),
            (11, "ethnicity",     "misc",           0.627, 0.9995),
            (12, "albums",        "album",          0.588, 0.9996),
            (13, "band",          "band",           0.921, 1.0000),
            (14, "band",          "band",           0.721, 1.0000),
            (15, "awards",        "award",          0.427, 0.9999),
            (16, "song",          "song",           0.673, 1.0000),
            (17, "singer",        "musicalartist",  0.676, 0.8708),
            (18, "genre",         "musicgenre",     0.369, 0.8534),
            (19, "music genre",   "album",          0.286, 0.8201),
            (20, "theatre",       "location",       0.814, 0.9789),
            (21, "album",         "song",           0.268, 0.9867),
        ],
    },
    "Politics": {
        "mean": 0.9639,
        "clusters": [
            (0,  "city",              "location",       0.959, 0.9799),
            (1,  "artist",            "politicalparty", 0.382, 0.8670),
            (2,  "election",          "election",       0.990, 1.0000),
            (3,  "actor",             "person",         0.937, 0.9991),
            (4,  "party",             "politicalparty", 0.598, 0.8785),
            (5,  "nationality",       "misc",           0.896, 0.9995),
            (6,  "city",              "location",       0.988, 0.9799),
            (7,  "organization",      "organisation",   0.908, 0.9994),
            (8,  "allies",            "event",          0.444, 0.9458),
            (9,  "empire",            "country",        0.878, 0.9994),
            (10, "war",               "event",          0.796, 0.9469),
            (11, "political party",   "politicalparty", 0.805, 0.9304),
            (12, "historical figure", "person",         0.469, 0.8508),
            (13, "university",        "organisation",   0.774, 0.9993),
            (14, "country",           "country",        0.734, 1.0000),
            (15, "party",             "politicalparty", 0.953, 0.8785),
            (16, "political parties", "politicalparty", 0.485, 0.9185),
            (17, "politician",        "politician",     0.880, 1.0000),
            (18, "country",           "country",        0.755, 1.0000),
            (19, "politician",        "politician",     0.704, 1.0000),
            (20, "election",          "election",       0.646, 1.0000),
            (21, "city",              "location",       0.941, 0.9799),
            (22, "country",           "location",       0.600, 0.9802),
            (23, "politician",        "politician",     0.735, 1.0000),
        ],
    },
    "Science": {
        "mean": 0.9499,
        "clusters": [
            (0,  "thing",                   "misc",               0.687, 0.9991),
            (1,  "astronomer",              "scientist",          0.813, 0.9970),
            (2,  "protein",                 "enzyme",             0.410, 0.9996),
            (3,  "university",              "university",         0.806, 1.0000),
            (4,  "elements",                "chemicalelement",    0.848, 0.8978),
            (5,  "language",                "misc",               0.708, 0.9986),
            (6,  "professional organization","organisation",      0.862, 0.8957),
            (7,  "asteroid",                "astronomicalobject", 0.890, 0.8657),
            (8,  "continents",              "location",           0.756, 0.9788),
            (9,  "journal",                 "misc",               0.243, 0.9991),
            (10, "field",                   "discipline",         0.473, 0.9940),
            (11, "molecule",                "chemicalcompound",   0.752, 0.8385),
            (12, "planet",                  "astronomicalobject", 0.915, 0.8666),
            (13, "biological process",      "misc",               0.862, 0.8804),
            (14, "scientist",               "scientist",          0.722, 1.0000),
            (15, "journal",                 "academicjournal",    0.841, 0.8409),
            (16, "historian",               "scientist",          0.868, 0.9989),
            (17, "protein",                 "misc",               0.386, 0.9993),
            (18, "actor",                   "person",             0.972, 0.9991),
            (19, "building",                "location",           0.692, 0.9819),
            (20, "awards",                  "award",              0.956, 0.9999),
            (21, "planet",                  "astronomicalobject", 0.942, 0.8666),
        ],
    },
}

# ── LaTeX generation ───────────────────────────────────────────────────────────

def f1_color(f1):
    if f1 >= 0.99:
        return r"\cellcolor{green!15}"
    elif f1 >= 0.90:
        return r"\cellcolor{yellow!25}"
    else:
        return r"\cellcolor{red!15}"

def normalize_name(name):
    """Singularize the last word: 'novels' → 'novel', 'political parties' → 'political party'."""
    words = name.split()
    last = words[-1]
    if last.endswith("ies") and len(last) > 4:
        last = last[:-3] + "y"
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        last = last[:-1]
    words[-1] = last
    return " ".join(words)


def gt_mapping_table(name, data):
    gt_types = sorted(GT_TYPES[name])
    gen_names = sorted(set(normalize_name(pred) for _, pred, _, _, _ in data["clusters"]))

    n = max(len(gt_types), len(gen_names))
    gt_padded  = gt_types  + [""] * (n - len(gt_types))
    gen_padded = gen_names + [""] * (n - len(gen_names))

    rows = []
    for gt, gen in zip(gt_padded, gen_padded):
        gt_cell  = rf"\texttt{{{gt}}}"  if gt  else ""
        gen_cell = rf"\textit{{{gen}}}" if gen else ""
        rows.append(rf"  {gt_cell} & {gen_cell} \\")
    body = "\n".join(rows)

    return rf"""
\begin{{table}}[ht]
\centering
\footnotesize
\caption{{Ground-truth entity types vs.\ LLM-generated cluster names --- {name} domain
({len(gt_types)} GT types, {len(gen_names)} unique generated names).}}
\label{{tab:gt_map_{name.lower()}}}
\setlength{{\tabcolsep}}{{8pt}}
\begin{{tabular}}{{l l}}
\toprule
\textbf{{Ground-Truth Entity Types}} & \textbf{{Generated Cluster Names (LLM)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}

\FloatBarrier
"""


def domain_table(name, data):
    n = len(data["clusters"])
    mean = data["mean"]
    rows = []
    for cid, pred, gt, purity, f1 in data["clusters"]:
        color = f1_color(f1)
        rows.append(
            f"  {cid:>2} & {pred:<30} & {gt:<22} & {purity:.3f} & {color}{f1:.4f} \\\\"
        )
    body = "\n".join(rows)
    return rf"""
\subsection{{{name} Domain ({n} clusters, mean $F_1 = {mean}$)}}

\begin{{table}}[ht]
\centering
\small
\caption{{Span2Type-LLM (MMR) cluster naming results --- {name} domain.
\colorbox{{green!15}}{{green}} $F_1 \geq 0.99$,
\colorbox{{yellow!25}}{{yellow}} $0.90 \leq F_1 < 0.99$,
\colorbox{{red!15}}{{red}} $F_1 < 0.90$.}}
\label{{tab:naming_{name.lower()}}}
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{r l l r r}}
\toprule
\textbf{{C\#}} & \textbf{{Predicted Name}} & \textbf{{GT Label}} & \textbf{{Purity}} & \textbf{{$F_1$}} \\
\midrule
{body}
\midrule
\multicolumn{{3}}{{l}}{{\textbf{{Mean}}}} & & \textbf{{{mean:.4f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\FloatBarrier
"""

# ── Overall summary table ──────────────────────────────────────────────────────

def summary_table():
    rows = []
    total_clusters = 0
    weighted_sum = 0.0
    for dname, data in DOMAINS.items():
        n = len(data["clusters"])
        mean = data["mean"]
        perfect = sum(1 for _, _, _, _, f1 in data["clusters"] if f1 >= 0.990)
        low = sum(1 for _, _, _, _, f1 in data["clusters"] if f1 < 0.90)
        rows.append(
            rf"  {dname:<12} & {n} & {mean:.4f} & {perfect} & {low} \\"
        )
        total_clusters += n
        weighted_sum += mean * n
    overall = weighted_sum / total_clusters
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{Summary of Span2Type-LLM (MMR) cluster naming quality across all CrossNER domains
(BERTScore-F1, RoBERTa-large). \textit{{Perfect}} $= F_1 \geq 0.99$; \textit{{Low}} $= F_1 < 0.90$.}}
\label{{tab:naming_summary}}
\begin{{tabular}}{{lrrrr}}
\toprule
\textbf{{Domain}} & \textbf{{\#Clusters}} & \textbf{{Mean $F_1$}} & \textbf{{Perfect}} & \textbf{{Low}} \\
\midrule
{body}
\midrule
  \textbf{{Overall}} & {total_clusters} & \textbf{{{overall:.4f}}} & & \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""

# ── Full document ──────────────────────────────────────────────────────────────

doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{caption}
\usepackage{placeins}
\usepackage{hyperref}
\usepackage{microtype}

\title{\textbf{Cluster Naming Results Report}\\[0.4em]
\large Span2Type-LLM (MMR) --- Ground-Truth vs.\ Predicted Names\\
CrossNER: All Five Domains}
\author{Span2Type Research}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Overview}

This report presents the complete per-cluster naming results of \textbf{Span2Type-LLM (MMR)}
across all five CrossNER target domains: AI, Literature, Music, Politics, and Science.

For each cluster the table shows:
\begin{itemize}
  \item \textbf{Predicted Name} --- the type label generated by Llama-3.1-8B (Ollama)
        from $n{=}16$ MMR-selected entity exemplars.
  \item \textbf{GT Label} --- the ground-truth CrossNER entity type that contributes
        the most entities to the cluster (column-wise argmax of the confusion matrix).
  \item \textbf{Purity} --- fraction of cluster entities belonging to the GT label.
  \item \textbf{$F_1$} --- BERTScore-F1 (RoBERTa-large) between predicted name and GT label.
\end{itemize}

Cell colour: \colorbox{green!15}{green} $F_1 \geq 0.99$ (near-perfect),
\colorbox{yellow!25}{yellow} $0.90 \leq F_1 < 0.99$ (good),
\colorbox{red!15}{red} $F_1 < 0.90$ (poor).

\section{Summary}
"""

doc += summary_table()

doc += r"""
\section{Per-Domain Results}
"""

for dname, data in DOMAINS.items():
    doc += gt_mapping_table(dname, data)
    doc += domain_table(dname, data)

doc += r"""
\section{Failure Analysis}

Three recurring failure patterns account for most low-scoring clusters ($F_1 < 0.90$):

\begin{enumerate}
  \item \textbf{Compound GT labels} (Music, Science): CrossNER uses concatenated labels
        such as \texttt{musicalartist}, \texttt{musicgenre}, \texttt{chemicalcompound},
        \texttt{astronomicalobject}, and \texttt{academicjournal}.
        RoBERTa tokenises these differently from the plain-English predicted names
        (\textit{musician}, \textit{music genres}, \textit{molecule}, \textit{planet},
        \textit{journal}), reducing BERTScore even when the meaning is correct.

  \item \textbf{Specificity mismatch} (AI, Literature, Science): The LLM generates a
        more specific or more general label than the GT label
        (e.g.\ \textit{computer scientist} vs.\ \texttt{researcher};
        \textit{film festival} vs.\ \texttt{event};
        \textit{professional organization} vs.\ \texttt{organisation}).

  \item \textbf{Mixed clusters} (AI, Politics): Low-purity clusters whose plurality
        GT label does not reflect the dominant entity concept
        (e.g.\ \textit{conference} $\to$ \texttt{misc} in AI-C6;
        \textit{artist} $\to$ \texttt{politicalparty} in Politics-C1).
\end{enumerate}

\end{document}
"""

out_path = "reports/naming_results_report.tex"
with open(out_path, "w") as f:
    f.write(doc)

print(f"Written: {out_path}")
print(f"Total clusters: {sum(len(d['clusters']) for d in DOMAINS.values())}")
for dname, data in DOMAINS.items():
    print(f"  {dname}: {len(data['clusters'])} clusters, mean F1={data['mean']:.4f}")
