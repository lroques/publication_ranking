from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import html
import asyncio
from collections import Counter, defaultdict
import re

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Author Profile Analysis (OpenAlex)", version="4.4")

OPENALEX = "https://api.openalex.org"

TOP_PAPERS = 30
TOP_COAUTHORS = 30

MAX_AUTHORS_DISPLAY = 12


def current_year() -> int:
    return datetime.now().year


# ----------------------------
# OpenAlex: works + author search
# ----------------------------
async def fetch_openalex_works(author_id: str, per_page: int = 200) -> List[Dict[str, Any]]:
    """
    author_id: full OpenAlex author id like 'https://openalex.org/A123...'
               or short like 'A123...'
    Returns raw works from OpenAlex (with authorships, type, etc.).
    """
    if author_id.startswith("http"):
        author = author_id
    else:
        author = f"https://openalex.org/{author_id}"

    works: List[Dict[str, Any]] = []
    cursor = "*"

    async with httpx.AsyncClient(timeout=50) as client:
        while True:
            url = f"{OPENALEX}/works"
            params = {
                "filter": f"authorships.author.id:{author}",
                "per-page": str(per_page),
                "cursor": cursor,
            }
            r = await client.get(url, params=params, headers={"User-Agent": "AuthorProfileAnalysis/4.4"})
            if r.status_code != 200:
                raise HTTPException(502, detail=f"OpenAlex error: {r.status_code} {r.text[:200]}")

            data = r.json()
            results = data.get("results", [])
            works.extend(results)

            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor

            if len(works) > 15000:
                break

    return works


def extract_authors_list(raw_work: Dict[str, Any], max_authors: int = MAX_AUTHORS_DISPLAY) -> str:
    """
    Extract authors display names from OpenAlex raw 'authorships'.
    Return a compact string, e.g. 'A. Smith, B. Jones, ...'
    """
    authorships = raw_work.get("authorships") or []
    names = []
    for a in authorships:
        author = (a.get("author") or {})
        name = author.get("display_name")
        if name:
            names.append(name)

    if not names:
        return ""

    if len(names) <= max_authors:
        return ", ".join(names)

    return ", ".join(names[:max_authors]) + ", et al."


def normalize_work(w: Dict[str, Any]) -> Dict[str, Any]:
    primary_location = w.get("primary_location") or {}
    source = primary_location.get("source") or {}

    y = w.get("publication_year")
    cites = int(w.get("cited_by_count") or 0)

    return {
        "id": w.get("id"),
        "doi": (w.get("doi") or "").replace("https://doi.org/", "") if w.get("doi") else None,
        "title": w.get("title"),
        "authors": extract_authors_list(w),
        "year": y,
        "citations": cites,
        "primary_location": source.get("display_name"),
        "url": w.get("id"),
        "type": w.get("type"),
    }


async def compute_author_concepts(
    client: httpx.AsyncClient,
    author_id: str,
    top_works: int = 20,
    top_concepts: int = 3,
) -> List[str]:
    """
    Compute reliable concepts for an author by aggregating concepts
    on their most cited works (helps disambiguation).
    """
    url = f"{OPENALEX}/works"
    params = {
        "filter": f"authorships.author.id:{author_id}",
        "per-page": str(min(top_works, 200)),
        "sort": "cited_by_count:desc",
    }
    r = await client.get(url, params=params, headers={"User-Agent": "AuthorProfileAnalysis/4.4"})
    if r.status_code != 200:
        return []

    data = r.json()
    works = data.get("results", []) or []

    scores: Dict[str, float] = {}

    for w in works:
        cited = float(w.get("cited_by_count") or 0.0)
        concepts = w.get("concepts") or []

        for c in concepts:
            name = c.get("display_name")
            if not name:
                continue
            level = c.get("level")
            # Skip broad concepts (level 0/1)
            if isinstance(level, int) and level <= 1:
                continue
            scores[name] = scores.get(name, 0.0) + 1.0 + 0.0005 * cited

    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in best[:top_concepts]]


async def search_openalex_authors(name: str, per_page: int = 8) -> List[Dict[str, Any]]:
    """
    Search OpenAlex authors and enrich each candidate with reliable concepts.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        url = f"{OPENALEX}/authors"
        params = {"search": name, "per-page": str(per_page)}
        r = await client.get(url, params=params, headers={"User-Agent": "AuthorProfileAnalysis/4.4"})
        if r.status_code != 200:
            raise HTTPException(502, detail=f"OpenAlex error: {r.status_code} {r.text[:200]}")
        data = r.json()
        results = data.get("results", [])

    out: List[Dict[str, Any]] = []
    for a in results:
        out.append({
            "id": a.get("id"),
            "display_name": a.get("display_name"),
            "works_count": a.get("works_count"),
            "cited_by_count": a.get("cited_by_count"),
            "orcid": (a.get("orcid") or "").replace("https://orcid.org/", "") if a.get("orcid") else None,
            "last_known_institution": (a.get("last_known_institution") or {}).get("display_name"),
            "concepts": [],
        })

    top_works_per_author = 20
    top_concepts_per_author = 3

    async with httpx.AsyncClient(timeout=40) as client:
        tasks = [
            compute_author_concepts(
                client,
                author_id=item["id"],
                top_works=top_works_per_author,
                top_concepts=top_concepts_per_author,
            )
            for item in out
            if item.get("id")
        ]
        concepts_list = await asyncio.gather(*tasks, return_exceptions=True)

    idx = 0
    for item in out:
        if not item.get("id"):
            continue
        c = concepts_list[idx]
        idx += 1
        item["concepts"] = [] if isinstance(c, Exception) else c

    return out


# ----------------------------
# Citation rate ranking
# ----------------------------
def citation_rate(citations: int, year: int, now_year: int) -> float:
    age = max(0, now_year - year)
    return citations / (1 + age)


def compute_rate_ranking(works: List[Dict[str, Any]], top_n: int = TOP_PAPERS) -> List[Dict[str, Any]]:
    now = current_year()
    rated = []

    for w in works:
        y = w.get("year")
        if not isinstance(y, int):
            continue
        cites = int(w.get("citations") or 0)
        rate = citation_rate(cites, y, now)
        item = dict(w)
        item["citation_rate"] = rate
        rated.append(item)

    rated.sort(
        key=lambda x: (
            x.get("citation_rate", 0.0),
            x.get("citations", 0),
            x.get("year", 0),
            (x.get("title") or ""),
        ),
        reverse=True,
    )
    return rated[:top_n]


# ----------------------------
# Co-author ranking (ARTICLES ONLY + MERGE BY NAME)
# ----------------------------
def _canonical_author_id(openalex_author_id: str) -> str:
    if openalex_author_id.startswith("http"):
        return openalex_author_id
    return f"https://openalex.org/{openalex_author_id}"


def is_published_article(raw_work: Dict[str, Any]) -> bool:
    return (raw_work.get("type") or "").lower().strip() == "article"


def normalize_person_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_coauthor_ranking_articles_only_merge_by_name(
    raw_works: List[Dict[str, Any]],
    main_author_id: str,
    top_n: int = TOP_COAUTHORS
) -> Tuple[List[Dict[str, Any]], int]:
    main_id = _canonical_author_id(main_author_id)

    name_counts: Counter[str] = Counter()
    name_display: Dict[str, str] = {}
    name_ids: Dict[str, Counter[str]] = defaultdict(Counter)

    article_works = [w for w in raw_works if is_published_article(w)]

    for w in article_works:
        authorships = w.get("authorships") or []
        for a in authorships:
            author = (a.get("author") or {})
            aid = author.get("id")
            if not aid or aid == main_id:
                continue

            disp = author.get("display_name") or "Unknown"
            key = normalize_person_name(disp)
            if not key:
                continue

            name_counts[key] += 1
            name_ids[key][aid] += 1

            if key not in name_display:
                name_display[key] = disp
            else:
                if len(disp) > len(name_display[key]):
                    name_display[key] = disp

    items: List[Dict[str, Any]] = []
    for key, c in name_counts.items():
        best_id = None
        if key in name_ids and name_ids[key]:
            best_id = name_ids[key].most_common(1)[0][0]

        items.append({
            "name_key": key,
            "name": name_display.get(key, "Unknown"),
            "count": c,
            "author_id": best_id,
            "merged_ids": len(name_ids[key]) if key in name_ids else 0,
        })

    items.sort(key=lambda x: (x["count"], x["name"]), reverse=True)
    return items[:top_n], len(article_works)


# ----------------------------
# Rendering helpers
# ----------------------------
def work_cell_html(w: Dict[str, Any]) -> str:
    title = html.escape(w.get("title") or "Untitled")
    authors = html.escape(w.get("authors") or "")
    url = w.get("url") or ""
    doi = w.get("doi")

    links = []
    if url:
        links.append(f'<a href="{html.escape(url)}" target="_blank" rel="noopener">OpenAlex</a>')
    if doi:
        links.append(f'<a href="https://doi.org/{html.escape(doi)}" target="_blank" rel="noopener">DOI</a>')

    links_html = " · ".join(links) if links else ""
    venue = html.escape(w.get("primary_location") or "")
    year = w.get("year") or ""
    cites = w.get("citations", 0)
    wtype = html.escape((w.get("type") or "").replace("_", " "))

    meta = f"{year} — {cites} citations"
    if venue:
        meta += f" — {venue}"
    if wtype:
        meta += f" — {wtype}"

    authors_line = f'<div class="authors">Authors: {authors}</div>' if authors else ""

    return f"""
    <div class="title">{title}</div>
    {authors_line}
    <div class="meta">{meta}</div>
    <div class="links">{links_html}</div>
    """


def format_rate(x: float) -> str:
    return f"{x:.2f}"


def author_cell_html(author_id: Optional[str], name: str, merged_ids: int) -> str:
    safe_name = html.escape(name or "Unknown")
    if author_id:
        safe_id = html.escape(author_id)
        link = f'<a href="{safe_id}" target="_blank" rel="noopener">{safe_name}</a>'
        extra = f" · merged IDs: {merged_ids}" if merged_ids and merged_ids > 1 else ""
        return f"""
        <div class="title">{link}</div>
        <div class="meta">{safe_id}{extra}</div>
        """
    else:
        extra = f" · merged IDs: {merged_ids}" if merged_ids and merged_ids > 1 else ""
        return f"""
        <div class="title">{safe_name}</div>
        <div class="meta">No OpenAlex link{extra}</div>
        """


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "OK",
        "endpoints": ["/author_search", "/ranking_html", "/ranking", "/docs"],
        "source": "openalex",
        "tools": [
            f"top_papers_by_citation_rate (top {TOP_PAPERS})",
            f"top_coauthors_by_joint_articles (top {TOP_COAUTHORS})",
        ],
    }


@app.get("/author_search")
async def author_search(
    name: str = Query(..., min_length=2),
    per_page: int = Query(8, ge=1, le=25),
):
    results = await search_openalex_authors(name, per_page=per_page)
    return JSONResponse({"query": name, "source": "openalex", "results": results})


@app.get("/ranking")
async def ranking(
    author_id: str = Query(..., description="OpenAlex Author ID like A123... or full URL"),
):
    raw = await fetch_openalex_works(author_id)
    works = [normalize_work(w) for w in raw if w.get("publication_year") is not None]

    paper_ranking = compute_rate_ranking(works, top_n=TOP_PAPERS)
    coauthor_ranking, counted_articles = compute_coauthor_ranking_articles_only_merge_by_name(
        raw, main_author_id=author_id, top_n=TOP_COAUTHORS
    )

    return {
        "source": "openalex",
        "author_id": author_id,
        "count_works_total": len(raw),
        "count_works_with_year": len(works),
        "current_year": current_year(),
        "paper_ranking": {
            "top_n": TOP_PAPERS,
            "formula": "citations / (1 + current_year - year)",
            "items": paper_ranking,
        },
        "coauthor_ranking": {
            "top_n": TOP_COAUTHORS,
            "scope": "articles_only (type == 'article')",
            "counted_article_works": counted_articles,
            "items": coauthor_ranking,
        },
    }


@app.get("/ranking_html", response_class=HTMLResponse)
async def ranking_html(
    author_id: Optional[str] = Query(None, description="OpenAlex Author ID like A123... or full URL"),
    tab: str = Query("papers", description="papers|coauthors"),
):
    author_display = "-"
    papers_table_html = "<div class='empty'>Search an author above, then click “Use” to display analysis.</div>"
    coauthors_table_html = "<div class='empty'>Search an author above, then click “Use” to display analysis.</div>"

    if author_id:
        raw = await fetch_openalex_works(author_id)
        works = [normalize_work(w) for w in raw if w.get("publication_year") is not None]

        # Tab 1: papers
        ranked_papers = compute_rate_ranking(works, top_n=TOP_PAPERS)
        paper_rows = []
        for i, w in enumerate(ranked_papers, start=1):
            rate = float(w.get("citation_rate") or 0.0)
            paper_rows.append(
                f"""
                <tr>
                  <td class="rank">{i}</td>
                  <td class="work">{work_cell_html(w)}</td>
                  <td class="score">{format_rate(rate)}</td>
                </tr>
                """
            )
        papers_table_html = f"""
        <div class="card">
          <div class="cardhead">
            <div class="cardtitle">Top {TOP_PAPERS} papers by citation rate</div>
            <div class="cardsub">Score = <code>citations / (1 + current_year - year)</code> · current_year = {current_year()}</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Paper</th>
                <th>Citation rate</th>
              </tr>
            </thead>
            <tbody>
              {"".join(paper_rows)}
            </tbody>
          </table>
        </div>
        """

        # Tab 2: coauthors (articles only)
        ranked_coauthors, counted_articles = compute_coauthor_ranking_articles_only_merge_by_name(
            raw, main_author_id=author_id, top_n=TOP_COAUTHORS
        )

        coauthor_rows = []
        for i, a in enumerate(ranked_coauthors, start=1):
            coauthor_rows.append(
                f"""
                <tr>
                  <td class="rank">{i}</td>
                  <td class="work">{author_cell_html(a.get("author_id"), a.get("name"), a.get("merged_ids", 0))}</td>
                  <td class="score">{a.get("count", 0)}</td>
                </tr>
                """
            )
        coauthors_table_html = f"""
        <div class="card">
          <div class="cardhead">
            <div class="cardtitle">Top {TOP_COAUTHORS} co-authors by number of joint works</div>
            <div class="cardsub">Computed on {counted_articles} published articles only (type = <code>article</code>).</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Co-author</th>
                <th>Joint works</th>
              </tr>
            </thead>
            <tbody>
              {"".join(coauthor_rows)}
            </tbody>
          </table>
        </div>
        """

        author_display = html.escape(author_id)

    init_tab = "coauthors" if (tab or "").lower().strip() == "coauthors" else "papers"

    page = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Author profile analysis</title>
      <style>
        :root {{
          --bg: #0b1020;
          --text: #e9ecf5;
          --muted: #aab3d3;
          --row1: rgba(255,255,255,0.04);
          --row2: rgba(255,255,255,0.02);
          --border: rgba(255,255,255,0.08);
          --accent: #7aa2ff;
          --tab: rgba(255,255,255,0.07);
          --tabActive: rgba(122,162,255,0.20);
        }}
        body {{
          margin: 0;
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
          background: radial-gradient(1200px 600px at 20% 0%, rgba(122,162,255,0.25), transparent 60%),
                      radial-gradient(900px 500px at 100% 0%, rgba(126,255,191,0.10), transparent 55%),
                      var(--bg);
          color: var(--text);
        }}
        .wrap {{
          max-width: 1150px;
          margin: 40px auto;
          padding: 0 16px;
        }}
        .header {{
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 14px;
          flex-wrap: wrap;
        }}
        h1 {{
          font-size: 22px;
          margin: 0;
          letter-spacing: 0.2px;
        }}
        .sub {{
          color: var(--muted);
          font-size: 13px;
          line-height: 1.35;
        }}

        .search {{
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 14px 14px;
          margin-bottom: 16px;
          background: rgba(255,255,255,0.03);
          box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }}
        .searchbar {{
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          align-items: center;
        }}
        input[type="text"] {{
          flex: 1;
          min-width: 240px;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: rgba(0,0,0,0.25);
          color: var(--text);
          outline: none;
        }}
        button {{
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: rgba(122,162,255,0.15);
          color: var(--text);
          cursor: pointer;
          font-weight: 650;
        }}
        button:hover {{
          background: rgba(122,162,255,0.25);
        }}
        .results {{
          margin-top: 12px;
          display: grid;
          grid-template-columns: 1fr;
          gap: 10px;
        }}
        .result {{
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 10px 12px;
          background: rgba(0,0,0,0.18);
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: center;
        }}
        .rmain {{
          display: flex;
          flex-direction: column;
          gap: 4px;
        }}
        .rname {{
          font-weight: 750;
        }}
        .rmeta {{
          color: var(--muted);
          font-size: 12.5px;
          line-height: 1.25;
        }}
        .use {{
          padding: 8px 10px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: rgba(61,220,151,0.10);
          white-space: nowrap;
        }}
        .use:hover {{
          background: rgba(61,220,151,0.18);
        }}

        /* Tabs */
        .tabs {{
          display: flex;
          gap: 10px;
          margin: 12px 0 14px 0;
        }}
        .tab {{
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid var(--border);
          background: var(--tab);
          cursor: pointer;
          font-weight: 750;
          font-size: 13px;
          color: var(--text);
          user-select: none;
        }}
        .tab.active {{
          background: var(--tabActive);
        }}
        .panel {{
          display: none;
        }}
        .panel.active {{
          display: block;
        }}

        .card {{
          background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
          border: 1px solid var(--border);
          border-radius: 18px;
          overflow: hidden;
          box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}
        .cardhead {{
          padding: 14px 16px;
          border-bottom: 1px solid var(--border);
          background: rgba(0,0,0,0.15);
        }}
        .cardtitle {{
          font-weight: 800;
          font-size: 14px;
          letter-spacing: 0.02em;
        }}
        .cardsub {{
          margin-top: 6px;
          color: var(--muted);
          font-size: 12.5px;
          line-height: 1.35;
        }}
        code {{
          background: rgba(0,0,0,0.25);
          padding: 2px 6px;
          border-radius: 8px;
          border: 1px solid var(--border);
          color: var(--text);
        }}

        table {{
          width: 100%;
          border-collapse: collapse;
        }}
        thead th {{
          text-align: left;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--muted);
          padding: 14px 16px;
          border-bottom: 1px solid var(--border);
          background: rgba(0,0,0,0.10);
        }}
        tbody tr:nth-child(odd) {{
          background: var(--row1);
        }}
        tbody tr:nth-child(even) {{
          background: var(--row2);
        }}
        td {{
          vertical-align: top;
          padding: 14px 16px;
          border-bottom: 1px solid var(--border);
        }}
        td.rank {{
          width: 70px;
          font-weight: 800;
          font-size: 15px;
          color: #ffffff;
          white-space: nowrap;
        }}
        td.score {{
          width: 160px;
          font-weight: 800;
          font-size: 14px;
          color: #ffffff;
          white-space: nowrap;
          text-align: right;
        }}

        .title {{
          font-weight: 650;
          font-size: 14px;
          line-height: 1.35;
          margin-bottom: 4px;
        }}
        .authors {{
          color: var(--muted);
          font-size: 12.5px;
          line-height: 1.35;
          margin-bottom: 6px;
        }}
        .meta {{
          color: var(--muted);
          font-size: 12.5px;
          line-height: 1.35;
          margin-bottom: 6px;
        }}
        a {{
          color: var(--accent);
          text-decoration: none;
        }}
        a:hover {{
          text-decoration: underline;
        }}
        .links {{
          font-size: 12.5px;
        }}
        .empty {{
          color: var(--muted);
          padding: 18px;
          border: 1px dashed var(--border);
          border-radius: 18px;
          text-align: center;
        }}
        .footer {{
          color: var(--muted);
          font-size: 12px;
          margin-top: 12px;
          text-align: right;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="header">
          <div>
            <h1>Author profile analysis</h1>
            <div class="sub">Tools: Top papers by citation rate, Top co-authors (published articles only)</div>
          </div>
          <div class="sub">Author: {author_display} · source: OpenAlex</div>
        </div>

        <div class="search">
          <div class="searchbar">
            <input id="q" type="text" placeholder="Search an author (e.g., Terence Tao, Samuel Soubeyrand, ...)" />
            <button onclick="doSearch()">Search</button>
          </div>
          <div id="status" class="sub" style="margin-top:8px;"></div>
          <div id="results" class="results"></div>
        </div>

        <div class="tabs">
          <div id="tab-papers" class="tab" onclick="showTab('papers')">Top papers</div>
          <div id="tab-coauthors" class="tab" onclick="showTab('coauthors')">Top co-authors</div>
        </div>

        <div id="panel-papers" class="panel">
          {papers_table_html}
        </div>

        <div id="panel-coauthors" class="panel">
          {coauthors_table_html}
        </div>

        <div class="footer">Data: OpenAlex · local HTML rendering</div>
      </div>

      <script>
        function showTab(which) {{
          const tabP = document.getElementById("tab-papers");
          const tabC = document.getElementById("tab-coauthors");
          const panP = document.getElementById("panel-papers");
          const panC = document.getElementById("panel-coauthors");

          if (which === "coauthors") {{
            tabC.classList.add("active");
            tabP.classList.remove("active");
            panC.classList.add("active");
            panP.classList.remove("active");
          }} else {{
            tabP.classList.add("active");
            tabC.classList.remove("active");
            panP.classList.add("active");
            panC.classList.remove("active");
          }}

          const params = new URLSearchParams(window.location.search);
          params.set("tab", which);
          const newUrl = window.location.pathname + "?" + params.toString();
          window.history.replaceState(null, "", newUrl);
        }}

        async function doSearch() {{
          const q = document.getElementById("q").value.trim();
          const status = document.getElementById("status");
          const results = document.getElementById("results");
          results.innerHTML = "";
          if (q.length < 2) {{
            status.textContent = "Please type at least 2 characters.";
            return;
          }}
          status.textContent = "Searching...";
          try {{
            const r = await fetch(`/author_search?name=${{encodeURIComponent(q)}}`);
            const data = await r.json();
            const items = data.results || [];
            if (!items.length) {{
              status.textContent = "No results.";
              return;
            }}
            status.textContent = `Results for "${{q}}":`;
            for (const a of items) {{
              const div = document.createElement("div");
              div.className = "result";
              const concepts = (a.concepts || []).filter(Boolean).join(", ");
              div.innerHTML = `
                <div class="rmain">
                  <div class="rname">${{a.display_name || "-"}}</div>
                  <div class="rmeta">
                    ID: <span style="color:#e9ecf5">${{a.id || "-"}}</span>
                    · Works: ${{a.works_count ?? "-"}}
                    · Citations: ${{a.cited_by_count ?? "-"}}
                    ${{a.last_known_institution ? " · " + a.last_known_institution : ""}}
                    ${{a.orcid ? " · ORCID: " + a.orcid : ""}}
                    ${{concepts ? "<br/>Concepts: " + concepts : ""}}
                  </div>
                </div>
                <button class="use" onclick="useAuthor('${{a.id}}')">Use</button>
              `;
              results.appendChild(div);
            }}
          }} catch (e) {{
            status.textContent = "Error during search.";
          }}
        }}

        function useAuthor(id) {{
          const params = new URLSearchParams(window.location.search);
          params.set("author_id", id);

          const active = document.getElementById("tab-coauthors").classList.contains("active") ? "coauthors" : "papers";
          params.set("tab", active);

          window.location.search = params.toString();
        }}

        document.addEventListener("DOMContentLoaded", () => {{
          const q = document.getElementById("q");
          q.addEventListener("keydown", (e) => {{
            if (e.key === "Enter") doSearch();
          }});

          showTab("{'coauthors' if (tab or '').lower().strip() == 'coauthors' else 'papers'}");
        }});
      </script>
    </body>
    </html>
    """

    return HTMLResponse(page)
