#!/usr/bin/env python3
"""
주간 태양물리/우주기상 논문 자동 리뷰 생성기
- arXiv API + NASA ADS API로 최신 논문 검색
- Claude API + Gemini API로 한국어 요약 및 에이전트 토론 생성
- HTML 파일 자동 생성 및 index.html 업데이트
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import io

from bs4 import BeautifulSoup

import anthropic
import requests

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# ── 설정 ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ADS_API_KEY = os.environ.get("ADS_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "JunmuYOUN"
REPO_NAME = "sswl-paper-review"

# 검색 키워드
ARXIV_CATEGORIES = ["astro-ph.SR", "physics.space-ph"]
SEARCH_KEYWORDS = [
    "solar flare", "coronal mass ejection", "space weather",
    "solar wind", "geomagnetic storm", "solar corona",
    "magnetosphere", "solar energetic particles",
    "sunspot", "heliosphere",
]

# 대상 저널 (ADS bibstem 코드)
TARGET_JOURNALS = [
    "ApJ",    # The Astrophysical Journal
    "ApJS",   # The Astrophysical Journal Supplement Series
    "ApJL",   # The Astrophysical Journal Letters
    "SpWea",  # Space Weather
    "A&A",    # Astronomy & Astrophysics
    "SoPh",   # Solar Physics
    "JGRA",   # Journal of Geophysical Research: Space Physics
    "GeoRL",  # Geophysical Research Letters
    "LRSP",   # Living Reviews in Solar Physics
    "NatAs",  # Nature Astronomy
]

# arXiv journal_ref 매칭용 (저널 이름 → 표시명)
JOURNAL_NAME_MAP = {
    "astrophysical journal": "ApJ",
    "apj": "ApJ",
    "astrophysical journal supplement": "ApJS",
    "apjs": "ApJS",
    "astrophysical journal letters": "ApJL",
    "apjl": "ApJL",
    "space weather": "Space Weather",
    "astronomy & astrophysics": "A&A",
    "astronomy and astrophysics": "A&A",
    "a&a": "A&A",
    "solar physics": "Solar Physics",
    "journal of geophysical research": "JGR",
    "geophysical research letters": "GeoRL",
    "nature astronomy": "Nature Astronomy",
    "living reviews in solar physics": "LRSP",
}

TARGET_PAPER_COUNT = 5
BASE_DIR = Path(__file__).resolve().parent.parent
POSTS_DIR = BASE_DIR / "posts"
INDEX_PATH = BASE_DIR / "index.html"


def get_previously_reviewed_titles():
    """기존 posts/*.html 파일에서 이미 요약된 논문 제목들을 수집"""
    reviewed = set()
    if not POSTS_DIR.exists():
        return reviewed

    for html_file in POSTS_DIR.glob("*.html"):
        try:
            soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
            for h2 in soup.select(".paper-card h2"):
                title = h2.get_text(strip=True)
                # 정규화: 소문자, 여러 공백 → 하나
                normalized = re.sub(r"\s+", " ", title.lower().strip())
                if normalized:
                    reviewed.add(normalized)
        except Exception as e:
            print(f"[기존 리뷰] {html_file.name} 파싱 오류: {e}")

    print(f"[기존 리뷰] 이전에 요약된 논문 {len(reviewed)}편 발견")
    return reviewed


def get_week_info(offset=0):
    """현재 또는 offset 주 후의 ISO 주차 정보 반환"""
    today = datetime.utcnow() + timedelta(weeks=offset)
    iso_year, iso_week, _ = today.isocalendar()
    # 이번 주 월요일~일요일
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return {
        "year": iso_year,
        "week": iso_week,
        "week_str": f"{iso_year}-W{iso_week:02d}",
        "monday": monday,
        "sunday": sunday,
        "date_range": f"{monday.strftime('%Y.%m.%d')} — {sunday.strftime('%m.%d')}",
        "date_str": monday.strftime("%Y.%m.%d"),
    }


def fetch_giscus_suggestions(week_str):
    """이번 주 placeholder 페이지의 Giscus 댓글에서 논문/주제 제안 수집"""
    if not GITHUB_TOKEN:
        print("[Giscus] GITHUB_TOKEN 없음 — 제안 수집 건너뜀")
        return [], []

    filename = f"posts/{week_str.lower()}.html"
    print(f"[Giscus] {filename} 댓글에서 제안 수집 중...")

    query = """
    query($owner: String!, $name: String!, $categoryId: ID!) {
      repository(owner: $owner, name: $name) {
        discussions(first: 20, categoryId: $categoryId) {
          nodes {
            title
            comments(first: 50) {
              nodes {
                body
                author { login }
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "owner": REPO_OWNER,
        "name": REPO_NAME,
        "categoryId": "DIC_kwDORwFYbs4C5Ozg",
    }
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        proposed_papers = []
        proposed_keywords = []

        discussions = (
            data.get("data", {})
            .get("repository", {})
            .get("discussions", {})
            .get("nodes", [])
        )

        for disc in discussions:
            title = disc.get("title", "")
            # Giscus pathname mapping: 제목에 파일 경로 포함
            if filename not in title and week_str.lower() not in title.lower():
                continue

            for comment in disc.get("comments", {}).get("nodes", []):
                body = comment.get("body", "").strip()
                author = comment.get("author", {}).get("login", "익명")

                if not body:
                    continue

                # arXiv/ADS URL 추출
                urls = re.findall(
                    r'https?://(?:arxiv\.org|ui\.adsabs\.harvard\.edu)/\S+',
                    body,
                )

                if urls:
                    for url in urls:
                        url = url.rstrip(".,;)")
                        proposed_papers.append({
                            "type": "paper",
                            "url": url,
                            "name": author,
                            "content": body,
                            "title": f"Giscus 제안: {author}",
                        })
                        print(f"[Giscus] 논문 제안: {url[:60]} (by {author})")
                else:
                    proposed_keywords.append(body)
                    print(f"[Giscus] 주제 제안: {body[:60]} (by {author})")

        print(f"[Giscus] 수집 완료: 논문 {len(proposed_papers)}편, 주제 {len(proposed_keywords)}개")
        return proposed_papers, proposed_keywords

    except Exception as e:
        print(f"[Giscus] 댓글 수집 실패: {e}")
        return [], []


def fetch_arxiv_papers(max_results=20):
    """arXiv API로 태양물리/우주기상 논문 검색"""
    print("[arXiv] 논문 검색 중...")
    cat_query = " OR ".join(f"cat:{cat}" for cat in ARXIV_CATEGORIES)
    kw_query = " OR ".join(f'all:"{kw}"' for kw in SEARCH_KEYWORDS[:5])
    query = f"({cat_query}) AND ({kw_query})"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    resp = requests.get("http://export.arxiv.org/api/query", params=params, timeout=30)
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        link = entry.find("atom:id", ns).text.strip()
        published = entry.find("atom:published", ns).text.strip()[:10]

        categories = [c.get("term") for c in entry.findall("atom:category", ns)]
        primary_cat = categories[0] if categories else "astro-ph.SR"

        # journal_ref 추출 (출판된 저널 정보)
        journal_ref_el = entry.find("arxiv:journal_ref", ns)
        journal_ref = journal_ref_el.text.strip() if journal_ref_el is not None and journal_ref_el.text else ""
        journal_name = _match_journal(journal_ref)

        source = f"arXiv: {primary_cat}"
        if journal_name:
            source = f"{journal_name} (arXiv: {primary_cat})"

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": summary,
            "url": link,
            "date": published,
            "source": source,
            "journal": journal_name,
        })

    print(f"[arXiv] {len(papers)}편 발견")
    return papers


def _match_journal(journal_ref):
    """journal_ref 문자열에서 대상 저널 이름 매칭"""
    if not journal_ref:
        return ""
    ref_lower = journal_ref.lower()
    for keyword, display_name in JOURNAL_NAME_MAP.items():
        if keyword in ref_lower:
            return display_name
    return ""


def fetch_ads_papers(max_results=20):
    """NASA ADS API로 논문 검색"""
    if not ADS_API_KEY:
        print("[ADS] API 키 없음 — 건너뜀")
        return []

    print("[ADS] 논문 검색 중...")
    one_year_ago = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # 저널 필터 생성
    journal_filter = " OR ".join(f'bibstem:"{j}"' for j in TARGET_JOURNALS)

    query = (
        f'({journal_filter}) '
        'abs:("solar flare" OR "coronal mass ejection" OR "space weather" '
        'OR "solar wind" OR "geomagnetic storm" OR "solar corona" '
        'OR "solar energetic particles" OR "magnetosphere" OR "sunspot" '
        'OR "heliosphere" OR "solar cycle") '
        f'pubdate:[{one_year_ago} TO {today}]'
    )

    headers = {"Authorization": f"Bearer {ADS_API_KEY}"}
    params = {
        "q": query,
        "fl": "title,author,abstract,bibcode,pubdate,pub,identifier",
        "rows": max_results,
        "sort": "date desc",
    }

    try:
        resp = requests.get(
            "https://api.adsabs.harvard.edu/v1/search/query",
            headers=headers, params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ADS] 오류: {e}")
        return []

    papers = []
    for doc in data.get("response", {}).get("docs", []):
        title = doc.get("title", [""])[0]
        authors = doc.get("author", [])[:5]
        abstract = doc.get("abstract", "")
        bibcode = doc.get("bibcode", "")
        pubdate = doc.get("pubdate", "")[:10]
        pub = doc.get("pub", "")
        url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}"

        # 저널명 표시
        source = pub if pub else "NASA ADS"

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": url,
            "date": pubdate,
            "source": source,
            "journal": pub,
        })

    print(f"[ADS] {len(papers)}편 발견")
    return papers


def deduplicate_papers(papers, previously_reviewed):
    """제목 기반 중복 제거 + 이전 주차에서 요약한 논문 제외"""
    seen = set()
    unique = []
    skipped = 0
    for p in papers:
        key = re.sub(r"\s+", " ", p["title"].lower().strip())
        if key in previously_reviewed:
            skipped += 1
            continue
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if skipped:
        print(f"[중복] 이전 주차에서 이미 요약한 논문 {skipped}편 제외")
    return unique


def _arxiv_pdf_url(url):
    """arXiv URL을 PDF 다운로드 URL로 변환"""
    # https://arxiv.org/abs/2501.12345 → https://arxiv.org/pdf/2501.12345
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/")
    if "arxiv.org/pdf/" in url:
        return url
    return None


def fetch_paper_fulltext(url, max_chars=8000):
    """논문 PDF를 다운로드하여 텍스트 추출 (arXiv만 지원)"""
    if PdfReader is None:
        print("[PDF] PyPDF2 미설치 — 건너뜀")
        return ""

    pdf_url = _arxiv_pdf_url(url)
    if not pdf_url:
        print(f"[PDF] arXiv URL이 아님 — 건너뜀: {url[:60]}")
        return ""

    try:
        print(f"[PDF] 다운로드 중: {pdf_url}")
        resp = requests.get(pdf_url, timeout=60, headers={
            "User-Agent": "SSWL-Paper-Review/1.0 (academic research)"
        })
        resp.raise_for_status()

        reader = PdfReader(io.BytesIO(resp.content))
        text_parts = []
        total_chars = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            total_chars += len(page_text)
            if total_chars >= max_chars:
                break

        full_text = "\n".join(text_parts)
        # max_chars로 자르기
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n... [본문 생략]"

        print(f"[PDF] 텍스트 추출 완료: {len(full_text)}자 ({len(reader.pages)}페이지)")
        return full_text

    except Exception as e:
        print(f"[PDF] 추출 실패: {e}")
        return ""


def select_and_summarize(papers, week_info, proposed_keywords=None):
    """Claude API로 논문 선별 및 한국어 요약 생성"""
    print(f"[Claude] {len(papers)}편 중 {TARGET_PAPER_COUNT}편 선별 및 요약 중...")

    paper_list_text = ""
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += " et al."
        paper_list_text += f"""
---
[논문 {i}]
제목: {p['title']}
저자: {authors_str}
날짜: {p['date']}
출처: {p['source']}
URL: {p['url']}
초록: {p['abstract'][:500]}
"""

    prompt = f"""당신은 태양물리학 및 우주기상(Space Weather) 분야의 전문 리뷰어입니다.

아래에 이번 주({week_info['date_range']}) 발표된 논문 목록이 있습니다.
이 중에서 태양물리/우주기상 연구자에게 가장 중요하고 흥미로운 논문을 최대 {TARGET_PAPER_COUNT}편 선별하여 한국어로 요약해 주세요.
{('연구실에서 다음 주제에 관심을 가지고 있으니 관련 논문을 우선 선별하세요: ' + ', '.join(proposed_keywords or [])) if proposed_keywords else ''}

각 논문에 대해 다음 JSON 형식으로 응답해 주세요:
```json
[
  {{
    "index": 원본 논문 번호(정수),
    "one_line_summary": "한 줄 요약 (핵심 발견을 간결하게, 전문 용어 포함 시 영문 병기)",
    "background": "연구 배경 (3-5문장, 전문 용어에 영문 병기)",
    "method": "연구 방법 (3-5문장, 관측기기/데이터/분석기법 명시)",
    "findings": "핵심 발견 (3-5문장, 수치 결과 포함)",
    "significance": "의의 및 시사점 (2-3문장)"
  }}
]
```

중요:
- 반드시 유효한 JSON 배열만 출력하세요 (코드 블록 마크다운 없이)
- 전문 용어 첫 등장 시 한국어(영문) 형식으로 병기
- 논문이 {TARGET_PAPER_COUNT}편 미만이면 있는 만큼만 선별

{paper_list_text}
"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text.strip()
    # JSON 코드 블록 제거
    response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)

    summaries = json.loads(response_text)
    print(f"[Claude] {len(summaries)}편 요약 완료")

    # 원본 논문 메타데이터와 병합
    results = []
    for s in summaries:
        idx = s["index"] - 1
        if 0 <= idx < len(papers):
            paper = papers[idx]
            authors_str = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors_str += " et al."
            results.append({
                "title": paper["title"],
                "authors": authors_str,
                "date": paper["date"],
                "source": paper["source"],
                "url": paper["url"],
                "one_line_summary": s["one_line_summary"],
                "background": s["background"],
                "method": s["method"],
                "findings": s["findings"],
                "significance": s["significance"],
            })

    return results



def generate_post_html(papers, week_info):
    """주간 리뷰 HTML 파일 생성"""
    week_str = week_info["week_str"]
    year = week_info["year"]
    week_num = week_info["week"]
    monday = week_info["monday"]
    sunday = week_info["sunday"]

    # 월 주차 계산
    month = monday.month
    week_of_month = (monday.day - 1) // 7 + 1

    paper_cards = ""
    for i, p in enumerate(papers, 1):
        paper_cards += f"""
    <!-- Paper {i} -->
    <article class="paper-card">
      <div class="paper-num">{i}</div>
      <h2>
        <a href="{week_str.lower()}/paper-{i}.html">
          {p['title']}
        </a>
      </h2>
      <div class="paper-meta">
        <span>{p['authors']}</span>
        <span>{p['date']}</span>
        <span class="source-badge">{p['source']}</span>
      </div>

      <div class="one-line-summary">
        {p['one_line_summary']}
      </div>

      <div class="card-actions">
        <div class="vote-bar" data-paper-id="paper-{i}">
          <span class="vote-label">이 논문이 유용했나요?</span>
          <div class="vote-buttons">
            <button class="vote-btn vote-up" onclick="vote('paper-{i}', 'up')" title="추천">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
              <span class="vote-count" id="paper-{i}-up">0</span>
            </button>
            <button class="vote-btn vote-down" onclick="vote('paper-{i}', 'down')" title="비추천">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38-9a2 2 0 0 0-2 2.3H10z"/><path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
              <span class="vote-count" id="paper-{i}-down">0</span>
            </button>
          </div>
        </div>
        <a href="{week_str.lower()}/paper-{i}.html" class="discuss-btn">토론 참여 &rarr;</a>
      </div>
    </article>
"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>주간 논문 리뷰 {week_str} | SSWL Paper Review</title>
  <style>
    :root {{
      --sun-gold: #E8A838;
      --sun-orange: #D4722A;
      --deep-space: #0B1426;
      --space-blue: #1A2744;
      --nebula-blue: #2A3F6E;
      --star-white: #F4F1EC;
      --aurora-cyan: #64B5C6;
      --corona-red: #C75B3A;
      --text-primary: #1A1A2E;
      --text-secondary: #4A4A6A;
      --text-light: #8888A8;
      --bg-warm: #FDFBF7;
      --bg-card: #FFFFFF;
      --border-subtle: #E8E4DC;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      font-family: Arial, sans-serif;
      background: var(--bg-warm);
      color: var(--text-primary);
      line-height: 1.8;
    }}

    /* Top Nav */
    .topnav {{
      background: var(--deep-space);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }}

    .topnav a {{
      color: rgba(244,241,236,0.7);
      text-decoration: none;
      font-size: 0.85rem;
      transition: color 0.2s;
    }}

    .topnav a:hover {{ color: var(--sun-gold); }}

    .topnav .brand {{
      font-family: Arial, sans-serif;
      font-size: 1rem;
      color: var(--star-white);
      font-weight: 600;
    }}

    .topnav .brand .accent {{ color: var(--sun-gold); }}

    /* Post Header */
    .post-header {{
      background: linear-gradient(180deg, var(--space-blue) 0%, var(--deep-space) 100%);
      padding: 60px 24px 50px;
      position: relative;
      overflow: hidden;
    }}

    .post-header::after {{
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 80px;
      background: linear-gradient(to top, var(--bg-warm), transparent);
    }}

    .post-header-inner {{
      max-width: 800px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
    }}

    .post-header .week-tag {{
      display: inline-block;
      background: rgba(232,168,56,0.2);
      border: 1px solid rgba(232,168,56,0.4);
      color: var(--sun-gold);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 4px;
      letter-spacing: 0.08em;
      margin-bottom: 16px;
    }}

    .post-header h1 {{
      font-family: Arial, sans-serif;
      font-size: clamp(1.5rem, 4vw, 2rem);
      font-weight: 700;
      color: var(--star-white);
      line-height: 1.4;
      margin-bottom: 16px;
    }}

    .post-header .meta {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      font-size: 0.85rem;
      color: rgba(244,241,236,0.6);
    }}

    .post-header .meta span {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    /* Content */
    .content {{
      max-width: 800px;
      margin: -20px auto 0;
      padding: 0 24px 80px;
      position: relative;
      z-index: 2;
    }}

    .intro-box {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-left: 4px solid var(--sun-gold);
      border-radius: 8px;
      padding: 24px 28px;
      margin-bottom: 40px;
      font-size: 0.95rem;
      color: var(--text-secondary);
    }}

    /* Paper Cards */
    .paper-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 32px;
      margin-bottom: 24px;
      transition: box-shadow 0.2s;
    }}

    .paper-card:hover {{
      box-shadow: 0 4px 20px rgba(11,20,38,0.06);
    }}

    .paper-num {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      background: var(--deep-space);
      color: var(--sun-gold);
      font-size: 0.75rem;
      font-weight: 700;
      border-radius: 6px;
      margin-bottom: 14px;
    }}

    .paper-card h2 {{
      font-family: Arial, sans-serif;
      font-size: 1.15rem;
      font-weight: 600;
      line-height: 1.5;
      margin-bottom: 8px;
    }}

    .paper-card h2 a {{
      color: var(--text-primary);
      text-decoration: none;
      border-bottom: 1px solid transparent;
      transition: border-color 0.2s, color 0.2s;
    }}

    .paper-card h2 a:hover {{
      color: var(--sun-orange);
      border-bottom-color: var(--sun-orange);
    }}

    .paper-meta {{
      font-size: 0.8rem;
      color: var(--text-light);
      margin-bottom: 18px;
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .paper-meta .source-badge {{
      background: var(--bg-warm);
      padding: 1px 8px;
      border-radius: 4px;
      font-weight: 500;
      font-size: 0.7rem;
    }}

    .one-line-summary {{
      background: linear-gradient(135deg, rgba(232,168,56,0.08), rgba(100,181,198,0.06));
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 18px;
      font-size: 0.95rem;
      font-weight: 500;
      color: var(--text-primary);
      line-height: 1.6;
    }}

    .card-actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-top: 8px;
      padding-top: 16px;
      border-top: 1px solid var(--border-subtle);
    }}

    .discuss-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 20px;
      background: linear-gradient(135deg, var(--deep-space), var(--space-blue));
      color: var(--sun-gold);
      border: none;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s;
      white-space: nowrap;
    }}

    .discuss-btn:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(11,20,38,0.2);
    }}

    /* Vote Bar */
    .vote-bar {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .vote-label {{
      font-size: 0.82rem;
      color: var(--text-light);
    }}

    .vote-buttons {{
      display: flex;
      gap: 8px;
    }}

    .vote-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      background: var(--bg-warm);
      color: var(--text-light);
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.2s;
    }}

    .vote-btn:hover {{
      border-color: var(--text-light);
      color: var(--text-secondary);
    }}

    .vote-btn.active.vote-up {{
      background: rgba(76, 175, 80, 0.1);
      border-color: #4CAF50;
      color: #4CAF50;
    }}

    .vote-btn.active.vote-down {{
      background: rgba(244, 67, 54, 0.1);
      border-color: #F44336;
      color: #F44336;
    }}

    .vote-count {{
      font-weight: 600;
      min-width: 12px;
      text-align: center;
    }}

    /* Footer */
    footer {{
      border-top: 1px solid var(--border-subtle);
      padding: 32px 24px;
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-light);
    }}

    footer a {{
      color: var(--sun-orange);
      text-decoration: none;
    }}

    @media (max-width: 640px) {{
      .paper-card {{ padding: 20px; }}
      .paper-meta {{ flex-direction: column; gap: 4px; }}
    }}
  </style>
</head>
<body>

  <!-- Navigation -->
  <nav class="topnav">
    <a href="../index.html" class="brand">&#9728; SSWL <span class="accent">Paper Review</span></a>
    <a href="../index.html">&larr; 목록으로</a>
  </nav>

  <!-- Post Header -->
  <header class="post-header">
    <div class="post-header-inner">
      <span class="week-tag">{week_str}</span>
      <h1>주간 태양/우주기상 논문 리뷰<br>{year}년 {month}월 {week_of_month}주차</h1>
      <div class="meta">
        <span>&#128197; {monday.strftime('%Y.%m.%d')} &mdash; {sunday.strftime('%m.%d')}</span>
        <span>&#128196; {len(papers)}편 리뷰</span>
        <span>&#129302; Assiworks 자동 생성</span>
      </div>
    </div>
  </header>

  <!-- Content -->
  <main class="content">

    <div class="intro-box">
      이번 주 태양물리 및 우주기상 분야에서 발표된 주요 논문 {len(papers)}편을 선별하여 요약합니다.
      arXiv, NASA ADS에서 자동 수집되었으며, AI가 전문 요약 및 한국어 번역을 수행했습니다.
    </div>
{paper_cards}
  </main>

  <footer>
    <p>
      Sun and Space Weather Laboratory &middot; Kyung Hee University<br>
      <a href="https://sunspaceweather.khu.ac.kr/">sunspaceweather.khu.ac.kr</a>
      &middot; Automated by <a href="https://assiworks.aifactory.space">Assiworks (AI Factory)</a>
    </p>
  </footer>

  <script>
    function getVotes() {{
      return JSON.parse(localStorage.getItem('sswl-votes') || '{{}}');
    }}

    function saveVotes(votes) {{
      localStorage.setItem('sswl-votes', JSON.stringify(votes));
    }}

    function vote(paperId, type) {{
      var votes = getVotes();
      var key = location.pathname + ':' + paperId;
      var prev = votes[key];

      if (prev === type) {{
        delete votes[key];
      }} else {{
        votes[key] = type;
      }}

      saveVotes(votes);
      renderVotes();
    }}

    function renderVotes() {{
      var votes = getVotes();
      var prefix = location.pathname + ':';
      var counts = {{}};

      for (var key in votes) {{
        if (key.startsWith(prefix)) {{
          var paperId = key.slice(prefix.length);
          if (!counts[paperId]) counts[paperId] = {{ up: 0, down: 0 }};
          counts[paperId][votes[key]]++;
        }}
      }}

      document.querySelectorAll('.vote-bar').forEach(function(bar) {{
        var paperId = bar.dataset.paperId;
        var userVote = votes[prefix + paperId] || null;
        var c = counts[paperId] || {{ up: 0, down: 0 }};

        var upBtn = bar.querySelector('.vote-up');
        var downBtn = bar.querySelector('.vote-down');

        upBtn.classList.toggle('active', userVote === 'up');
        downBtn.classList.toggle('active', userVote === 'down');

        bar.querySelector('#' + paperId + '-up').textContent = c.up;
        bar.querySelector('#' + paperId + '-down').textContent = c.down;
      }});
    }}

    renderVotes();
  </script>

</body>
</html>"""

    return html


def generate_paper_page_html(paper, paper_index, week_info, total_papers):
    """개별 논문 토론 페이지 HTML 생성"""
    week_str = week_info["week_str"]
    i = paper_index  # 1-based

    # Build paper nav links
    nav_links = ""
    for j in range(1, total_papers + 1):
        active_cls = ' class="active"' if j == i else ''
        nav_links += f'    <a href="paper-{j}.html"{active_cls}>{j}</a>\n'

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Paper {i}: {paper['title'][:60]} | SSWL Paper Review</title>
  <style>
    :root {{
      --sun-gold: #E8A838;
      --sun-orange: #D4722A;
      --deep-space: #0B1426;
      --space-blue: #1A2744;
      --nebula-blue: #2A3F6E;
      --star-white: #F4F1EC;
      --aurora-cyan: #64B5C6;
      --corona-red: #C75B3A;
      --text-primary: #1A1A2E;
      --text-secondary: #4A4A6A;
      --text-light: #8888A8;
      --bg-warm: #FDFBF7;
      --bg-card: #FFFFFF;
      --border-subtle: #E8E4DC;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      font-family: Arial, sans-serif;
      background: var(--bg-warm);
      color: var(--text-primary);
      line-height: 1.8;
    }}

    /* Top Nav */
    .topnav {{
      background: var(--deep-space);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }}

    .topnav a {{
      color: rgba(244,241,236,0.7);
      text-decoration: none;
      font-size: 0.85rem;
      transition: color 0.2s;
    }}

    .topnav a:hover {{ color: var(--sun-gold); }}

    .topnav .brand {{
      font-family: Arial, sans-serif;
      font-size: 1rem;
      color: var(--star-white);
      font-weight: 600;
    }}

    .topnav .brand .accent {{ color: var(--sun-gold); }}

    /* Paper Nav */
    .paper-nav {{
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-subtle);
      padding: 10px 24px;
      display: flex;
      justify-content: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .paper-nav a {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 8px;
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 600;
      transition: all 0.2s;
      border: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      background: var(--bg-warm);
    }}

    .paper-nav a:hover {{
      border-color: var(--sun-gold);
      color: var(--sun-gold);
    }}

    .paper-nav a.active {{
      background: var(--deep-space);
      color: var(--sun-gold);
      border-color: var(--deep-space);
    }}

    /* Post Header */
    .post-header {{
      background: linear-gradient(180deg, var(--space-blue) 0%, var(--deep-space) 100%);
      padding: 50px 24px 40px;
      position: relative;
      overflow: hidden;
    }}

    .post-header::after {{
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 80px;
      background: linear-gradient(to top, var(--bg-warm), transparent);
    }}

    .post-header-inner {{
      max-width: 800px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
    }}

    .post-header .week-tag {{
      display: inline-block;
      background: rgba(232,168,56,0.2);
      border: 1px solid rgba(232,168,56,0.4);
      color: var(--sun-gold);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 4px;
      letter-spacing: 0.08em;
      margin-bottom: 16px;
    }}

    .post-header h1 {{
      font-family: Arial, sans-serif;
      font-size: clamp(1.2rem, 3.5vw, 1.6rem);
      font-weight: 700;
      color: var(--star-white);
      line-height: 1.5;
      margin-bottom: 12px;
    }}

    .post-header .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 0.82rem;
      color: rgba(244,241,236,0.6);
    }}

    .post-header .meta a {{
      color: var(--aurora-cyan);
      text-decoration: none;
    }}

    .post-header .meta a:hover {{ text-decoration: underline; }}

    /* Content */
    .content {{
      max-width: 800px;
      margin: -20px auto 0;
      padding: 0 24px 80px;
      position: relative;
      z-index: 2;
    }}

    /* Paper Card (for discussion injection) */
    .paper-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 32px;
      margin-bottom: 24px;
    }}

    .summary-section {{
      margin-bottom: 16px;
    }}

    .summary-section .label {{
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--sun-orange);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }}

    .summary-section p {{
      font-size: 0.92rem;
      color: var(--text-secondary);
      line-height: 1.75;
    }}

    .one-line-summary {{
      background: linear-gradient(135deg, rgba(232,168,56,0.08), rgba(100,181,198,0.06));
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 20px;
      font-size: 0.95rem;
      font-weight: 500;
      color: var(--text-primary);
      line-height: 1.6;
    }}

    /* Vote Bar */
    .vote-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 20px;
      padding-top: 16px;
      border-top: 1px solid var(--border-subtle);
    }}

    .vote-label {{
      font-size: 0.82rem;
      color: var(--text-light);
    }}

    .vote-buttons {{
      display: flex;
      gap: 8px;
    }}

    .vote-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      background: var(--bg-warm);
      color: var(--text-light);
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.2s;
    }}

    .vote-btn:hover {{
      border-color: var(--text-light);
      color: var(--text-secondary);
    }}

    .vote-btn.active.vote-up {{
      background: rgba(76, 175, 80, 0.1);
      border-color: #4CAF50;
      color: #4CAF50;
    }}

    .vote-btn.active.vote-down {{
      background: rgba(244, 67, 54, 0.1);
      border-color: #F44336;
      color: #F44336;
    }}

    .vote-count {{
      font-weight: 600;
      min-width: 12px;
      text-align: center;
    }}

    /* Discussion Section */
    .discussion-section {{
      margin-top: 24px;
      padding-top: 20px;
      border-top: 1px dashed var(--border-subtle);
    }}

    .discussion-title {{
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--nebula-blue);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .discussion-title::before {{
      content: '';
      display: inline-block;
      width: 18px;
      height: 18px;
      background: linear-gradient(135deg, var(--sun-gold), var(--aurora-cyan));
      border-radius: 4px;
    }}

    .discussion-thread {{ display: flex; flex-direction: column; gap: 12px; }}

    .day-divider {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 8px 0;
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .day-divider::before, .day-divider::after {{
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border-subtle);
    }}

    .agent-comment {{
      padding: 16px 18px;
      border-radius: 10px;
      font-size: 0.9rem;
      line-height: 1.7;
    }}

    .agent-comment.theorist {{
      background: rgba(147, 51, 234, 0.04);
      border: 1px solid rgba(147, 51, 234, 0.15);
    }}
    .agent-comment.observer {{
      background: rgba(66, 133, 244, 0.04);
      border: 1px solid rgba(66, 133, 244, 0.15);
    }}
    .agent-comment.critic {{
      background: rgba(234, 88, 12, 0.04);
      border: 1px solid rgba(234, 88, 12, 0.15);
    }}
    .agent-comment.student {{
      background: rgba(22, 163, 74, 0.04);
      border: 1px solid rgba(22, 163, 74, 0.15);
    }}
    .agent-comment.professor {{
      background: linear-gradient(135deg, rgba(232,168,56,0.06), rgba(100,181,198,0.06));
      border: 1px solid rgba(232,168,56,0.2);
    }}
    .agent-comment.human {{
      background: rgba(59, 130, 246, 0.04);
      border: 1px solid rgba(59, 130, 246, 0.25);
      border-left: 3px solid #3B82F6;
    }}

    .agent-comment p {{ color: var(--text-secondary); margin: 0; white-space: pre-line; }}

    .agent-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      margin-bottom: 8px;
    }}

    .claude-badge {{ color: #7C3AED; }}
    .gemini-badge {{ color: #4285F4; }}
    .critic-badge {{ color: #EA580C; }}
    .student-badge {{ color: #16A34A; }}
    .professor-badge {{ color: #B45309; }}
    .human-badge {{ color: #3B82F6; }}

    .agent-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      border-radius: 5px;
      font-size: 0.7rem;
      font-weight: 700;
      color: white;
    }}

    .claude-badge .agent-icon {{ background: #7C3AED; }}
    .gemini-badge .agent-icon {{ background: #4285F4; }}
    .critic-badge .agent-icon {{ background: #EA580C; }}
    .student-badge .agent-icon {{ background: #16A34A; }}
    .professor-badge .agent-icon {{ background: #B45309; }}
    .human-badge .agent-icon {{ background: #3B82F6; }}

    .agent-model {{
      font-weight: 400;
      color: var(--text-light);
      font-size: 0.72rem;
    }}

    /* Comment Board */
    .comment-board {{
      margin-top: 40px;
      padding-top: 32px;
      border-top: 2px solid var(--border-subtle);
    }}

    .comment-board h2 {{
      font-family: Arial, sans-serif;
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 8px;
    }}

    .comment-notice {{
      font-size: 0.8rem;
      color: var(--text-light);
      margin-bottom: 20px;
    }}

    .comment-form {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }}

    .comment-form-row {{
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }}

    .comment-form input[type="text"] {{
      flex: 1;
      padding: 10px 14px;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      font-size: 0.9rem;
      background: var(--bg-warm);
      color: var(--text-primary);
      outline: none;
      transition: border-color 0.2s;
    }}

    .comment-form input[type="text"]:focus {{
      border-color: var(--sun-gold);
    }}

    .comment-form textarea {{
      width: 100%;
      min-height: 80px;
      padding: 12px 14px;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      font-size: 0.9rem;
      font-family: Arial, sans-serif;
      background: var(--bg-warm);
      color: var(--text-primary);
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
      margin-bottom: 12px;
    }}

    .comment-form textarea:focus {{
      border-color: var(--sun-gold);
    }}

    .comment-form button {{
      padding: 10px 24px;
      background: linear-gradient(135deg, var(--deep-space), var(--space-blue));
      color: var(--sun-gold);
      border: none;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}

    .comment-form button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(11,20,38,0.2);
    }}

    .comment-item {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 12px;
    }}

    .comment-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}

    .comment-author {{
      font-weight: 600;
      font-size: 0.88rem;
      color: var(--nebula-blue);
    }}

    .comment-time {{
      font-size: 0.75rem;
      color: var(--text-light);
    }}

    .comment-body {{
      font-size: 0.9rem;
      color: var(--text-secondary);
      line-height: 1.7;
      white-space: pre-line;
    }}

    .comment-delete {{
      background: none;
      border: none;
      color: var(--text-light);
      cursor: pointer;
      font-size: 0.75rem;
      padding: 2px 6px;
      border-radius: 4px;
      transition: all 0.2s;
    }}

    .comment-delete:hover {{
      background: rgba(244,67,54,0.1);
      color: #F44336;
    }}

    .no-comments {{
      text-align: center;
      padding: 24px;
      color: var(--text-light);
      font-size: 0.9rem;
    }}

    /* Footer */
    footer {{
      border-top: 1px solid var(--border-subtle);
      padding: 32px 24px;
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-light);
    }}

    footer a {{
      color: var(--sun-orange);
      text-decoration: none;
    }}

    @media (max-width: 640px) {{
      .paper-card {{ padding: 20px; }}
      .paper-meta {{ flex-direction: column; gap: 4px; }}
      .comment-form-row {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>

  <!-- Navigation -->
  <nav class="topnav">
    <a href="../../index.html" class="brand">&#9728; SSWL <span class="accent">Paper Review</span></a>
    <a href="../{week_str.lower()}.html">&larr; 주간 리뷰</a>
  </nav>

  <!-- Paper Navigation -->
  <div class="paper-nav">
{nav_links}  </div>

  <!-- Post Header -->
  <header class="post-header">
    <div class="post-header-inner">
      <span class="week-tag">{week_str} &middot; Paper {i}</span>
      <h1>{paper['title']}</h1>
      <div class="meta">
        <span>{paper['authors']}</span>
        <span>{paper['date']}</span>
        <span>{paper['source']}</span>
        <a href="{paper['url']}" target="_blank">원문 보기 &nearr;</a>
      </div>
    </div>
  </header>

  <!-- Content -->
  <main class="content">

    <article class="paper-card">
      <div class="one-line-summary">
        {paper['one_line_summary']}
      </div>

      <div class="summary-section">
        <div class="label">연구 배경</div>
        <p>{paper['background']}</p>
      </div>

      <div class="summary-section">
        <div class="label">연구 방법</div>
        <p>{paper['method']}</p>
      </div>

      <div class="summary-section">
        <div class="label">핵심 발견</div>
        <p>{paper['findings']}</p>
      </div>

      <div class="summary-section">
        <div class="label">의의 및 시사점</div>
        <p>{paper['significance']}</p>
      </div>

      <div class="vote-bar" data-paper-id="paper-{i}">
        <span class="vote-label">이 논문이 유용했나요?</span>
        <div class="vote-buttons">
          <button class="vote-btn vote-up" onclick="vote('paper-{i}', 'up')" title="추천">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
            <span class="vote-count" id="paper-{i}-up">0</span>
          </button>
          <button class="vote-btn vote-down" onclick="vote('paper-{i}', 'down')" title="비추천">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
            <span class="vote-count" id="paper-{i}-down">0</span>
          </button>
        </div>
      </div>
    </article>

    <!-- Comment Board -->
    <section class="comment-board">
      <h2>자유 토론</h2>
      <p class="comment-notice">GitHub 계정 없이 자유롭게 의견을 남길 수 있습니다. 댓글은 현재 기기의 브라우저에 저장됩니다.</p>

      <div class="comment-form">
        <div class="comment-form-row">
          <input type="text" id="comment-name" placeholder="닉네임 (선택사항)">
        </div>
        <textarea id="comment-text" placeholder="이 논문에 대한 의견, 질문, 토론 내용을 자유롭게 작성하세요..."></textarea>
        <button onclick="addComment()">작성하기</button>
      </div>

      <div id="comments-list"></div>
    </section>

  </main>

  <footer>
    <p>
      Sun and Space Weather Laboratory &middot; Kyung Hee University<br>
      <a href="https://sunspaceweather.khu.ac.kr/">sunspaceweather.khu.ac.kr</a>
      &middot; Automated by <a href="https://assiworks.aifactory.space">Assiworks (AI Factory)</a>
    </p>
  </footer>

  <script>
    /* ── Vote System ── */
    function getVotes() {{
      return JSON.parse(localStorage.getItem('sswl-votes') || '{{}}');
    }}

    function saveVotes(votes) {{
      localStorage.setItem('sswl-votes', JSON.stringify(votes));
    }}

    function vote(paperId, type) {{
      var votes = getVotes();
      var key = location.pathname + ':' + paperId;
      var prev = votes[key];
      if (prev === type) {{
        delete votes[key];
      }} else {{
        votes[key] = type;
      }}
      saveVotes(votes);
      renderVotes();
    }}

    function renderVotes() {{
      var votes = getVotes();
      var prefix = location.pathname + ':';
      var counts = {{}};
      for (var key in votes) {{
        if (key.startsWith(prefix)) {{
          var paperId = key.slice(prefix.length);
          if (!counts[paperId]) counts[paperId] = {{ up: 0, down: 0 }};
          counts[paperId][votes[key]]++;
        }}
      }}
      document.querySelectorAll('.vote-bar').forEach(function(bar) {{
        var paperId = bar.dataset.paperId;
        var userVote = votes[prefix + paperId] || null;
        var c = counts[paperId] || {{ up: 0, down: 0 }};
        var upBtn = bar.querySelector('.vote-up');
        var downBtn = bar.querySelector('.vote-down');
        upBtn.classList.toggle('active', userVote === 'up');
        downBtn.classList.toggle('active', userVote === 'down');
        bar.querySelector('#' + paperId + '-up').textContent = c.up;
        bar.querySelector('#' + paperId + '-down').textContent = c.down;
      }});
    }}

    renderVotes();

    /* ── Comment Board ── */
    var COMMENT_KEY = 'sswl-comments-{week_str.lower()}-paper-{i}';

    function getComments() {{
      return JSON.parse(localStorage.getItem(COMMENT_KEY) || '[]');
    }}

    function saveComments(comments) {{
      localStorage.setItem(COMMENT_KEY, JSON.stringify(comments));
    }}

    function addComment() {{
      var nameEl = document.getElementById('comment-name');
      var textEl = document.getElementById('comment-text');
      var text = textEl.value.trim();
      if (!text) return;

      var comments = getComments();
      comments.push({{
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        name: nameEl.value.trim() || '익명',
        text: text,
        time: new Date().toISOString()
      }});
      saveComments(comments);
      textEl.value = '';
      renderComments();
    }}

    function deleteComment(id) {{
      var comments = getComments().filter(function(c) {{ return c.id !== id; }});
      saveComments(comments);
      renderComments();
    }}

    function renderComments() {{
      var comments = getComments();
      var list = document.getElementById('comments-list');

      if (comments.length === 0) {{
        list.innerHTML = '<div class="no-comments">아직 댓글이 없습니다. 첫 번째 의견을 남겨보세요!</div>';
        return;
      }}

      var html = '';
      comments.forEach(function(c) {{
        var date = new Date(c.time);
        var timeStr = date.getFullYear() + '.' +
          String(date.getMonth() + 1).padStart(2, '0') + '.' +
          String(date.getDate()).padStart(2, '0') + ' ' +
          String(date.getHours()).padStart(2, '0') + ':' +
          String(date.getMinutes()).padStart(2, '0');

        html += '<div class="comment-item">' +
          '<div class="comment-header">' +
            '<span class="comment-author">' + escapeHtml(c.name) + '</span>' +
            '<span>' +
              '<span class="comment-time">' + timeStr + '</span> ' +
              '<button class="comment-delete" onclick="deleteComment(\\''+c.id+'\\')">삭제</button>' +
            '</span>' +
          '</div>' +
          '<div class="comment-body">' + escapeHtml(c.text) + '</div>' +
        '</div>';
      }});
      list.innerHTML = html;
    }}

    function escapeHtml(text) {{
      var div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }}

    // Textarea Enter key submit
    document.getElementById('comment-text').addEventListener('keydown', function(e) {{
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {{
        addComment();
      }}
    }});

    renderComments();
  </script>

</body>
</html>"""

    return html


def generate_paper_pages(papers, week_info):
    """개별 논문 토론 페이지들 생성"""
    week_str = week_info["week_str"]
    paper_dir = POSTS_DIR / week_str.lower()
    paper_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(papers, 1):
        page_html = generate_paper_page_html(p, i, week_info, len(papers))
        page_path = paper_dir / f"paper-{i}.html"
        page_path.write_text(page_html, encoding="utf-8")
        print(f"[개별 페이지] {page_path} 생성 완료")


def generate_next_week_placeholder(next_week_info):
    """다음 주차 제안 페이지 (placeholder) 생성 — Giscus로 논문/주제 제안 수집"""
    week_str = next_week_info["week_str"]
    filename = f"{week_str.lower()}.html"
    filepath = POSTS_DIR / filename

    # 이미 실제 리뷰가 있으면 건너뜀
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        if "paper-card" in content:
            print(f"[다음주] {filename} 이미 리뷰 존재 — 건너뜀")
            return

    year = next_week_info["year"]
    monday = next_week_info["monday"]
    sunday = next_week_info["sunday"]
    month = monday.month
    week_of_month = (monday.day - 1) // 7 + 1

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>다음 주 논문 제안 {week_str} | SSWL Paper Review</title>
  <style>
    :root {{
      --sun-gold: #E8A838;
      --sun-orange: #D4722A;
      --deep-space: #0B1426;
      --space-blue: #1A2744;
      --nebula-blue: #2A3F6E;
      --star-white: #F4F1EC;
      --aurora-cyan: #64B5C6;
      --text-primary: #1A1A2E;
      --text-secondary: #4A4A6A;
      --text-light: #8888A8;
      --bg-warm: #FDFBF7;
      --bg-card: #FFFFFF;
      --border-subtle: #E8E4DC;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      font-family: Arial, sans-serif;
      background: var(--bg-warm);
      color: var(--text-primary);
      line-height: 1.8;
    }}

    .topnav {{
      background: var(--deep-space);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }}

    .topnav a {{
      color: rgba(244,241,236,0.7);
      text-decoration: none;
      font-size: 0.85rem;
      transition: color 0.2s;
    }}

    .topnav a:hover {{ color: var(--sun-gold); }}

    .topnav .brand {{
      font-size: 1rem;
      color: var(--star-white);
      font-weight: 600;
    }}

    .topnav .brand .accent {{ color: var(--sun-gold); }}

    .post-header {{
      background: linear-gradient(180deg, var(--space-blue) 0%, var(--deep-space) 100%);
      padding: 60px 24px 50px;
      position: relative;
      overflow: hidden;
    }}

    .post-header::after {{
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 80px;
      background: linear-gradient(to top, var(--bg-warm), transparent);
    }}

    .post-header-inner {{
      max-width: 800px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
    }}

    .post-header .week-tag {{
      display: inline-block;
      background: rgba(232,168,56,0.2);
      border: 1px solid rgba(232,168,56,0.4);
      color: var(--sun-gold);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 4px;
      letter-spacing: 0.08em;
      margin-bottom: 16px;
    }}

    .post-header h1 {{
      font-size: clamp(1.5rem, 4vw, 2rem);
      font-weight: 700;
      color: var(--star-white);
      line-height: 1.4;
      margin-bottom: 16px;
    }}

    .post-header .meta {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      font-size: 0.85rem;
      color: rgba(244,241,236,0.6);
    }}

    .content {{
      max-width: 800px;
      margin: -20px auto 0;
      padding: 0 24px 80px;
      position: relative;
      z-index: 2;
    }}

    .intro-box {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-left: 4px solid var(--sun-gold);
      border-radius: 8px;
      padding: 24px 28px;
      margin-bottom: 40px;
      font-size: 0.95rem;
      color: var(--text-secondary);
      line-height: 1.9;
    }}

    .intro-box strong {{
      color: var(--text-primary);
      font-size: 1.05rem;
    }}

    .intro-box b {{
      color: var(--sun-orange);
    }}

    .giscus-section {{
      margin-top: 48px;
      padding-top: 32px;
      border-top: 2px solid var(--border-subtle);
    }}

    .giscus-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 20px;
    }}

    footer {{
      border-top: 1px solid var(--border-subtle);
      padding: 32px 24px;
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-light);
    }}

    footer a {{
      color: var(--sun-orange);
      text-decoration: none;
    }}

    @media (max-width: 640px) {{
      .intro-box {{ padding: 18px; }}
    }}
  </style>
</head>
<body>

  <nav class="topnav">
    <a href="../index.html" class="brand">&#9728; SSWL <span class="accent">Paper Review</span></a>
    <a href="../index.html">&larr; 목록으로</a>
  </nav>

  <header class="post-header">
    <div class="post-header-inner">
      <span class="week-tag">{week_str} (예정)</span>
      <h1>다음 주 논문 리뷰 제안<br>{year}년 {month}월 {week_of_month}주차</h1>
      <div class="meta">
        <span>&#128197; {monday.strftime('%Y.%m.%d')} &mdash; {sunday.strftime('%m.%d')}</span>
        <span>&#128161; 제안 받는 중</span>
      </div>
    </div>
  </header>

  <main class="content">

    <div class="intro-box">
      <strong>다음 주에 리뷰할 논문이나 주제를 아래 댓글로 제안해주세요!</strong><br><br>
      <b>논문 제안:</b> arXiv 또는 ADS 링크를 댓글에 남겨주세요.<br>
      <b>주제 제안:</b> 관심 있는 키워드나 주제를 자유롭게 작성해주세요.<br><br>
      제안된 내용은 다음 주 논문 선별 시 우선 반영됩니다.
    </div>

    <section class="giscus-section">
      <h2 class="giscus-title">논문/주제 제안</h2>
      <script src="https://giscus.app/client.js"
        data-repo="JunmuYOUN/sswl-paper-review"
        data-repo-id="R_kgDORwFYbg"
        data-category="General"
        data-category-id="DIC_kwDORwFYbs4C5Ozg"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="light"
        data-lang="ko"
        crossorigin="anonymous"
        async>
      </script>
    </section>
  </main>

  <footer>
    <p>
      Sun and Space Weather Laboratory &middot; Kyung Hee University<br>
      <a href="https://sunspaceweather.khu.ac.kr/">sunspaceweather.khu.ac.kr</a>
      &middot; Automated by <a href="https://assiworks.aifactory.space">Assiworks (AI Factory)</a>
    </p>
  </footer>

</body>
</html>"""

    POSTS_DIR.mkdir(exist_ok=True)
    filepath.write_text(html, encoding="utf-8")
    print(f"[다음주] {filename} placeholder 생성 완료")

    # index.html에 제안 카드 추가
    index_html = INDEX_PATH.read_text(encoding="utf-8")
    if filename not in index_html:
        new_card = f"""
      <!-- {week_str} placeholder -->
      <a href="posts/{filename}" class="post-card">
        <span class="week-tag">{week_str} (예정)</span>
        <h3>다음 주 논문 리뷰 제안 — {year}년 {month}월 {week_of_month}주차</h3>
        <p class="post-desc">
          다음 주에 리뷰할 논문이나 주제를 댓글로 제안해주세요.
        </p>
        <div class="post-footer">
          <span class="paper-count">
            <span class="icon">&#128161;</span> 제안 받는 중
          </span>
          <span>{monday.strftime('%Y.%m.%d')} 예정</span>
        </div>
      </a>"""

        marker = '<div class="post-list">'
        if marker in index_html:
            index_html = index_html.replace(marker, marker + new_card, 1)
            INDEX_PATH.write_text(index_html, encoding="utf-8")
            print(f"[다음주] index.html에 제안 카드 추가")


def update_index(week_info, paper_count):
    """index.html에 새 리뷰 카드 추가"""
    week_str = week_info["week_str"]
    monday = week_info["monday"]
    month = monday.month
    week_of_month = (monday.day - 1) // 7 + 1
    filename = f"{week_str.lower()}.html"

    index_html = INDEX_PATH.read_text(encoding="utf-8")

    # placeholder 카드가 있으면 제거 후 실제 카드로 교체
    placeholder_marker = f"<!-- {week_str} placeholder -->"
    if placeholder_marker in index_html:
        start = index_html.index(placeholder_marker)
        end = index_html.index("</a>", start) + len("</a>")
        index_html = index_html[:start] + index_html[end:]
        print(f"[index] placeholder 카드 제거: {week_str}")
    elif filename in index_html:
        print(f"[index] {filename} 이미 존재 — 건너뜀")
        return

    new_card = f"""
      <!-- {week_str} -->
      <a href="posts/{filename}" class="post-card">
        <span class="week-tag">{week_str}</span>
        <h3>주간 태양/우주기상 논문 리뷰 — {week_info['year']}년 {month}월 {week_of_month}주차</h3>
        <p class="post-desc">
          태양물리·우주기상 분야 최신 논문 {paper_count}편을 자동 검색·요약했습니다.
        </p>
        <div class="post-footer">
          <span class="paper-count">
            <span class="icon">&#128196;</span> {paper_count}편 리뷰
          </span>
          <span>{monday.strftime('%Y.%m.%d')}</span>
          <div class="sources">
            <span>arXiv</span>
            <span>ADS</span>
          </div>
        </div>
      </a>"""

    # <div class="post-list"> 바로 뒤에 삽입
    marker = '<div class="post-list">'
    if marker in index_html:
        index_html = index_html.replace(marker, marker + new_card, 1)
        INDEX_PATH.write_text(index_html, encoding="utf-8")
        print(f"[index] 새 카드 추가 완료: {filename}")
    else:
        print("[index] post-list 마커를 찾을 수 없음")


def main():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")

    week_info = get_week_info()
    print(f"=== 주간 논문 리뷰 생성: {week_info['week_str']} ===")

    # 이전 주차에서 이미 요약한 논문 수집
    previously_reviewed = get_previously_reviewed_titles()

    # Giscus 댓글에서 논문/주제 제안 수집
    giscus_papers, giscus_keywords = fetch_giscus_suggestions(week_info["week_str"])

    # 발제된 논문/주제 확인 (Giscus 제안 + topic_proposals.json)
    proposals_file = BASE_DIR / "data" / "topic_proposals.json"
    proposed_papers = list(giscus_papers)
    proposed_keywords = list(giscus_keywords)
    if proposals_file.exists():
        proposals = json.loads(proposals_file.read_text(encoding="utf-8"))
        for p in proposals.get("proposals", []):
            if p.get("type") == "paper" and p.get("url"):
                proposed_papers.append(p)
                print(f"[발제] 논문: {p.get('title', p['url'])}")
            elif p.get("type") == "topic" and p.get("content"):
                proposed_keywords.append(p["content"])
                print(f"[발제] 주제: {p['content']}")
        # 발제 처리 후 비우기
        proposals["proposals"] = []
        proposals_file.write_text(
            json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 논문 검색 (1년 이내)
    arxiv_papers = fetch_arxiv_papers(max_results=50)
    ads_papers = fetch_ads_papers(max_results=50)

    # 발제된 논문을 우선 포함
    for pp in proposed_papers:
        all_papers_manual = [{
            "title": pp.get("title", "발제 논문"),
            "authors": [pp.get("name", "발제자")],
            "abstract": pp.get("content", ""),
            "url": pp["url"],
            "date": "",
            "source": "발제",
            "journal": "",
        }]
        arxiv_papers = all_papers_manual + arxiv_papers

    all_papers = arxiv_papers + ads_papers
    all_papers = deduplicate_papers(all_papers, previously_reviewed)
    print(f"[총] 중복 제거 후 {len(all_papers)}편")

    if not all_papers:
        print("검색된 논문이 없습니다. 종료합니다.")
        return

    # Claude로 요약 (발제 주제가 있으면 우선 반영)
    if proposed_keywords:
        print(f"[발제] 주제 키워드 {len(proposed_keywords)}개를 선별에 반영")
    summaries = select_and_summarize(all_papers, week_info, proposed_keywords)
    if not summaries:
        print("요약 결과가 없습니다. 종료합니다.")
        return

    # 각 논문의 본문 텍스트 추출 (arXiv 논문만)
    print("[PDF] 논문 본문 추출 시작...")
    for s in summaries:
        fulltext = fetch_paper_fulltext(s.get("url", ""))
        s["fulltext"] = fulltext
    fulltext_count = sum(1 for s in summaries if s.get("fulltext"))
    print(f"[PDF] 본문 추출 완료: {fulltext_count}/{len(summaries)}편")

    # 논문 데이터 저장 (토론 스크립트에서 사용)
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    papers_data = {
        "week_info": {
            "year": week_info["year"],
            "week": week_info["week"],
            "week_str": week_info["week_str"],
            "monday": week_info["monday"].strftime("%Y-%m-%d"),
            "sunday": week_info["sunday"].strftime("%Y-%m-%d"),
            "date_range": week_info["date_range"],
            "date_str": week_info["date_str"],
        },
        "papers": summaries,
    }
    (DATA_DIR / "current_papers.json").write_text(
        json.dumps(papers_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[데이터] data/current_papers.json 저장 완료")

    # HTML 생성 (토론 없이)
    POSTS_DIR.mkdir(exist_ok=True)
    post_html = generate_post_html(summaries, week_info)
    filename = f"{week_info['week_str'].lower()}.html"
    post_path = POSTS_DIR / filename
    post_path.write_text(post_html, encoding="utf-8")
    print(f"[파일] {post_path} 생성 완료")

    # 개별 논문 토론 페이지 생성
    generate_paper_pages(summaries, week_info)

    # index.html 업데이트
    update_index(week_info, len(summaries))

    # 다음 주차 placeholder 생성 (Giscus로 제안 수집)
    next_week_info = get_week_info(offset=1)
    generate_next_week_placeholder(next_week_info)

    print("=== 완료 ===")


if __name__ == "__main__":
    main()
