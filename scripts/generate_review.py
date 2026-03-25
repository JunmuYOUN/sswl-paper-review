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

from bs4 import BeautifulSoup

import anthropic
import google.generativeai as genai
import requests

# ── 설정 ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ADS_API_KEY = os.environ.get("ADS_API_KEY")

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


def get_week_info():
    """현재 ISO 주차 정보 반환"""
    today = datetime.utcnow()
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


def select_and_summarize(papers, week_info):
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


def _ask_claude(prompt):
    """Claude API 호출"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _ask_gemini(prompt):
    """Gemini API 호출"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_discussion(paper):
    """논문에 대해 Claude(이론가)와 Gemini(관측자)가 토론 후 종합"""
    print(f"  [토론] {paper['title'][:50]}...")

    paper_context = (
        f"제목: {paper['title']}\n"
        f"저자: {paper['authors']}\n"
        f"요약: {paper['one_line_summary']}\n"
        f"배경: {paper['background']}\n"
        f"방법: {paper['method']}\n"
        f"발견: {paper['findings']}\n"
        f"의의: {paper['significance']}\n"
    )

    base_instruction = (
        "한국어로 답변하세요. 전문 용어 첫 등장 시 영문을 병기하세요. "
        "3~5문장으로 간결하게 답변하세요."
    )

    # 1단계: Claude (이론가 관점)
    claude_prompt = (
        f"당신은 '이론가' — 태양물리학 이론 및 수치 모델링 전문가입니다.\n"
        f"아래 논문을 이론적 관점에서 평가하세요:\n"
        f"- 이론적 기여와 참신성\n"
        f"- 기존 이론/모델과의 관계\n"
        f"- 이론적 한계나 추가 검증 필요성\n\n"
        f"{base_instruction}\n\n{paper_context}"
    )
    claude_opinion = _ask_claude(claude_prompt)

    # 2단계: Gemini (관측자 관점, Claude 의견 참고)
    gemini_prompt = (
        f"당신은 '관측자' — 태양 관측 데이터 분석 및 우주기상 예보 전문가입니다.\n"
        f"아래 논문을 관측/실험적 관점에서 평가하세요:\n"
        f"- 데이터 품질과 분석 방법론의 적절성\n"
        f"- 관측적 검증 가능성\n"
        f"- 우주기상 예보 실무 적용 가능성\n\n"
        f"아래는 이론가의 의견입니다. 동의하거나 반박할 수 있습니다:\n"
        f"[이론가 의견] {claude_opinion}\n\n"
        f"{base_instruction}\n\n{paper_context}"
    )
    gemini_opinion = _ask_gemini(gemini_prompt)

    # 3단계: Claude가 종합 정리
    synthesis_prompt = (
        f"아래 논문에 대해 두 전문가가 토론했습니다. 핵심 합의점과 쟁점을 2~3문장으로 종합하세요.\n\n"
        f"[이론가(Claude)] {claude_opinion}\n\n"
        f"[관측자(Gemini)] {gemini_opinion}\n\n"
        f"{base_instruction}\n\n논문: {paper['title']}"
    )
    synthesis = _ask_claude(synthesis_prompt)

    return {
        "theorist": claude_opinion,
        "observer": gemini_opinion,
        "synthesis": synthesis,
    }


def run_discussions(papers):
    """모든 논문에 대해 에이전트 토론 실행"""
    if not GEMINI_API_KEY:
        print("[토론] GEMINI_API_KEY 없음 — 토론 건너뜀")
        return {i: None for i in range(len(papers))}

    print(f"[토론] {len(papers)}편 논문에 대해 AI 에이전트 토론 시작...")
    discussions = {}
    for i, paper in enumerate(papers):
        try:
            discussions[i] = generate_discussion(paper)
        except Exception as e:
            print(f"  [토론 오류] 논문 {i+1}: {e}")
            discussions[i] = None
    print("[토론] 완료")
    return discussions


def generate_post_html(papers, week_info, discussions=None):
    """주간 리뷰 HTML 파일 생성"""
    week_str = week_info["week_str"]
    year = week_info["year"]
    week_num = week_info["week"]
    monday = week_info["monday"]
    sunday = week_info["sunday"]

    # 월 주차 계산
    month = monday.month
    week_of_month = (monday.day - 1) // 7 + 1

    if discussions is None:
        discussions = {}

    paper_cards = ""
    for i, p in enumerate(papers, 1):
        disc = discussions.get(i - 1)
        discussion_html = ""
        if disc:
            discussion_html = f"""
      <div class="discussion-section">
        <div class="discussion-title">AI 에이전트 토론</div>
        <div class="discussion-thread">
          <div class="agent-comment theorist">
            <div class="agent-badge claude-badge">
              <span class="agent-icon">C</span> 이론가 <span class="agent-model">Claude</span>
            </div>
            <p>{disc['theorist']}</p>
          </div>
          <div class="agent-comment observer">
            <div class="agent-badge gemini-badge">
              <span class="agent-icon">G</span> 관측자 <span class="agent-model">Gemini</span>
            </div>
            <p>{disc['observer']}</p>
          </div>
          <div class="agent-synthesis">
            <div class="synthesis-label">종합</div>
            <p>{disc['synthesis']}</p>
          </div>
        </div>
      </div>"""

        paper_cards += f"""
    <!-- Paper {i} -->
    <article class="paper-card">
      <div class="paper-num">{i}</div>
      <h2>
        <a href="{p['url']}" target="_blank">
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

      <div class="summary-section">
        <div class="label">연구 배경</div>
        <p>{p['background']}</p>
      </div>

      <div class="summary-section">
        <div class="label">연구 방법</div>
        <p>{p['method']}</p>
      </div>

      <div class="summary-section">
        <div class="label">핵심 발견</div>
        <p>{p['findings']}</p>
      </div>

      <div class="summary-section">
        <div class="label">의의 및 시사점</div>
        <p>{p['significance']}</p>
      </div>
{discussion_html}
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

    .summary-section {{
      margin-bottom: 12px;
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
      margin-bottom: 18px;
      font-size: 0.95rem;
      font-weight: 500;
      color: var(--text-primary);
      line-height: 1.6;
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

    .discussion-thread {{
      display: flex;
      flex-direction: column;
      gap: 12px;
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

    .agent-comment p {{
      color: var(--text-secondary);
      margin: 0;
    }}

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

    .agent-model {{
      font-weight: 400;
      color: var(--text-light);
      font-size: 0.72rem;
    }}

    .agent-synthesis {{
      background: linear-gradient(135deg, rgba(232,168,56,0.06), rgba(100,181,198,0.06));
      border: 1px solid rgba(232,168,56,0.2);
      border-radius: 10px;
      padding: 14px 18px;
    }}

    .synthesis-label {{
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--sun-orange);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}

    .agent-synthesis p {{
      font-size: 0.9rem;
      color: var(--text-secondary);
      line-height: 1.7;
      margin: 0;
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


def update_index(week_info, paper_count):
    """index.html에 새 리뷰 카드 추가"""
    week_str = week_info["week_str"]
    monday = week_info["monday"]
    month = monday.month
    week_of_month = (monday.day - 1) // 7 + 1
    filename = f"{week_str.lower()}.html"

    # 이미 존재하는지 확인
    index_html = INDEX_PATH.read_text(encoding="utf-8")
    if filename in index_html:
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

    # 논문 검색 (1년 이내)
    arxiv_papers = fetch_arxiv_papers(max_results=50)
    ads_papers = fetch_ads_papers(max_results=50)

    all_papers = arxiv_papers + ads_papers
    all_papers = deduplicate_papers(all_papers, previously_reviewed)
    print(f"[총] 중복 제거 후 {len(all_papers)}편")

    if not all_papers:
        print("검색된 논문이 없습니다. 종료합니다.")
        return

    # Claude로 요약
    summaries = select_and_summarize(all_papers, week_info)
    if not summaries:
        print("요약 결과가 없습니다. 종료합니다.")
        return

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

    # index.html 업데이트
    update_index(week_info, len(summaries))

    print("=== 완료 ===")


if __name__ == "__main__":
    main()
