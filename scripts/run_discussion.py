#!/usr/bin/env python3
"""
AI 에이전트 일별 토론 스크립트 (5개 에이전트 + 사람 참여)

스케줄: 화~금 매일 01:00 KST (4일간)
- 매일 모든 에이전트에게 기회 부여
- 할 말이 있는 에이전트만 참여 (PASS 가능)
- 사람 댓글(human_comments.json)을 읽어서 에이전트가 참고·답변
- 금요일(Day 4)에 교수가 최종 종합
"""

import os
import json
from pathlib import Path

import anthropic
import google.generativeai as genai

# ── 설정 ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POSTS_DIR = BASE_DIR / "posts"

PAPERS_FILE = DATA_DIR / "current_papers.json"
STATE_FILE = DATA_DIR / "discussion_state.json"
HUMAN_COMMENTS_FILE = DATA_DIR / "human_comments.json"
TOPIC_PROPOSALS_FILE = DATA_DIR / "topic_proposals.json"

TOTAL_ROUNDS = 12  # 화~금 × 하루 3회
ROUNDS_PER_DAY = 3

# ── 에이전트 정의 ─────────────────────────────────────
AGENTS = [
    {
        "id": "theorist",
        "model": "claude",
        "label": "이론가",
        "model_label": "Claude",
        "css_class": "theorist",
        "icon": "T",
        "color_class": "claude-badge",
        "system": (
            "당신은 '이론가' — 태양물리학 이론 및 수치 모델링 전문가입니다.\n"
            "이론적 기여, 참신성, 기존 모델과의 관계, 이론적 한계 등을 평가합니다."
        ),
    },
    {
        "id": "observer",
        "model": "gemini",
        "label": "관측자",
        "model_label": "Gemini",
        "css_class": "observer",
        "icon": "O",
        "color_class": "gemini-badge",
        "system": (
            "당신은 '관측자' — 태양 관측 데이터 분석 및 우주기상 예보 전문가입니다.\n"
            "데이터 품질, 분석 방법론, 관측적 검증 가능성, 실무 적용성을 평가합니다."
        ),
    },
    {
        "id": "critic",
        "model": "claude",
        "label": "비평가",
        "model_label": "Claude",
        "css_class": "critic",
        "icon": "!",
        "color_class": "critic-badge",
        "system": (
            "당신은 '비평가' — 학술 논문 심사 경험이 풍부한 비판적 리뷰어입니다.\n"
            "연구의 근본적 한계, 다른 참가자가 놓친 문제점, 개선 방향을 냉정하게 지적합니다."
        ),
    },
    {
        "id": "student",
        "model": "gemini",
        "label": "학생",
        "model_label": "Gemini",
        "css_class": "student",
        "icon": "?",
        "color_class": "student-badge",
        "system": (
            "당신은 '대학원생' — 태양물리학을 공부하는 석사과정 학생입니다.\n"
            "이해가 어려운 전문 용어, 방법론의 이유, 결과 해석 등에 대해 질문합니다.\n"
            "대학원생 수준의 독자가 궁금해할 만한 것을 찾아 질문하세요."
        ),
    },
    {
        "id": "professor",
        "model": "claude",
        "label": "교수",
        "model_label": "Claude",
        "css_class": "professor",
        "icon": "P",
        "color_class": "professor-badge",
        "system": (
            "당신은 '교수' — 20년 경력의 태양물리학 교수이자 대학원 지도교수입니다.\n"
            "학생이나 사람의 질문에 친절하고 명확하게 답변합니다.\n"
            "전문 용어는 쉽게 풀어서 설명하고, 필요하면 비유나 예시를 사용합니다."
        ),
    },
]

AGENT_MAP = {a["id"]: a for a in AGENTS}


def _ask_claude(system, prompt):
    """Claude API 호출 (system prompt 분리)"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _ask_gemini(system, prompt):
    """Gemini API 호출"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system,
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return None


def save_state(state):
    DATA_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_human_comments():
    """사람 댓글 로드"""
    if HUMAN_COMMENTS_FILE.exists():
        return json.loads(HUMAN_COMMENTS_FILE.read_text(encoding="utf-8"))
    return {}


def load_topic_proposals():
    """다음 주차 논문/주제 발제 로드"""
    if TOPIC_PROPOSALS_FILE.exists():
        return json.loads(TOPIC_PROPOSALS_FILE.read_text(encoding="utf-8"))
    return {}


def round_to_day(round_num):
    """라운드 번호 → 요일 (1~4)과 하루 내 세션 (1~3)"""
    day = (round_num - 1) // ROUNDS_PER_DAY + 1
    session = (round_num - 1) % ROUNDS_PER_DAY + 1
    return day, session


def day_session_label(round_num):
    """라운드 → 사람이 읽기 쉬운 라벨"""
    day, session = round_to_day(round_num)
    day_names = {1: "화", 2: "수", 3: "목", 4: "금"}
    session_names = {1: "오전", 2: "오후", 3: "저녁"}
    return f"{day_names.get(day, f'D{day}')} {session_names.get(session, f'S{session}')}"


def init_state(papers_data):
    state = {
        "week_str": papers_data["week_info"]["week_str"],
        "current_round": 0,
        "papers": [],
    }
    for p in papers_data["papers"]:
        state["papers"].append({
            "title": p["title"],
            "authors": p["authors"],
            "one_line_summary": p["one_line_summary"],
            "background": p["background"],
            "method": p["method"],
            "findings": p["findings"],
            "significance": p["significance"],
            "thread": [],
        })
    return state


def build_thread_text(thread):
    """스레드를 텍스트로 변환"""
    if not thread:
        return ""
    text = "\n=== 지금까지의 토론 ===\n"
    for t in thread:
        r = t.get("round", 0)
        label = day_session_label(r) if r else "?"
        prefix = f"[{t['label']}"
        if t.get("name"):
            prefix += f" — {t['name']}"
        if t.get("model_label"):
            prefix += f" ({t['model_label']})"
        prefix += f", {label}]"
        text += f"\n{prefix}\n{t['content']}\n"
    text += "\n=== 토론 끝 ===\n"
    return text


def build_paper_context(paper):
    return (
        f"제목: {paper['title']}\n"
        f"저자: {paper['authors']}\n"
        f"요약: {paper['one_line_summary']}\n"
        f"배경: {paper['background']}\n"
        f"방법: {paper['method']}\n"
        f"발견: {paper['findings']}\n"
        f"의의: {paper['significance']}\n"
    )


def run_agent(agent, paper, thread, round_num):
    """개별 에이전트 실행. PASS면 None 반환."""
    paper_context = build_paper_context(paper)
    thread_text = build_thread_text(thread)
    is_final = (round_num == TOTAL_ROUNDS)
    label = day_session_label(round_num)

    # 사람 질문이 있는지 확인
    human_comments = [t for t in thread if t.get("agent") == "human"]
    human_mention = ""
    if human_comments:
        human_mention = "\n참고: 사람(연구원)이 남긴 댓글이 있습니다. 관련 있다면 답변에 반영하세요.\n"

    # 다음 주차 발제가 있는지
    proposals = load_topic_proposals()
    proposal_mention = ""
    if proposals.get("proposals"):
        proposal_mention = "\n참고: 다음 주차에 대한 발제/주제 제안이 있습니다:\n"
        for prop in proposals["proposals"]:
            proposal_mention += f"- [{prop.get('name', '익명')}] {prop.get('content', '')}\n"
        proposal_mention += "관련이 있다면 발제 내용도 참고하여 논의하세요.\n"

    context = f"현재 {label} (라운드 {round_num}/{TOTAL_ROUNDS})입니다."
    if is_final:
        context += " 이번 주 마지막 세션입니다."
        if agent["id"] == "professor":
            context += " 전체 토론을 종합해주세요."

    prompt = (
        f"{context}\n"
        f"{human_mention}"
        f"{proposal_mention}\n"
        f"아래 논문에 대한 토론에 참여하고 있습니다.\n"
        f"이전 토론을 읽고, 새로 기여할 의견이 있다면 한국어로 작성하세요.\n"
        f"전문 용어 첫 등장 시 영문을 병기하세요. 3~5문장으로 간결하게.\n"
        f"기여할 내용이 없다면 정확히 'PASS'라고만 답변하세요.\n\n"
        f"--- 논문 정보 ---\n{paper_context}\n"
        f"{thread_text}"
    )

    system = agent["system"]

    if agent["model"] == "claude":
        response = _ask_claude(system, prompt)
    else:
        response = _ask_gemini(system, prompt)

    if response.strip().upper() == "PASS":
        return None
    return response


def inject_human_comments(state, human_comments, round_num):
    """사람 댓글을 스레드에 삽입 (아직 삽입 안 된 것만)"""
    for i, paper in enumerate(state["papers"]):
        key = f"paper-{i}"
        comments = human_comments.get(key, [])
        existing_human = {
            (t.get("name", ""), t.get("content", ""))
            for t in paper["thread"]
            if t.get("agent") == "human"
        }
        for c in comments:
            ident = (c.get("name", ""), c.get("content", ""))
            if ident not in existing_human:
                paper["thread"].append({
                    "round": round_num,
                    "agent": "human",
                    "label": "연구원",
                    "name": c.get("name", "익명"),
                    "model_label": "",
                    "content": c["content"],
                })
                print(f"  [사람] {c.get('name', '익명')}의 댓글 추가 (논문 {i+1})")


def run_round(state, round_num):
    """라운드 실행 — 모든 에이전트에게 기회 부여"""
    label = day_session_label(round_num)
    is_final = (round_num == TOTAL_ROUNDS)
    print(f"=== 라운드 {round_num}/{TOTAL_ROUNDS} ({label}) {'[최종]' if is_final else ''} ===")

    for pi, paper in enumerate(state["papers"]):
        print(f"\n[논문 {pi+1}] {paper['title'][:50]}...")
        contributions = 0

        for agent in AGENTS:
            print(f"  [{agent['label']}] ", end="")
            try:
                result = run_agent(agent, paper, paper["thread"], round_num)
            except Exception as e:
                print(f"오류: {e}")
                continue

            if result is None:
                print("PASS")
            else:
                paper["thread"].append({
                    "round": round_num,
                    "agent": agent["id"],
                    "label": agent["label"],
                    "model_label": agent["model_label"],
                    "content": result,
                })
                contributions += 1
                print(f"참여 ({len(result)}자)")

        print(f"  → {label} 참여: {contributions}명")

    state["current_round"] = round_num
    return state


# ── CSS ───────────────────────────────────────────────
DISCUSSION_CSS = """
    /* Discussion Section */
    .discussion-section {
      margin-top: 24px;
      padding-top: 20px;
      border-top: 1px dashed var(--border-subtle);
    }

    .discussion-title {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--nebula-blue);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .discussion-title::before {
      content: '';
      display: inline-block;
      width: 18px;
      height: 18px;
      background: linear-gradient(135deg, var(--sun-gold), var(--aurora-cyan));
      border-radius: 4px;
    }

    .discussion-thread { display: flex; flex-direction: column; gap: 12px; }

    .day-divider {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 8px 0;
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .day-divider::before, .day-divider::after {
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border-subtle);
    }

    .agent-comment {
      padding: 16px 18px;
      border-radius: 10px;
      font-size: 0.9rem;
      line-height: 1.7;
    }

    .agent-comment.theorist {
      background: rgba(147, 51, 234, 0.04);
      border: 1px solid rgba(147, 51, 234, 0.15);
    }
    .agent-comment.observer {
      background: rgba(66, 133, 244, 0.04);
      border: 1px solid rgba(66, 133, 244, 0.15);
    }
    .agent-comment.critic {
      background: rgba(234, 88, 12, 0.04);
      border: 1px solid rgba(234, 88, 12, 0.15);
    }
    .agent-comment.student {
      background: rgba(22, 163, 74, 0.04);
      border: 1px solid rgba(22, 163, 74, 0.15);
    }
    .agent-comment.professor {
      background: linear-gradient(135deg, rgba(232,168,56,0.06), rgba(100,181,198,0.06));
      border: 1px solid rgba(232,168,56,0.2);
    }
    .agent-comment.human {
      background: rgba(59, 130, 246, 0.04);
      border: 1px solid rgba(59, 130, 246, 0.25);
      border-left: 3px solid #3B82F6;
    }

    .agent-comment p { color: var(--text-secondary); margin: 0; white-space: pre-line; }

    .agent-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      margin-bottom: 8px;
    }

    .claude-badge { color: #7C3AED; }
    .gemini-badge { color: #4285F4; }
    .critic-badge { color: #EA580C; }
    .student-badge { color: #16A34A; }
    .professor-badge { color: #B45309; }
    .human-badge { color: #3B82F6; }

    .agent-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      border-radius: 5px;
      font-size: 0.7rem;
      font-weight: 700;
      color: white;
    }

    .claude-badge .agent-icon { background: #7C3AED; }
    .gemini-badge .agent-icon { background: #4285F4; }
    .critic-badge .agent-icon { background: #EA580C; }
    .student-badge .agent-icon { background: #16A34A; }
    .professor-badge .agent-icon { background: #B45309; }
    .human-badge .agent-icon { background: #3B82F6; }

    .agent-model {
      font-weight: 400;
      color: var(--text-light);
      font-size: 0.72rem;
    }

    .human-comment-guide {
      margin-top: 16px;
      padding: 14px 18px;
      background: var(--bg-warm);
      border: 1px dashed var(--border-subtle);
      border-radius: 8px;
      font-size: 0.82rem;
      color: var(--text-light);
      text-align: center;
    }

    .human-comment-guide a {
      color: var(--aurora-cyan);
      text-decoration: none;
      font-weight: 600;
    }

    .human-comment-guide a:hover { text-decoration: underline; }
"""


def update_post_html(state, papers_data):
    """토론 결과를 HTML에 반영"""
    week_str = state["week_str"]
    filename = f"{week_str.lower()}.html"
    post_path = POSTS_DIR / filename

    if not post_path.exists():
        print(f"[HTML] {post_path} 파일 없음 — 건너뜀")
        return

    from bs4 import BeautifulSoup

    html = post_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    for old in soup.select(".discussion-section"):
        old.decompose()

    paper_cards = soup.select(".paper-card")

    # GitHub 편집 URL
    repo_url = "https://github.com/JunmuYOUN/sswl-paper-review"
    comments_edit_url = f"{repo_url}/edit/main/data/human_comments.json"

    for i, paper in enumerate(state["papers"]):
        if i >= len(paper_cards):
            break

        card = paper_cards[i]
        thread = paper.get("thread", [])
        if not thread:
            continue

        discussion_div = soup.new_tag("div")
        discussion_div["class"] = "discussion-section"

        title_div = soup.new_tag("div")
        title_div["class"] = "discussion-title"
        cr = state["current_round"]
        cur_day, _ = round_to_day(cr) if cr > 0 else (0, 0)
        title_div.string = f"AI 에이전트 토론 (라운드 {cr}/{TOTAL_ROUNDS})"
        discussion_div.append(title_div)

        thread_div = soup.new_tag("div")
        thread_div["class"] = "discussion-thread"

        # 날짜별 구분선 삽입
        last_day = 0
        for t in thread:
            r = t.get("round", 0)
            day, _ = round_to_day(r) if r > 0 else (0, 0)
            if day > last_day:
                if last_day > 0:
                    day_names = {1: "화요일", 2: "수요일", 3: "목요일", 4: "금요일"}
                    divider = soup.new_tag("div")
                    divider["class"] = "day-divider"
                    divider.string = day_names.get(day, f"Day {day}")
                    thread_div.append(divider)
                last_day = day

            agent_id = t.get("agent", "")
            is_human = (agent_id == "human")

            if is_human:
                css_class = "human"
                color_class = "human-badge"
                icon_char = "@"
                label = t.get("label", "연구원")
                name = t.get("name", "")
                model_label = ""
            else:
                agent_def = AGENT_MAP.get(agent_id, AGENTS[0])
                css_class = agent_def["css_class"]
                color_class = agent_def["color_class"]
                icon_char = agent_def["icon"]
                label = t["label"]
                name = ""
                model_label = t.get("model_label", "")

            comment_div = soup.new_tag("div")
            comment_div["class"] = f"agent-comment {css_class}"

            badge_div = soup.new_tag("div")
            badge_div["class"] = f"agent-badge {color_class}"

            icon_span = soup.new_tag("span")
            icon_span["class"] = "agent-icon"
            icon_span.string = icon_char
            badge_div.append(icon_span)

            badge_text = f" {label}"
            if name:
                badge_text += f" ({name})"
            badge_div.append(badge_text + " ")

            round_label = day_session_label(r) if r else ""
            if model_label:
                model_span = soup.new_tag("span")
                model_span["class"] = "agent-model"
                model_span.string = f"{model_label} · {round_label}"
                badge_div.append(model_span)
            elif round_label:
                rl_span = soup.new_tag("span")
                rl_span["class"] = "agent-model"
                rl_span.string = round_label
                badge_div.append(rl_span)

            comment_div.append(badge_div)

            p_tag = soup.new_tag("p")
            p_tag.string = t["content"]
            comment_div.append(p_tag)

            thread_div.append(comment_div)

        discussion_div.append(thread_div)

        # 사람 댓글 안내
        guide_div = soup.new_tag("div")
        guide_div["class"] = "human-comment-guide"
        guide_link = soup.new_tag("a")
        guide_link["href"] = comments_edit_url
        guide_link["target"] = "_blank"
        guide_link.string = "human_comments.json"
        guide_div.append("토론에 참여하고 싶다면 ")
        guide_div.append(guide_link)
        guide_div.append(f" 파일에서 paper-{i} 항목에 댓글을 추가하세요.")
        discussion_div.append(guide_div)

        vote_bar = card.select_one(".vote-bar")
        if vote_bar:
            vote_bar.insert_before(discussion_div)
        else:
            card.append(discussion_div)

    # CSS 추가
    style_tag = soup.select_one("style")
    if style_tag and ".discussion-section" not in str(style_tag):
        style_tag.string = (style_tag.string or "") + DISCUSSION_CSS

    post_path.write_text(str(soup), encoding="utf-8")
    print(f"[HTML] {post_path} 업데이트 완료")


def main():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    if not PAPERS_FILE.exists():
        print("current_papers.json 파일 없음 — 종료합니다.")
        return

    papers_data = json.loads(PAPERS_FILE.read_text(encoding="utf-8"))

    state = load_state()
    if state is None or state["week_str"] != papers_data["week_info"]["week_str"]:
        print("[상태] 새 주차 — 토론 초기화")
        state = init_state(papers_data)

    current = state["current_round"]
    next_round = current + 1

    if next_round > TOTAL_ROUNDS:
        print(f"[완료] 이미 모든 라운드({TOTAL_ROUNDS}) 완료됨")
        return

    # 사람 댓글 주입
    human_comments = load_human_comments()
    if human_comments:
        inject_human_comments(state, human_comments, next_round)

    # 라운드 실행
    state = run_round(state, next_round)
    save_state(state)

    # HTML 업데이트
    update_post_html(state, papers_data)

    label = day_session_label(next_round)
    if next_round == TOTAL_ROUNDS:
        print(f"=== 모든 토론 완료! ({label}) ===")
    else:
        print(f"=== 라운드 {next_round}/{TOTAL_ROUNDS} ({label}) 완료 ===")


if __name__ == "__main__":
    main()
