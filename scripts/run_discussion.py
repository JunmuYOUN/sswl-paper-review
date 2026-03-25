#!/usr/bin/env python3
"""
AI 에이전트 라운드별 토론 스크립트 (5개 에이전트)
- 매시간 1라운드씩 실행 (1AM~5AM KST)
- Round 1: 이론가(Claude) — 이론적 평가
- Round 2: 관측자(Gemini) — 관측적 평가 + 이론가에 반응
- Round 3: 비평가(Claude) — 한계점·개선방향 지적
- Round 4: 학생(Gemini) — 대학원생이 어려워할 부분 질문
- Round 5: 교수(Claude) — 학생 질문 답변 + 전체 종합
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

TOTAL_ROUNDS = 5

# ── 에이전트 정의 ─────────────────────────────────────
AGENT_STYLES = {
    "theorist": {"css_class": "theorist", "icon": "T", "color_class": "claude-badge"},
    "observer": {"css_class": "observer", "icon": "O", "color_class": "gemini-badge"},
    "critic":   {"css_class": "critic",   "icon": "!", "color_class": "critic-badge"},
    "student":  {"css_class": "student",  "icon": "?", "color_class": "student-badge"},
    "professor":{"css_class": "professor","icon": "P", "color_class": "professor-badge"},
}

# ── 라운드별 설정 ─────────────────────────────────────
ROUNDS = {
    1: {
        "agent": "theorist",
        "model": "claude",
        "label": "이론가",
        "model_label": "Claude",
        "instruction": (
            "당신은 '이론가' — 태양물리학 이론 및 수치 모델링 전문가입니다.\n"
            "아래 논문에 대해 이론적 관점에서 첫 의견을 제시하세요:\n"
            "- 이론적 기여와 참신성\n"
            "- 기존 이론/모델과의 관계\n"
            "- 이론적 한계나 추가 검증 필요성"
        ),
    },
    2: {
        "agent": "observer",
        "model": "gemini",
        "label": "관측자",
        "model_label": "Gemini",
        "instruction": (
            "당신은 '관측자' — 태양 관측 데이터 분석 및 우주기상 예보 전문가입니다.\n"
            "이론가의 의견을 참고하여, 관측/실험적 관점에서 평가하세요:\n"
            "- 이론가 의견에 동의하거나 반박할 부분\n"
            "- 데이터 품질과 분석 방법론의 적절성\n"
            "- 우주기상 예보 실무 적용 가능성"
        ),
    },
    3: {
        "agent": "critic",
        "model": "claude",
        "label": "비평가",
        "model_label": "Claude",
        "instruction": (
            "당신은 '비평가' — 학술 논문 심사 경험이 풍부한 비판적 리뷰어입니다.\n"
            "이론가와 관측자의 토론을 읽고, 제3자 관점에서 냉정하게 평가하세요:\n"
            "- 이 연구의 근본적 한계와 약점\n"
            "- 이론가와 관측자가 놓친 문제점\n"
            "- 구체적인 개선 방향과 후속 연구 제안\n"
            "긍정적인 면도 인정하되, 비판적 시각을 유지하세요."
        ),
    },
    4: {
        "agent": "student",
        "model": "gemini",
        "label": "학생",
        "model_label": "Gemini",
        "instruction": (
            "당신은 '대학원생' — 태양물리학을 공부하는 석사과정 학생입니다.\n"
            "위 논문 요약과 전문가들의 토론을 읽었지만, 이해가 어려운 부분이 있습니다.\n"
            "대학원생 수준의 독자가 궁금해할 만한 질문을 3~4개 제시하세요:\n"
            "- 전문 용어나 개념 중 추가 설명이 필요한 것\n"
            "- 방법론에서 '왜 이 방법을 선택했는지' 궁금한 부분\n"
            "- 결과 해석에서 직관적으로 이해하기 어려운 부분\n"
            "- 이 연구가 본인의 연구에 어떻게 활용될 수 있을지\n"
            "각 질문에 번호를 매겨주세요."
        ),
    },
    5: {
        "agent": "professor",
        "model": "claude",
        "label": "교수",
        "model_label": "Claude",
        "instruction": (
            "당신은 '교수' — 20년 경력의 태양물리학 교수이자 대학원 지도교수입니다.\n"
            "학생의 질문에 친절하고 명확하게 답변하세요.\n"
            "답변 시 이전 라운드의 이론가·관측자·비평가 의견을 활용할 수 있습니다.\n"
            "- 학생의 각 질문에 번호별로 답변\n"
            "- 전문 용어는 쉽게 풀어서 설명\n"
            "- 필요하면 비유나 예시를 사용\n"
            "마지막에 전체 토론을 2~3문장으로 종합해주세요."
        ),
    },
}


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


def load_state():
    """토론 상태 로드"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return None


def save_state(state):
    """토론 상태 저장"""
    DATA_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def init_state(papers_data):
    """논문 데이터로부터 초기 토론 상태 생성"""
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


def build_prompt(round_info, paper, thread):
    """라운드별 프롬프트 생성"""
    paper_context = (
        f"제목: {paper['title']}\n"
        f"저자: {paper['authors']}\n"
        f"요약: {paper['one_line_summary']}\n"
        f"배경: {paper['background']}\n"
        f"방법: {paper['method']}\n"
        f"발견: {paper['findings']}\n"
        f"의의: {paper['significance']}\n"
    )

    thread_text = ""
    if thread:
        thread_text = "\n\n=== 이전 토론 내용 ===\n"
        for t in thread:
            thread_text += f"\n[{t['label']}({t['model_label']}) — 라운드 {t['round']}]\n{t['content']}\n"
        thread_text += "\n=== 이전 토론 끝 ===\n"

    length_guide = "3~5문장으로 간결하게 답변하세요."
    if round_info["agent"] == "student":
        length_guide = "3~4개의 질문을 제시하세요. 각 질문은 1~2문장입니다."
    elif round_info["agent"] == "professor":
        length_guide = "각 질문에 2~3문장으로 답변하고, 마지막에 2~3문장으로 전체 종합을 작성하세요."

    prompt = (
        f"{round_info['instruction']}\n\n"
        f"한국어로 답변하세요. 전문 용어 첫 등장 시 영문을 병기하세요. "
        f"{length_guide}\n\n"
        f"{paper_context}"
        f"{thread_text}"
    )
    return prompt


def run_round(state, round_num):
    """특정 라운드 실행"""
    round_info = ROUNDS[round_num]
    model_type = round_info["model"]

    print(f"=== 라운드 {round_num}/{TOTAL_ROUNDS}: {round_info['label']}({round_info['model_label']}) ===")

    for i, paper in enumerate(state["papers"]):
        print(f"  [{i+1}/{len(state['papers'])}] {paper['title'][:50]}...")

        prompt = build_prompt(round_info, paper, paper["thread"])

        if model_type == "claude":
            content = _ask_claude(prompt)
        else:
            content = _ask_gemini(prompt)

        paper["thread"].append({
            "round": round_num,
            "agent": round_info["agent"],
            "label": round_info["label"],
            "model_label": round_info["model_label"],
            "content": content,
        })

    state["current_round"] = round_num
    return state


# ── CSS for discussion ────────────────────────────────
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

    .discussion-thread {
      display: flex;
      flex-direction: column;
      gap: 12px;
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

    .agent-model {
      font-weight: 400;
      color: var(--text-light);
      font-size: 0.72rem;
    }
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

    # 기존 discussion-section 제거 (재실행 시)
    for old in soup.select(".discussion-section"):
        old.decompose()

    paper_cards = soup.select(".paper-card")

    for i, paper in enumerate(state["papers"]):
        if i >= len(paper_cards):
            break

        card = paper_cards[i]
        thread = paper.get("thread", [])
        if not thread:
            continue

        # 토론 HTML 생성
        discussion_div = soup.new_tag("div")
        discussion_div["class"] = "discussion-section"

        title_div = soup.new_tag("div")
        title_div["class"] = "discussion-title"
        round_num = state["current_round"]
        title_div.string = f"AI 에이전트 토론 ({round_num}/{TOTAL_ROUNDS} 라운드 진행)"
        discussion_div.append(title_div)

        thread_div = soup.new_tag("div")
        thread_div["class"] = "discussion-thread"

        for t in thread:
            agent_key = t["agent"]
            style = AGENT_STYLES.get(agent_key, AGENT_STYLES["theorist"])

            comment_div = soup.new_tag("div")
            comment_div["class"] = f"agent-comment {style['css_class']}"

            # 배지
            badge_div = soup.new_tag("div")
            badge_div["class"] = f"agent-badge {style['color_class']}"

            icon_span = soup.new_tag("span")
            icon_span["class"] = "agent-icon"
            icon_span.string = style["icon"]
            badge_div.append(icon_span)

            badge_div.append(f" {t['label']} ")

            model_span = soup.new_tag("span")
            model_span["class"] = "agent-model"
            model_span.string = f"{t['model_label']} · R{t['round']}"
            badge_div.append(model_span)

            comment_div.append(badge_div)

            p_tag = soup.new_tag("p")
            p_tag.string = t["content"]
            comment_div.append(p_tag)

            thread_div.append(comment_div)

        discussion_div.append(thread_div)

        # vote-bar 앞에 삽입
        vote_bar = card.select_one(".vote-bar")
        if vote_bar:
            vote_bar.insert_before(discussion_div)
        else:
            card.append(discussion_div)

    # discussion CSS 추가
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

    # 논문 데이터 로드
    if not PAPERS_FILE.exists():
        print("current_papers.json 파일 없음 — 종료합니다.")
        return

    papers_data = json.loads(PAPERS_FILE.read_text(encoding="utf-8"))

    # 상태 로드 또는 초기화
    state = load_state()
    if state is None or state["week_str"] != papers_data["week_info"]["week_str"]:
        print("[상태] 새 주차 — 토론 초기화")
        state = init_state(papers_data)

    current_round = state["current_round"]
    next_round = current_round + 1

    if next_round > TOTAL_ROUNDS:
        print(f"[완료] 이미 모든 라운드({TOTAL_ROUNDS}) 완료됨")
        return

    print(f"[진행] 현재 라운드: {current_round}, 다음: {next_round}/{TOTAL_ROUNDS}")

    # 라운드 실행
    state = run_round(state, next_round)
    save_state(state)
    print(f"[저장] 라운드 {next_round} 상태 저장 완료")

    # 매 라운드마다 HTML 업데이트 (진행 상황 반영)
    update_post_html(state, papers_data)

    if next_round == TOTAL_ROUNDS:
        print("=== 모든 토론 라운드 완료! ===")
    else:
        remaining = TOTAL_ROUNDS - next_round
        print(f"=== 라운드 {next_round}/{TOTAL_ROUNDS} 완료. 남은 라운드: {remaining} ===")


if __name__ == "__main__":
    main()
