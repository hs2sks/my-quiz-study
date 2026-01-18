import streamlit as st
from typing import Dict, List

st.set_page_config(page_title="퀴즈", layout="centered", page_icon="📝")

# session_state 초기화
if "current_question_idx" not in st.session_state:
    st.session_state["current_question_idx"] = 0
if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = {}
if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False

# 메인 페이지에서 생성된 questions가 있는지 확인
if "questions" not in st.session_state or not st.session_state["questions"]:
    st.warning("⚠️ 생성된 문항이 없습니다. 먼저 메인 페이지에서 문항을 생성해주세요.")
    if st.button("📄 문항 생성 페이지로 이동"):
        st.switch_page("app.py")
    st.stop()

questions = st.session_state["questions"]
current_idx = st.session_state["current_question_idx"]
total = len(questions)

# 제출 완료 화면
if st.session_state["quiz_submitted"]:
    st.title("🎉 퀴즈 완료!")
    
    # 채점
    correct_count = 0
    results = []
    for q in questions:
        user_answer = st.session_state["quiz_answers"].get(q.qid, "")
        is_correct = user_answer.strip().lower() == q.answer.strip().lower()
        if is_correct:
            correct_count += 1
        results.append({
            "qid": q.qid,
            "prompt": q.prompt,
            "user_answer": user_answer,
            "correct_answer": q.answer,
            "is_correct": is_correct,
            "qtype": q.qtype,
            "choices": q.choices,
        })
    
    score = (correct_count / total) * 100
    
    # 결과 요약
    st.metric("점수", f"{correct_count} / {total}", f"{score:.1f}%")
    
    st.divider()
    
    # 상세 결과
    st.subheader("📊 상세 결과")
    for i, result in enumerate(results, 1):
        with st.expander(f"문항 {i} - {'✅ 정답' if result['is_correct'] else '❌ 오답'}"):
            st.write(f"**문제:** {result['prompt']}")
            if result['qtype'] == "객관식" and result['choices']:
                st.write(f"**보기:** {', '.join(result['choices'])}")
            st.write(f"**내 답:** {result['user_answer'] or '(답안 없음)'}")
            if not result['is_correct']:
                st.write(f"**정답:** {result['correct_answer']}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 다시 풀기", use_container_width=True):
            st.session_state["current_question_idx"] = 0
            st.session_state["quiz_answers"] = {}
            st.session_state["quiz_submitted"] = False
            st.rerun()
    with col2:
        if st.button("📄 메인으로 돌아가기", use_container_width=True):
            st.switch_page("app.py")
    
    st.stop()

# 퀴즈 진행 화면
st.title("📝 퀴즈 풀기")
st.caption(f"문항 {current_idx + 1} / {total}")

# 진행률 바
progress = (current_idx + 1) / total
st.progress(progress)

# 현재 문항
q = questions[current_idx]

# 카드 스타일 컨테이너
with st.container():
    st.markdown(
        f"""
        <div style="
            padding: 2rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 2rem 0;
        ">
            <h3 style="color: #1f77b4; margin-bottom: 1rem;">문항 {current_idx + 1}</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">{q.prompt}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")  # 여백

# 답안 입력
answer_key = f"quiz_answer_{q.qid}"
if q.qtype == "객관식" and q.choices:
    # 객관식
    default_idx = 0
    if q.qid in st.session_state["quiz_answers"]:
        prev_answer = st.session_state["quiz_answers"][q.qid]
        if prev_answer in q.choices:
            default_idx = q.choices.index(prev_answer)
    
    selected = st.radio(
        "답을 선택하세요:",
        q.choices,
        index=default_idx,
        key=answer_key,
    )
    st.session_state["quiz_answers"][q.qid] = selected
else:
    # 주관식
    default_text = st.session_state["quiz_answers"].get(q.qid, "")
    user_input = st.text_input(
        "답을 입력하세요:",
        value=default_text,
        key=answer_key,
    )
    if user_input:
        st.session_state["quiz_answers"][q.qid] = user_input

st.write("")  # 여백

# 네비게이션 버튼
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if current_idx > 0:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state["current_question_idx"] -= 1
            st.rerun()
    else:
        st.write("")  # 빈 공간

with col2:
    if st.button("📄 메인으로", use_container_width=True):
        if st.session_state["quiz_answers"]:
            st.warning("진행 중인 답안이 있습니다. 정말 나가시겠습니까?")
        st.switch_page("app.py")

with col3:
    if current_idx < total - 1:
        if st.button("다음 ➡️", use_container_width=True):
            st.session_state["current_question_idx"] += 1
            st.rerun()
    else:
        if st.button("✅ 제출", use_container_width=True, type="primary"):
            # 답안 체크
            answered = len(st.session_state["quiz_answers"])
            if answered < total:
                st.error(f"⚠️ {total - answered}개 문항이 미응답 상태입니다. 모두 답해주세요.")
            else:
                st.session_state["quiz_submitted"] = True
                st.rerun()

# 문항 미리보기 (하단에 작은 네비게이션)
st.divider()
st.caption("**빠른 이동:**")
cols = st.columns(min(10, total))
for i in range(total):
    with cols[i % 10]:
        is_answered = questions[i].qid in st.session_state["quiz_answers"]
        is_current = i == current_idx
        label = f"{'✓' if is_answered else ''}{i+1}"
        button_type = "primary" if is_current else "secondary"
        if st.button(label, key=f"nav_{i}", use_container_width=True, disabled=is_current):
            st.session_state["current_question_idx"] = i
            st.rerun()
