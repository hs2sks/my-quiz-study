import csv
import difflib
import io
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import google.generativeai as genai
import pytesseract
from PyPDF2 import PdfReader


APP_VERSION = "2.0"


@dataclass
class Question:
    qid: str
    qtype: str
    prompt: str
    answer: str
    choices: Optional[List[str]]
    source: str


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def apply_denoise(image: Image.Image, size: int) -> Image.Image:
    if size <= 1:
        return image
    if size % 2 == 0:
        size += 1
    return image.filter(ImageFilter.MedianFilter(size=size))


def auto_rotate_image(image: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image)
        match = re.search(r"Rotate:\s+(\d+)", osd)
        if match:
            rotate = int(match.group(1))
            if rotate != 0:
                return image.rotate(-rotate, expand=True)
    except Exception:
        return image
    return image


def deskew_image(image: Image.Image) -> Image.Image:
    try:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(bw > 0))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = cv_img.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            cv_img,
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return image


def preprocess_image(
    image: Image.Image,
    threshold: int,
    denoise_size: int,
    auto_rotate: bool,
    deskew: bool,
) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    sharpened = gray.filter(ImageFilter.SHARPEN)
    denoised = apply_denoise(sharpened, denoise_size)
    if auto_rotate:
        denoised = auto_rotate_image(denoised)
    if deskew:
        denoised = deskew_image(denoised)
    if threshold < 255:
        binary = denoised.point(lambda p: 255 if p > threshold else 0, mode="1")
        return binary.convert("L")
    return denoised


def ocr_text_from_pdf(
    file_bytes: bytes,
    dpi: int,
    threshold: int,
    denoise_size: int,
    auto_rotate: bool,
    deskew: bool,
) -> str:
    images = convert_from_bytes(file_bytes, dpi=dpi)
    parts: List[str] = []
    for image in images:
        processed = preprocess_image(
            image,
            threshold,
            denoise_size,
            auto_rotate,
            deskew,
        )
        text = pytesseract.image_to_string(processed, lang="kor+eng")
        parts.append(text or "")
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_text_quality(text: str) -> tuple[bool, str]:
    """텍스트 품질 검증: OCR 오류나 깨진 텍스트 감지"""
    if not text or len(text) < 100:
        return False, "텍스트가 너무 짧습니다. (최소 100자 필요)"
    
    # 한글 비율 확인
    korean_chars = len(re.findall(r"[가-힣]", text))
    total_chars = len(re.sub(r"\s", "", text))
    
    if total_chars == 0:
        return False, "유효한 문자가 없습니다."
    
    korean_ratio = korean_chars / total_chars
    
    # 한글 비율이 너무 낮으면 OCR 오류 가능성
    if korean_ratio < 0.3:
        return False, f"한글 비율이 너무 낮습니다 ({korean_ratio:.1%}). OCR 설정을 확인해주세요."
    
    # 의미 없는 단일 글자 연속 확인 (OCR 오류 패턴)
    single_chars = re.findall(r"\s[가-힣]\s", text)
    if len(single_chars) > len(text) * 0.05:  # 5% 이상이면 의심
        return False, "텍스트에 깨진 글자가 많이 포함되어 있습니다. OCR 설정을 조정해주세요."
    
    return True, "텍스트 품질이 양호합니다."


def validate_question(q: Question, text: str) -> tuple[bool, str]:
    """생성된 문항의 유효성 검증"""
    # 문항이 너무 짧은지 확인
    if len(q.prompt) < 10:
        return False, "문항이 너무 짧습니다."
    
    # 정답이 비어있는지 확인
    if not q.answer or len(q.answer.strip()) < 1:
        return False, "정답이 비어있습니다."
    
    # 객관식인 경우 보기 검증
    if q.qtype == "객관식":
        if not q.choices or len(q.choices) < 2:
            return False, "객관식 보기가 부족합니다."
        
        # 보기에 정답이 포함되어 있는지 확인
        if q.answer not in q.choices:
            return False, "보기에 정답이 포함되어 있지 않습니다."
        
        # 보기가 모두 유효한지 확인 (너무 짧거나 의미 없는 단어)
        for choice in q.choices:
            if len(choice.strip()) < 2:
                return False, f"보기가 너무 짧습니다: '{choice}'"
            
            # 한글이 하나도 없고 영문도 없으면 의미 없는 보기
            if not re.search(r"[가-힣a-zA-Z]", choice):
                return False, f"유효하지 않은 보기: '{choice}'"
    
    # 문항이나 정답에 깨진 글자가 많은지 확인
    combined = q.prompt + " " + q.answer
    # 의미 없는 특수문자나 깨진 패턴 감지
    if re.search(r"[^\w\s가-힣a-zA-Z0-9\(\)\[\]\{\}\.,?!~\-:;\"\'%]", combined):
        return False, "문항에 비정상적인 문자가 포함되어 있습니다."
    
    return True, "문항이 유효합니다."


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    # 한국어/영문 문장 마침표 기준 분리 (간단 규칙)
    raw = re.split(r"(?<=[\.\?!。！？])\s+", text)
    sentences = [s.strip() for s in raw if len(s.strip()) >= 20]
    return sentences


def pick_keyword(sentence: str, difficulty: str) -> Optional[str]:
    candidates = re.findall(r"[A-Za-z0-9가-힣]{3,}", sentence)
    if not candidates:
        return None
    candidates.sort(key=len)
    if difficulty == "쉬움":
        return candidates[-1]
    if difficulty == "어려움":
        return candidates[0]
    return candidates[len(candidates) // 2]


def build_question(
    sentence: str,
    qtype: str,
    word_pool: List[str],
    idx: int,
    difficulty: str,
    num_choices: int,
    distractor_mode: str,
) -> Optional[Question]:
    keyword = pick_keyword(sentence, difficulty)
    if not keyword:
        return None
    blanked = sentence.replace(keyword, "____", 1)
    qid = f"q{idx}"
    if qtype == "주관식":
        return Question(
            qid=qid,
            qtype=qtype,
            prompt=blanked,
            answer=keyword,
            choices=None,
            source=sentence,
        )

    # 객관식
    distractors = [w for w in word_pool if w != keyword]
    distractors = list(dict.fromkeys(distractors))
    
    # 한국어 조사 등 필터링 강화
    stop_suffixes = ['은', '는', '이', '가', '을', '를', '에', '의', '로', '으로']
    filtered = []
    for d in distractors:
        is_clean = True
        for s in stop_suffixes:
            if d.endswith(s):
                is_clean = False
                break
        if is_clean:
            filtered.append(d)
    distractors = filtered if filtered else distractors

    if distractor_mode == "유사 길이":
        target_len = len(keyword)
        distractors.sort(key=lambda w: abs(len(w) - target_len))
    elif distractor_mode == "혼동(유사 문자)":
        distractors.sort(
            key=lambda w: difflib.SequenceMatcher(None, keyword, w).ratio(),
            reverse=True,
        )
    else:
        random.shuffle(distractors)
    choices = [keyword] + distractors[: max(0, num_choices - 1)]
    while len(choices) < num_choices:
        choices.append("기타")
    random.shuffle(choices)
    return Question(
        qid=qid,
        qtype=qtype,
        prompt=blanked,
        answer=keyword,
        choices=choices,
        source=sentence,
    )


def generate_questions(
    sentences: List[str],
    count: int,
    qtype: str,
    difficulty: str,
    num_choices: int,
    distractor_mode: str,
    mixed_choice_ratio: int,
) -> List[Question]:
    if not sentences:
        return []
    word_pool = []
    for s in sentences:
        word_pool.extend(re.findall(r"[A-Za-z0-9가-힣]{3,}", s))
    if difficulty == "쉬움":
        sentences = sorted(sentences, key=len)
    elif difficulty == "어려움":
        sentences = sorted(sentences, key=len, reverse=True)
    else:
        random.shuffle(sentences)

    questions: List[Question] = []
    idx = 1
    types = [qtype]
    if qtype == "혼합":
        types = build_mixed_types(count, mixed_choice_ratio)
    
    # 전체 텍스트 재구성 (검증용)
    full_text = " ".join(sentences)

    for sentence in sentences:
        current_type = types[len(questions)] if len(questions) < len(types) else types[-1]
        q = build_question(
            sentence,
            current_type,
            word_pool,
            idx,
            difficulty,
            num_choices,
            distractor_mode,
        )
        if q:
            # 문항 유효성 검증
            is_valid, _ = validate_question(q, full_text)
            if is_valid:
                questions.append(q)
                idx += 1
        if len(questions) >= count:
            break
    return questions


def export_questions_json(questions: List[Question], answers: Dict[str, str]) -> str:
    payload = []
    for q in questions:
        payload.append(
            {
                "id": q.qid,
                "type": q.qtype,
                "prompt": q.prompt,
                "choices": q.choices,
                "answer": q.answer,
                "user_answer": answers.get(q.qid, ""),
                "source": q.source,
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_questions_csv(questions: List[Question], answers: Dict[str, str]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "type", "prompt", "choices", "answer", "user_answer", "source"])
    for q in questions:
        choices = "|".join(q.choices or [])
        writer.writerow(
            [
                q.qid,
                q.qtype,
                q.prompt,
                choices,
                q.answer,
                answers.get(q.qid, ""),
                q.source,
            ]
        )
    return output.getvalue()


def build_gemini_prompt(
    text: str,
    qtype: str,
    count: int,
    difficulty: str,
    num_choices: int,
    mixed_choice_ratio: int,
) -> str:
    mixed_line = ""
    if qtype == "혼합":
        mixed_line = f"- 혼합 비율: 객관식 {mixed_choice_ratio}% / 주관식 {100 - mixed_choice_ratio}%"
    return f"""
다음 텍스트를 바탕으로 고품질 학습 문제를 만들어 주세요.

**중요 규칙:**
1. 제공된 텍스트의 핵심 내용만을 기반으로 문제를 제작하세요.
2. 텍스트에서 명확하게 언급된 개념, 용어, 사실만 사용하세요.
3. 모든 문제와 보기는 한국어 문법에 맞고 의미가 명확해야 합니다.
4. **객관식 보기의 품질이 매우 중요합니다.**

**문항 요구사항:**
- 문항 수: {count}
- 문항 유형: {qtype}
{mixed_line}
- 난이도: {difficulty}
- 객관식 보기 개수: {num_choices}

**객관식 보기 작성 규칙 (필독):**
- **품사 일치**: 정답이 명사(예: '재취학')라면, 모든 오답도 반드시 명사여야 합니다. ('~하는 것', '~에 대하여' 같은 표현 금지)
- **카테고리 일치**: 정답과 유사한 카테고리의 용어를 오답으로 구성하세요. (예: 교육 행정 용어가 정답이면 오답도 교육 행정 용어로)
- **그럴듯한 오답**: 단순히 본문의 무작위 단어가 아니라, 학습자가 혼동할 수 있는 관련 개념을 오답으로 넣으세요.
- **문장 구조**: 보기는 짧고 간결해야 하며, 모든 보기의 길이와 형태가 유사해야 합니다.

**출력 형식:**
JSON 배열로만 반환하세요.

[
  {{
    "type": "객관식|주관식",
    "prompt": "문항 본문",
    "answer": "정확한 정답",
    "choices": ["정답", "유사개념 오답1", "유사개념 오답2", "유사개념 오답3"]
  }}
]

**참고 텍스트:**
\"\"\"{text}\"\"\"

위 텍스트의 핵심 내용을 기반으로 {count}개의 고품질 문제를 생성하세요.
""".strip()


def build_choices_from_text(
    answer: str,
    text: str,
    num_choices: int,
    existing_choices: List[str] = None
) -> List[str]:
    # 기존 보기가 있으면 활용
    choices = existing_choices if existing_choices else [answer]
    if answer not in choices:
        choices.insert(0, answer)
    
    # 본문에서 의미 있는 명사 위주로 추출 (조사 제거 시도)
    words = re.findall(r"[가-힣]{2,10}", text)
    # 조사나 불필요한 단어 필터링
    stop_suffixes = ['은', '는', '이', '가', '을', '를', '에', '의', '로', '으로', '하며', '하여', '함']
    
    filtered_words = []
    for w in words:
        if w == answer: continue
        # 너무 짧거나 조사로 끝나는 단어 제외 시도
        is_clean = True
        for suffix in stop_suffixes:
            if w.endswith(suffix) and len(w) > 2:
                is_clean = False
                break
        if is_clean:
            filtered_words.append(w)
            
    distractors = list(dict.fromkeys(filtered_words))
    random.shuffle(distractors)
    
    # 필요한 만큼 추가
    while len(choices) < num_choices and distractors:
        cand = distractors.pop(0)
        if cand not in choices:
            choices.append(cand)
            
    while len(choices) < num_choices:
        choices.append(f"기타 옵션 {len(choices)}")
        
    random.shuffle(choices)
    return choices


def build_mixed_types(count: int, mixed_choice_ratio: int) -> List[str]:
    ratio = max(0, min(100, mixed_choice_ratio))
    mc_count = round(count * ratio / 100)
    mc_count = max(0, min(count, mc_count))
    sc_count = count - mc_count
    types = ["객관식"] * mc_count + ["주관식"] * sc_count
    random.shuffle(types)
    return types


def to_display_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name.split("/", 1)[1]
    return model_name


def to_api_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def extract_supporting_snippets(text: str, answer: str, limit: int = 2) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    if answer:
        matched = [s for s in sentences if answer.lower() in s.lower()]
    else:
        matched = []
    snippets = matched[:limit]
    if not snippets:
        snippets = sentences[:limit]
    return snippets


def build_explanation_prompt(
    question: Question,
    snippets: List[str],
) -> str:
    sources = "\n".join([f"{i+1}. {s}" for i, s in enumerate(snippets)])
    return f"""
다음 문제에 대해 해설을 만들어 주세요.
요구사항:
- 반드시 한국어로 작성
- 제공된 자료(출처)에서만 근거를 사용
- 결과는 JSON 형식으로만 반환

문제:
- 유형: {question.qtype}
- 본문: {question.prompt}
- 정답: {question.answer}

출처(첨부 PDF 발췌):
{sources}

JSON 스키마:
{{
  "explanation": "해설 문장",
  "sources": [1,2]
}}
""".strip()


def build_local_explanation(question: Question, snippets: List[str]) -> Dict[str, object]:
    if not snippets:
        return {
            "explanation": "첨부 PDF에서 관련 근거를 찾지 못했습니다.",
            "sources": [],
            "used_gemini": False,
            "model": "",
        }
    explanation = f"정답은 '{question.answer}'입니다. 첨부 PDF의 관련 내용은 다음과 같습니다."
    return {
        "explanation": explanation,
        "sources": snippets,
        "used_gemini": False,
        "model": "",
    }


def generate_explanation_with_gemini(
    question: Question,
    text: str,
    model_name: str,
) -> Dict[str, object]:
    snippets = extract_supporting_snippets(text, question.answer, limit=2)
    if not snippets:
        return build_local_explanation(question, snippets)
    prompt = build_explanation_prompt(question, snippets)
    api_key = get_gemini_api_key()
    if not api_key:
        return build_local_explanation(question, snippets)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(to_api_model_name(model_name))
    response = model.generate_content(prompt)
    raw = response.text or ""
    try:
        data = json.loads(raw)
        explanation = str(data.get("explanation", "")).strip()
        used_sources = data.get("sources", [])
        if not explanation:
            return build_local_explanation(question, snippets)
        source_snippets: List[str] = []
        if isinstance(used_sources, list):
            for idx in used_sources:
                if isinstance(idx, int) and 1 <= idx <= len(snippets):
                    source_snippets.append(snippets[idx - 1])
        if not source_snippets:
            source_snippets = snippets
        return {
            "explanation": explanation,
            "sources": source_snippets,
            "used_gemini": True,
            "model": to_display_model_name(model_name),
        }
    except Exception:
        return build_local_explanation(question, snippets)


def get_gemini_api_key() -> str:
    if "GEMINI_API_KEY" in st.secrets:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    return os.getenv("GEMINI_API_KEY", "").strip()


def generate_questions_with_gemini(
    text: str,
    qtype: str,
    count: int,
    difficulty: str,
    num_choices: int,
    model_name: str,
    mixed_choice_ratio: int,
) -> List[Question]:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. (secrets 또는 환경변수)")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(to_api_model_name(model_name))
    prompt = build_gemini_prompt(
        text,
        qtype,
        count,
        difficulty,
        num_choices,
        mixed_choice_ratio,
    )
    response = model.generate_content(prompt)
    raw = response.text or ""
    
    # JSON 추출 (마크다운 코드 블록 제거)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    
    data = json.loads(raw)
    questions: List[Question] = []
    types = [qtype]
    if qtype == "혼합":
        types = build_mixed_types(count, mixed_choice_ratio)
    
    rejected_count = 0
    for idx, item in enumerate(data, start=1):
        q_type = item.get("type", "주관식")
        if qtype != "혼합":
            q_type = qtype
        elif q_type not in {"객관식", "주관식"}:
            q_type = types[len(questions)] if len(questions) < len(types) else types[-1]
        elif qtype == "혼합":
            q_type = types[len(questions)] if len(questions) < len(types) else types[-1]
        prompt_text = item.get("prompt", "")
        answer = item.get("answer", "")
        choices = item.get("choices")
        
        if q_type == "객관식":
            # Gemini가 준 보기가 있다면 최대한 활용하고 부족한 경우에만 채움
            if isinstance(choices, list) and len(choices) >= 2:
                if len(choices) < num_choices:
                    choices = build_choices_from_text(answer, text, num_choices, existing_choices=choices)
            else:
                choices = build_choices_from_text(answer, text, num_choices)
        else:
            choices = None
        
        # 문항 생성
        q = Question(
            qid=f"q{len(questions)+1}",
            qtype=q_type,
            prompt=prompt_text,
            answer=answer,
            choices=choices,
            source="Gemini",
        )
        
        # 문항 유효성 검증
        is_valid, message = validate_question(q, text)
        if is_valid:
            questions.append(q)
            if len(questions) >= count:
                break
        else:
            rejected_count += 1
            # 너무 많은 문항이 거부되면 경고
            if rejected_count > count:
                break
    
    # 생성된 문항이 요청한 개수보다 적으면 경고
    if len(questions) < count * 0.7:  # 70% 미만이면 경고
        raise RuntimeError(
            f"생성된 문항의 품질이 좋지 않습니다. "
            f"({len(questions)}/{count}개만 유효) "
            f"PDF 텍스트 품질을 확인하거나 OCR 설정을 조정해주세요."
        )
    
    return questions


def list_gemini_models() -> List[str]:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. (secrets 또는 환경변수)")
    genai.configure(api_key=api_key)
    models: List[str] = []
    for model in genai.list_models():
        methods = model.supported_generation_methods or []
        if "generateContent" in methods:
            models.append(to_display_model_name(model.name))
    return models


def reset_state() -> None:
    keys = [
        "pdf_text",
        "questions",
        "answers",
        "graded",
        "wrong_questions",
        "review_mode",
        "explanations",
        "explanations_ready",
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


st.set_page_config(page_title="PDF 문항 생성기", layout="wide")
st.title("PDF 문항 생성기")
st.caption("PDF 내용을 기반으로 간단한 문제를 자동 생성합니다. (Version 2.0)")

# 세션 상태 초기화
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "answers" not in st.session_state:
    st.session_state["answers"] = {}
if "graded" not in st.session_state:
    st.session_state["graded"] = False
if "wrong_questions" not in st.session_state:
    st.session_state["wrong_questions"] = []
if "review_mode" not in st.session_state:
    st.session_state["review_mode"] = False
if "explanations" not in st.session_state:
    st.session_state["explanations"] = {}
if "explanations_ready" not in st.session_state:
    st.session_state["explanations_ready"] = False
if "num_choices" not in st.session_state:
    st.session_state["num_choices"] = 4
if "distractor_mode" not in st.session_state:
    st.session_state["distractor_mode"] = "무작위"
if "preset_on" not in st.session_state:
    st.session_state["preset_on"] = False
if "mixed_choice_ratio" not in st.session_state:
    st.session_state["mixed_choice_ratio"] = 50
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "gemini-2.5-flash-preview"
if "current_pdf_name" not in st.session_state:
    st.session_state["current_pdf_name"] = ""
if "use_custom_file" not in st.session_state:
    st.session_state["use_custom_file"] = False

# 사용 가능한 Gemini 모델 리스트
AVAILABLE_MODELS = [
    "gemini-2.5-flash-preview",
    "gemini-3-flash-preview",
    "gemini-1.5-flash"
]

# 기본 PDF 자동 로드 (교직실무.pdf)
DEFAULT_PDF_PATH = "교직실무.pdf"
if "pdf_text" not in st.session_state and os.path.exists(DEFAULT_PDF_PATH):
    try:
        with open(DEFAULT_PDF_PATH, "rb") as f:
            pdf_bytes = f.read()
        default_text = extract_text_from_pdf(pdf_bytes)
        if default_text and len(default_text.strip()) > 100:
            st.session_state["pdf_text"] = default_text
            st.session_state["current_pdf_name"] = "교직실무.pdf (기본)"
    except Exception:
        pass  # 로드 실패 시 무시하고 계속 진행

col_left, col_right = st.columns([2, 1])

with col_left:
    # 현재 로드된 파일 표시
    if st.session_state.get("current_pdf_name"):
        st.success(f"📄 **현재 로드된 파일:** {st.session_state['current_pdf_name']}")
        col_file1, col_file2 = st.columns([3, 1])
        with col_file1:
            use_custom = st.checkbox(
                "다른 파일 사용하기",
                value=st.session_state["use_custom_file"],
                key="use_custom_file_checkbox",
            )
            st.session_state["use_custom_file"] = use_custom
        with col_file2:
            if st.button("🗑️ 파일 삭제", help="현재 로드된 파일을 삭제합니다"):
                st.session_state["pdf_text"] = ""
                st.session_state["current_pdf_name"] = ""
                st.session_state["use_custom_file"] = True
                reset_state()
                st.rerun()
    else:
        st.info("📄 PDF 파일을 업로드하세요.")
        st.session_state["use_custom_file"] = True
    
    # 다른 파일 사용 시에만 업로더 및 OCR 옵션 표시
    if st.session_state["use_custom_file"]:
        st.divider()
        use_ocr = st.checkbox(
            "OCR 사용(스캔 PDF)",
            value=False,
            help="스캔본 PDF는 OCR이 필요합니다. (Tesseract 필요)",
        )
        
        if use_ocr:
            st.info("💡 **OCR 품질 향상 팁**\n"
                    "- DPI를 높이면(300) 더 정확합니다\n"
                    "- 임계값을 조정해 글자가 선명해지도록 설정\n"
                    "- 노이즈 제거는 흐릿한 문서에 효과적")
        
        ocr_dpi = st.slider("OCR 해상도(DPI)", min_value=150, max_value=350, value=300, step=25)
        ocr_threshold = st.slider(
            "OCR 전처리 임계값(높을수록 더 흰 배경)",
            min_value=0,
            max_value=255,
            value=200,
            step=5,
        )
        use_denoise = st.checkbox("노이즈 제거(미디언 필터)", value=True)
        denoise_size = st.slider(
            "노이즈 제거 강도(필터 크기)",
            min_value=1,
            max_value=7,
            value=3,
            step=2,
        )
        use_auto_rotate = st.checkbox("페이지 자동 회전", value=True)
        use_deskew = st.checkbox("기울기 보정", value=True)
        uploaded = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
        if uploaded:
            pdf_bytes = uploaded.read()
            base_text = extract_text_from_pdf(pdf_bytes)
            text = base_text
            if use_ocr:
                try:
                    ocr_text = ocr_text_from_pdf(
                        pdf_bytes,
                        ocr_dpi,
                        ocr_threshold,
                        denoise_size if use_denoise else 1,
                        use_auto_rotate,
                        use_deskew,
                    )
                    if normalize_text(ocr_text):
                        text = ocr_text
                    else:
                        st.warning("OCR 결과가 비어 있어 일반 텍스트 추출 결과를 사용합니다.")
                except Exception as exc:
                    st.error(f"OCR 처리에 실패했습니다. 일반 텍스트 추출 결과를 사용합니다. ({exc})")
            st.session_state["pdf_text"] = text
            st.session_state["current_pdf_name"] = uploaded.name
            st.session_state["use_custom_file"] = False
            st.success("✅ PDF 업로드 완료!")
            st.rerun()

with col_right:
    qtype = st.selectbox("문항 유형", ["객관식", "주관식", "혼합"], key="qtype")
    if qtype == "혼합":
        mixed_choice_ratio = st.slider(
            "객관식 비율(%)",
            min_value=0,
            max_value=100,
            value=st.session_state["mixed_choice_ratio"],
            step=5,
            key="mixed_choice_ratio",
        )
    else:
        mixed_choice_ratio = st.session_state["mixed_choice_ratio"]
    qcount = st.slider("문항 개수", min_value=10, max_value=50, value=10, step=1, key="qcount")
    difficulty = st.selectbox("난이도", ["쉬움", "보통", "어려움"], key="difficulty")
    
    # 객관식 보기는 항상 4개로 고정
    num_choices = 4
    distractor_mode = "혼동(유사 문자)"
    use_gemini = st.checkbox(
        "Gemini로 문항 생성",
        value=True,
        help="GEMINI_API_KEY가 secrets 또는 환경변수에 필요",
    )
    
    if use_gemini:
        gemini_model = st.selectbox(
            "Gemini 모델 선택",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state["gemini_model"]) if st.session_state["gemini_model"] in AVAILABLE_MODELS else 0,
            key="gemini_model_select"
        )
        st.session_state["gemini_model"] = gemini_model
    else:
        gemini_model = st.session_state["gemini_model"]
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("문항 생성"):
            text = st.session_state.get("pdf_text", "")
            
            # PDF 로드 여부 확인
            if not text:
                st.error("❌ PDF 파일이 로드되지 않았습니다. 먼저 PDF를 업로드하거나 기본 파일을 로드해주세요.")
                st.stop()
            
            # 텍스트 품질 검증
            is_quality_ok, quality_msg = check_text_quality(text)
            if not is_quality_ok:
                st.error(f"❌ {quality_msg}")
                st.info("💡 권장 사항: OCR 설정(해상도, 임계값)을 조정하거나 더 선명한 PDF를 사용해주세요.")
                st.stop()
            
            sentences = split_sentences(text)
            if use_gemini:
                try:
                    with st.spinner("🤖 Gemini로 고품질 문항을 생성 중..."):
                        st.session_state["questions"] = generate_questions_with_gemini(
                            text,
                            qtype,
                            qcount,
                            difficulty,
                            num_choices,
                            gemini_model,
                            mixed_choice_ratio,
                        )
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "quota" in exc_str.lower():
                        st.error("🚨 **Gemini 할당량 초과!**")
                        st.warning("현재 선택한 모델의 무료 사용량이 소진되었습니다. 다른 모델(예: 2.5-flash)을 선택하거나, 약 1분 후에 다시 시도해 주세요.")
                    else:
                        st.error(f"Gemini 생성 실패: {exc}")
                    
                    st.info("로컬 생성으로 대체합니다.")
                    st.session_state["questions"] = generate_questions(
                        sentences,
                        qcount,
                        qtype,
                        difficulty,
                        num_choices,
                        distractor_mode,
                        mixed_choice_ratio,
                    )
            else:
                st.session_state["questions"] = generate_questions(
                    sentences,
                    qcount,
                    qtype,
                    difficulty,
                    num_choices,
                    distractor_mode,
                    mixed_choice_ratio,
                )
            st.session_state["answers"] = {}
            st.session_state["graded"] = False
            st.session_state["wrong_questions"] = []
            st.session_state["review_mode"] = False
            st.session_state["explanations"] = {}
            st.session_state["explanations_ready"] = False
    with col_b:
        if st.button("문항 재제작"):
            text = st.session_state.get("pdf_text", "")
            
            # PDF 로드 여부 확인
            if not text:
                st.error("❌ PDF 파일이 로드되지 않았습니다. 먼저 PDF를 업로드하거나 기본 파일을 로드해주세요.")
                st.stop()
            
            # 텍스트 품질 검증
            is_quality_ok, quality_msg = check_text_quality(text)
            if not is_quality_ok:
                st.error(f"❌ {quality_msg}")
                st.info("💡 권장 사항: OCR 설정(해상도, 임계값)을 조정하거나 더 선명한 PDF를 사용해주세요.")
                st.stop()
            
            sentences = split_sentences(text)
            if use_gemini:
                try:
                    with st.spinner("🤖 Gemini로 고품질 문항을 다시 생성 중..."):
                        st.session_state["questions"] = generate_questions_with_gemini(
                            text,
                            qtype,
                            qcount,
                            difficulty,
                            num_choices,
                            gemini_model,
                            mixed_choice_ratio,
                        )
                except Exception as exc:
                    st.error(f"Gemini 생성 실패: {exc}")
                    st.session_state["questions"] = generate_questions(
                        sentences,
                        qcount,
                        qtype,
                        difficulty,
                        num_choices,
                        distractor_mode,
                        mixed_choice_ratio,
                    )
            else:
                st.session_state["questions"] = generate_questions(
                    sentences,
                    qcount,
                    qtype,
                    difficulty,
                    num_choices,
                    distractor_mode,
                    mixed_choice_ratio,
                )
            st.session_state["answers"] = {}
            st.session_state["graded"] = False
            st.session_state["wrong_questions"] = []
            st.session_state["review_mode"] = False
            st.session_state["explanations"] = {}
            st.session_state["explanations_ready"] = False
    with col_c:
        if st.button("초기화"):
            reset_state()
            st.rerun()

st.divider()

if st.session_state["questions"]:
    st.success(f"✅ {len(st.session_state['questions'])}개의 문항이 생성되었습니다!")
    
    # 문항 미리보기 (간단히)
    with st.expander("📋 생성된 문항 미리보기"):
        for i, q in enumerate(st.session_state["questions"][:5], 1):
            st.write(f"**{i}.** {q.prompt}")
        if len(st.session_state["questions"]) > 5:
            st.caption(f"... 외 {len(st.session_state['questions']) - 5}개 문항")
    
    st.write("")
    
    # 퀴즈 시작 버튼
    col_quiz, col_export = st.columns([1, 1])
    with col_quiz:
        if st.button("🎯 카드형 퀴즈 시작하기", type="primary", use_container_width=True):
            # 퀴즈 상태 초기화
            st.session_state["current_question_idx"] = 0
            st.session_state["quiz_answers"] = {}
            st.session_state["quiz_submitted"] = False
            st.switch_page("pages/quiz.py")
    
    with col_export:
        st.write("")  # 여백
else:
    st.info("📄 PDF를 업로드하고 문항을 생성하세요.")

if st.session_state["questions"]:
    st.divider()
    st.subheader("📥 문항 내보내기")
    export_col_a, export_col_b = st.columns(2)
    with export_col_a:
        st.download_button(
            "📄 문항 CSV 다운로드",
            data=export_questions_csv(st.session_state["questions"], st.session_state["answers"]),
            file_name="questions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_col_b:
        st.download_button(
            "📄 문항 JSON 다운로드",
            data=export_questions_json(st.session_state["questions"], st.session_state["answers"]),
            file_name="questions.json",
            mime="application/json",
            use_container_width=True,
        )

