import streamlit as st
import re
import io
import os
import zipfile
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageDraw

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


def load_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #21262d;
  --border: #30363d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --text-muted: #6e7681;
  --accent-cyan: #58a6ff;
  --accent-green: #3fb950;
  --accent-yellow: #d29922;
  --accent-red: #f85149;
  --accent-purple: #a371f7;
}

* { font-family: 'Inter', -apple-system, sans-serif; }
code, .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; }

.stApp { background: var(--bg-primary); }

.block-container {
  padding-top: 2rem !important;
  max-width: 1000px !important;
}

#MainMenu, footer, header { visibility: hidden; }

.header {
  border-bottom: 1px solid var(--border);
  padding: 0 0 1.25rem 0;
  margin-bottom: 1.5rem;
}

.header-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.header-title::before {
  content: "λ";
  color: var(--accent-cyan);
}

.header-desc {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin: 0.4rem 0 0 1.1rem;
  font-family: 'JetBrains Mono', monospace;
}

.section {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  margin: 1rem 0;
}

.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--accent-cyan);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 0 0 0.5rem 0;
}

.section-hint {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
}

.tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 500;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  color: var(--text-secondary);
}

.tag-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-muted);
}

.tag--success { border-color: var(--accent-green); color: var(--accent-green); }
.tag--success .tag-dot { background: var(--accent-green); }
.tag--warning { border-color: var(--accent-yellow); color: var(--accent-yellow); }
.tag--warning .tag-dot { background: var(--accent-yellow); }
.tag--error { border-color: var(--accent-red); color: var(--accent-red); }
.tag--error .tag-dot { background: var(--accent-red); }
.tag--info { border-color: var(--accent-cyan); color: var(--accent-cyan); }
.tag--info .tag-dot { background: var(--accent-cyan); }

.result-row {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  margin: 8px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  transition: border-color 0.15s ease;
}

.result-row:hover { border-color: var(--accent-cyan); }

.result-label { color: var(--text-muted); font-size: 0.7rem; }
.result-value { color: var(--text-primary); }
.result-arrow { color: var(--accent-cyan); margin: 0 8px; }

.stSlider label, .stSelectbox label, .stCheckbox label {
  color: var(--text-secondary) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  text-transform: lowercase !important;
}

div[data-testid="stFileUploader"] {
  background: var(--bg-tertiary) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 6px !important;
}

div[data-testid="stFileUploader"]:hover {
  border-color: var(--accent-cyan) !important;
}

.stButton > button {
  background: transparent !important;
  border: 1px solid var(--accent-cyan) !important;
  border-radius: 6px !important;
  color: var(--accent-cyan) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  padding: 10px 24px !important;
  text-transform: lowercase;
  letter-spacing: 0.05em;
  width: auto !important;
  transition: all 0.15s ease;
}

.stButton > button:hover {
  background: var(--accent-cyan) !important;
  color: var(--bg-primary) !important;
}

.stDownloadButton > button {
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  text-transform: lowercase;
}

.stDownloadButton > button:hover {
  border-color: var(--accent-green) !important;
  color: var(--accent-green) !important;
}

details {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}

details summary {
  color: var(--text-secondary) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.8rem !important;
}

.stMarkdown, p, span, div { color: var(--text-primary); }
.stCaption { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace !important; }

.stSuccess {
  background: rgba(63, 185, 80, 0.1) !important;
  border: 1px solid var(--accent-green) !important;
  border-radius: 6px !important;
}

.stWarning {
  background: rgba(210, 153, 34, 0.1) !important;
  border: 1px solid var(--accent-yellow) !important;
  border-radius: 6px !important;
}

.upload-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
  text-transform: lowercase;
}

.preview-section {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px;
  margin: 8px 0;
}

.no-changes {
  color: var(--accent-green);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  padding: 1rem;
  text-align: center;
  border: 1px dashed var(--accent-green);
  border-radius: 6px;
  background: rgba(63, 185, 80, 0.05);
}
</style>
""", unsafe_allow_html=True)


def similarity(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return 0.0
    if fuzz is not None:
        return fuzz.token_set_ratio(a, b) / 100.0
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lower()
    text = text.replace("×", "x").replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\bx\s*10\s*\^\s*3\s*/\s*u\s*l\b", "x10^3/ul", text)
    text = re.sub(r"\bx10\s*\^\s*3\s*/\s*u\s*l\b", "x10^3/ul", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def clean_label(label: str) -> str:
    label = label.lower()
    label = re.sub(r"[^a-z0-9\s]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def parse_number(s: str) -> Optional[str]:
    t = s.strip().lower()
    t = t.replace("o", "0").replace("l", "1").replace(",", ".")
    if re.fullmatch(r"\d+(\.\d+)?\s*-\s*\d+(\.\d+)?", t):
        return None
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return t
    return None


class Extractor:
    @staticmethod
    def pdf_native_text(path: str) -> str:
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(path)
            parts = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if t.strip():
                    parts.append(t)
            return "\n".join(parts).strip()
        except Exception:
            return ""

    @staticmethod
    def looks_scanned(text: str, min_chars: int = 80) -> bool:
        return not text or len(text.strip()) < min_chars

    @staticmethod
    def preprocess_image(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        img = ImageOps.grayscale(img)
        return img

    @staticmethod
    def ocr_image_text(img: Image.Image, psm: int = 6) -> str:
        img = Extractor.preprocess_image(img)
        return pytesseract.image_to_string(img, config=f"--psm {psm}")

    @staticmethod
    def extract_pdf_text(path: str, dpi: int = 250, psm: int = 6) -> Tuple[str, bool]:
        native = Extractor.pdf_native_text(path)
        if not Extractor.looks_scanned(native):
            return native, False
        imgs = convert_from_path(path, dpi=dpi)
        out = []
        for im in imgs:
            out.append(Extractor.ocr_image_text(im, psm=psm))
        return "\n".join(out), True

    @staticmethod
    def extract_docx_text(path: str) -> str:
        import docx
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])


@dataclass
class OCRWord:
    page: int
    text: str
    norm: str
    conf: int
    bbox: Tuple[int, int, int, int]


def ocr_words_from_image(img: Image.Image, page: int, psm: int = 6) -> List[OCRWord]:
    im = Extractor.preprocess_image(img)
    data = pytesseract.image_to_data(im, output_type=Output.DICT, config=f"--psm {psm}")
    words: List[OCRWord] = []
    n = len(data.get("text", []))
    for i in range(n):
        w = (data["text"][i] or "").strip()
        if not w:
            continue
        conf = int(data.get("conf", ["-1"])[i])
        x, y, ww, hh = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        x1, y1, x2, y2 = x, y, x + ww, y + hh
        norm = w.lower().strip()
        norm = norm.replace("×", "x").replace("µ", "u").replace("μ", "u")
        norm = norm.replace("o", "0").replace("l", "1").replace(",", ".")
        norm = re.sub(r"[^a-z0-9\.\^/]+", "", norm)
        words.append(OCRWord(page=page, text=w, norm=norm, conf=conf, bbox=(x1, y1, x2, y2)))
    return words


def draw_boxes(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        d.rectangle([x1, y1, x2, y2], outline=(248, 81, 73), width=4)
    return out


def zip_images(pages: List[Tuple[int, bytes]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pno, b in pages:
            zf.writestr(f"highlighted_page_{pno:03d}.png", b)
    mem.seek(0)
    return mem.getvalue()


@dataclass
class Pair:
    label: str
    value: str
    raw_label: str
    raw_value: str
    context: str


def extract_label_value_pairs(text: str) -> List[Pair]:
    t = normalize_text(text)
    pairs: List[Pair] = []
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    
    for ln in lines:
        if len(ln) > 200:
            continue
        m = re.search(r"^([a-z][a-z0-9 \-/]{2,70}?)[\s:]+(\d+(?:[.,]\d+)?)\b", ln)
        if m:
            raw_lab, raw_val = m.group(1).strip(), m.group(2).strip()
            val = parse_number(raw_val)
            if val:
                lab = clean_label(raw_lab)
                if 3 <= len(lab) <= 80:
                    pairs.append(Pair(label=lab, value=val, raw_label=raw_lab, raw_value=raw_val, context=ln[:160]))
            continue
        m = re.search(r"\b([a-z][a-z0-9 \-/]{2,70}?)\s+(\d+(?:[.,]\d+)?)\b", ln)
        if m:
            raw_lab, raw_val = m.group(1).strip(), m.group(2).strip()
            val = parse_number(raw_val)
            if val:
                lab = clean_label(raw_lab)
                if 3 <= len(lab) <= 80 and not lab.startswith(("method", "unit", "reference", "interval")):
                    pairs.append(Pair(label=lab, value=val, raw_label=raw_lab, raw_value=raw_val, context=ln[:160]))

    abs_matches = re.finditer(
        r"\b(absolute\s+[a-z][a-z0-9 ]{2,40}\s+count)\b[\s\S]{0,120}?\b(\d+(?:[.,]\d+)?)\b",
        t, flags=re.IGNORECASE
    )
    for m in abs_matches:
        raw_lab, raw_val = m.group(1), m.group(2)
        val = parse_number(raw_val)
        if not val:
            continue
        lab = clean_label(raw_lab)
        snippet_start = max(0, m.start() - 40)
        snippet_end = min(len(t), m.end() + 40)
        context = t[snippet_start:snippet_end].replace("\n", " ")[:160]
        pairs.append(Pair(label=lab, value=val, raw_label=raw_lab, raw_value=raw_val, context=context))

    seen = set()
    uniq: List[Pair] = []
    for p in pairs:
        key = (p.label, p.value)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


@dataclass
class PairMatch:
    gt_label: str
    gt_value: str
    user_label: str
    user_value: str
    label_score: float
    value_changed: bool
    name_changed: bool


def match_pairs(gt_pairs: List[Pair], user_pairs: List[Pair], min_label_sim: float = 0.82) -> Tuple[List[PairMatch], List[Pair], List[Pair]]:
    used_user = set()
    matches: List[PairMatch] = []
    missing: List[Pair] = []

    for g in gt_pairs:
        best_u, best_s, best_ui = None, 0.0, -1
        for ui, u in enumerate(user_pairs):
            if ui in used_user:
                continue
            s = similarity(g.label, u.label)
            if s > best_s:
                best_s, best_u, best_ui = s, u, ui

        if best_u is None or best_s < min_label_sim:
            missing.append(g)
            continue

        used_user.add(best_ui)
        matches.append(PairMatch(
            gt_label=g.label, gt_value=g.value,
            user_label=best_u.label, user_value=best_u.value,
            label_score=best_s,
            value_changed=(g.value != best_u.value),
            name_changed=(g.label != best_u.label)
        ))

    extras = [u for i, u in enumerate(user_pairs) if i not in used_user]
    matches.sort(key=lambda m: (m.value_changed or m.name_changed, m.label_score), reverse=True)
    return matches, missing, extras


def highlight_tokens_on_user_pages(user_pages: List[Image.Image], user_words: List[OCRWord], tokens: List[str]) -> Tuple[Optional[bytes], List[bytes]]:
    if not user_pages or not user_words or not tokens:
        return None, []

    targets = []
    for t in tokens:
        nt = t.lower().strip().replace("×", "x").replace("µ", "u").replace("μ", "u")
        nt = nt.replace("o", "0").replace("l", "1").replace(",", ".")
        nt = re.sub(r"[^a-z0-9\.\^/]+", "", nt)
        if nt:
            targets.append(nt)

    by_page: Dict[int, List[Tuple[int, int, int, int]]] = {i + 1: [] for i in range(len(user_pages))}
    for w in user_words:
        if w.norm in targets:
            by_page[w.page].append(w.bbox)

    pages_bytes: List[Tuple[int, bytes]] = []
    previews: List[bytes] = []
    for pno, img in enumerate(user_pages, start=1):
        boxes = by_page.get(pno, [])
        if not boxes:
            continue
        annotated = draw_boxes(img, boxes)
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        b = buf.getvalue()
        pages_bytes.append((pno, b))
        if len(previews) < 3:
            previews.append(b)

    if not pages_bytes:
        return None, []
    return zip_images(pages_bytes), previews


def main():
    st.set_page_config(page_title="ocr-compare", layout="wide", initial_sidebar_state="collapsed")
    load_custom_css()

    # Header
    st.markdown("""
<div class="header">
  <div class="header-title">ocr-compare</div>
  <p class="header-desc"></p>
</div>
""", unsafe_allow_html=True)

    # Config section
    st.markdown("""<div class="section"><div class="section-label">config</div></div>""", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        min_label_sim = st.slider("min_label_similarity", 0.60, 0.95, 0.78)
    with c2:
        psm = st.selectbox("tesseract_psm", [6, 4, 3, 11], index=1)
    with c3:
        show_all_matches = st.checkbox("show_all_matches", value=False)

    # Input section
    st.markdown("""<div class="section"><div class="section-label">input</div></div>""", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown('<p class="upload-label">ground_truth</p>', unsafe_allow_html=True)
        gt = st.file_uploader("gt", type=["pdf", "docx", "png", "jpg", "jpeg", "tiff"], key="gt", label_visibility="collapsed")
    with colB:
        st.markdown('<p class="upload-label">user_document</p>', unsafe_allow_html=True)
        ud = st.file_uploader("ud", type=["pdf", "docx", "png", "jpg", "jpeg", "tiff"], key="ud", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("run"):
        if not gt or not ud:
            st.warning("upload both documents")
            return

        with st.spinner("extracting..."):
            with tempfile.TemporaryDirectory() as td:
                gt_path = os.path.join(td, gt.name)
                ud_path = os.path.join(td, ud.name)
                with open(gt_path, "wb") as f:
                    f.write(gt.read())
                with open(ud_path, "wb") as f:
                    f.write(ud.read())

                gt_used_ocr = False
                if gt.name.lower().endswith(".pdf"):
                    gt_text, gt_used_ocr = Extractor.extract_pdf_text(gt_path, dpi=250, psm=psm)
                elif gt.name.lower().endswith(".docx"):
                    gt_text = Extractor.extract_docx_text(gt_path)
                else:
                    gt_text = Extractor.ocr_image_text(Image.open(gt_path), psm=psm)
                    gt_used_ocr = True

                user_used_ocr = False
                user_text = ""
                user_pages: List[Image.Image] = []
                user_words: List[OCRWord] = []

                if ud.name.lower().endswith(".pdf"):
                    native = Extractor.pdf_native_text(ud_path) if PdfReader else ""
                    if native and not Extractor.looks_scanned(native):
                        user_text = native
                    else:
                        user_used_ocr = True
                        user_pages = convert_from_path(ud_path, dpi=250)
                        out = []
                        for pi, img in enumerate(user_pages, start=1):
                            out.append(Extractor.ocr_image_text(img, psm=psm))
                            user_words.extend(ocr_words_from_image(img, page=pi, psm=psm))
                        user_text = "\n".join(out)
                elif ud.name.lower().endswith(".docx"):
                    user_text = Extractor.extract_docx_text(ud_path)
                else:
                    user_used_ocr = True
                    img = Image.open(ud_path)
                    user_pages = [img]
                    user_text = Extractor.ocr_image_text(img, psm=psm)
                    user_words = ocr_words_from_image(img, page=1, psm=psm)

                gt_pairs = extract_label_value_pairs(gt_text)
                user_pairs = extract_label_value_pairs(user_text)
                matches, missing, extras = match_pairs(gt_pairs, user_pairs, min_label_sim=min_label_sim)
                changes = [m for m in matches if (m.value_changed or m.name_changed)]
                shown = matches if show_all_matches else changes

                highlight_tokens = []
                for m in changes:
                    highlight_tokens.append(m.user_value)
                    label_tokens = m.user_label.split()
                    if label_tokens:
                        highlight_tokens.append(label_tokens[-2] if len(label_tokens) >= 2 else label_tokens[-1])

                highlight_zip, previews = (None, [])
                if user_used_ocr:
                    highlight_zip, previews = highlight_tokens_on_user_pages(user_pages, user_words, highlight_tokens)

                st.session_state.res = {
                    "gt_used_ocr": gt_used_ocr, "user_used_ocr": user_used_ocr,
                    "gt_pairs": gt_pairs, "user_pairs": user_pairs,
                    "matches": matches, "changes": changes, "shown": shown,
                    "missing": missing, "extras": extras,
                    "highlight_zip": highlight_zip, "previews": previews,
                    "gt_text": gt_text, "user_text": user_text,
                    "engine": "rapidfuzz" if fuzz else "difflib",
                }

        st.success("done")

    if "res" not in st.session_state:
        return

    res = st.session_state.res
    changes = res["changes"]
    shown = res["shown"]

    # Output section
    st.markdown(f"""
<div class="section">
  <div class="section-label">output</div>
  <div style="display:flex; gap:8px; flex-wrap:wrap; margin:12px 0;">
    <span class="tag {'tag--warning' if res['gt_used_ocr'] else 'tag--success'}">
      <span class="tag-dot"></span>gt_ocr: {str(res['gt_used_ocr']).lower()}
    </span>
    <span class="tag {'tag--warning' if res['user_used_ocr'] else 'tag--success'}">
      <span class="tag-dot"></span>user_ocr: {str(res['user_used_ocr']).lower()}
    </span>
    <span class="tag tag--info">
      <span class="tag-dot"></span>matcher: {res['engine']}
    </span>
    <span class="tag tag--error">
      <span class="tag-dot"></span>changes: {len(changes)}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Downloads
    st.markdown("""<div class="section"><div class="section-label">export</div></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if res["highlight_zip"]:
            st.download_button("download_highlights.zip", data=res["highlight_zip"], file_name="highlighted_pages.zip", mime="application/zip")
        else:
            st.caption("highlights available only for ocr'd documents")
    with c2:
        st.download_button("download_extracted.txt",
            data=("# gt\n" + res["gt_text"] + "\n\n# user\n" + res["user_text"]).encode("utf-8"),
            file_name="extracted_texts.txt", mime="text/plain")

    # Preview
    if res["previews"]:
        st.markdown("""<div class="section"><div class="section-label">preview</div></div>""", unsafe_allow_html=True)
        for b in res["previews"]:
            st.markdown('<div class="preview-section">', unsafe_allow_html=True)
            st.image(b, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Results
    st.markdown("""<div class="section"><div class="section-label">diff</div></div>""", unsafe_allow_html=True)

    if not shown:
        st.markdown('<div class="no-changes">✓ no changes detected</div>', unsafe_allow_html=True)
    else:
        topn = st.slider("limit", 5, 300, min(60, max(5, len(shown))))
        for m in shown[:topn]:
            if m.value_changed:
                status_class, status_text = "tag--error", "value_changed"
            elif m.name_changed:
                status_class, status_text = "tag--warning", "name_changed"
            else:
                status_class, status_text = "tag--success", "match"

            val_color = "var(--accent-red)" if m.value_changed else "var(--accent-green)"

            st.markdown(f"""
<div class="result-row">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
    <span class="tag {status_class}"><span class="tag-dot"></span>{status_text}</span>
    <span style="color:var(--text-muted); font-size:0.65rem;">score: {m.label_score:.3f}</span>
  </div>
  <div style="margin:4px 0;">
    <span class="result-label">gt:</span>
    <span class="result-value">{m.gt_label}</span>
    <span class="result-arrow">→</span>
    <span style="color:var(--accent-green);">{m.gt_value}</span>
  </div>
  <div>
    <span class="result-label">user:</span>
    <span class="result-value">{m.user_label}</span>
    <span class="result-arrow">→</span>
    <span style="color:{val_color};">{m.user_value}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Debug
    with st.expander("debug: extracted pairs"):
        st.write({"gt_pairs": len(res["gt_pairs"]), "user_pairs": len(res["user_pairs"])})
        st.markdown("**gt sample**")
        st.write([{"label": p.label, "value": p.value, "context": p.context} for p in res["gt_pairs"][:40]])
        st.markdown("**user sample**")
        st.write([{"label": p.label, "value": p.value, "context": p.context} for p in res["user_pairs"][:40]])


if __name__ == "__main__":
    main()