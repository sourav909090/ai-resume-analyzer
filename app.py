import streamlit as st
import fitz
import pytesseract
import re
import os
import shutil
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# DARK UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0b0f19;
    color: #f5f7ff;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(135deg, #111827, #1e1b4b);
    border: 1px solid #3730a3;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.section-title {
    color: #ffffff;
    font-size: 24px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 14px;
}

.score-card {
    background: linear-gradient(135deg, #111827, #17133b);
    border: 1px solid #6366f1;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin: 15px 0 25px 0;
}

.score-label {
    color: #cbd5e1;
    font-size: 17px;
}

.score-value {
    color: #818cf8;
    font-size: 46px;
    font-weight: 800;
    margin-top: 5px;
}

.info-card {
    background: #111827;
    border: 1px solid #273244;
    border-radius: 15px;
    padding: 20px;
}

.skill-found {
    display: inline-block;
    background: #052e1b;
    color: #4ade80;
    border: 1px solid #166534;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
}

.skill-missing {
    display: inline-block;
    background: #3b1111;
    color: #f87171;
    border: 1px solid #991b1b;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
}

.skill-required {
    display: inline-block;
    background: #172554;
    color: #60a5fa;
    border: 1px solid #1d4ed8;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>📄 AI-Powered Resume Analyzer</h1>
    <p>
        Compare your resume with a job description,
        measure compatibility and identify important skill gaps.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Resume Analyzer")

    st.write("Upload a resume and paste a target job description.")

    st.divider()

    st.write("### Features")

    st.write("📊 Resume Match Score")
    st.write("🎯 Skill Gap Analysis")
    st.write("🔍 Keyword Coverage")
    st.write("📈 Resume Statistics")
    st.write("📑 Section Detection")
    st.write("💡 Improvement Suggestions")
    st.write("🖨️ OCR Support")
    st.write("📥 Download Report")


# =========================================================
# INPUT
# =========================================================

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "📄 Upload Resume PDF",
        type=["pdf"]
    )

with col2:

    job_description = st.text_area(
        "💼 Job Description",
        height=220,
        placeholder="Paste the job description here..."
    )


# =========================================================
# SKILLS
# =========================================================

skills = [
    "python",
    "java",
    "javascript",
    "sql",
    "html",
    "css",
    "react",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "ai agents",
    "automation",
    "nlp",
    "pandas",
    "numpy",
    "scikit-learn",
    "streamlit",
    "api",
    "rest api",
    "json",
    "git",
    "github",
    "oop",
    "object oriented programming",
    "llm",
    "tensorflow",
    "pytorch",
    "excel",
    "power bi",
    "data analysis",
    "data science",
    "aws",
    "azure",
    "docker"
]


# =========================================================
# SKILL CHECK
# =========================================================

def contains_skill(text, skill):

    text = text.lower()
    skill = skill.lower()

    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

    return re.search(pattern, text) is not None


# =========================================================
# PDF EXTRACTION + OCR
# =========================================================

def extract_resume_text(pdf_bytes):

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in pdf:

        page_text = page.get_text()

        if page_text.strip():

            text += page_text + "\n"

        else:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            ocr_text = pytesseract.image_to_string(img)

            text += ocr_text + "\n"

    pdf.close()

    return text


# =========================================================
# TESSERACT
# =========================================================

tesseract_path = shutil.which("tesseract")

windows_tesseract = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if tesseract_path:

    pytesseract.pytesseract.tesseract_cmd = tesseract_path

elif os.path.exists(windows_tesseract):

    pytesseract.pytesseract.tesseract_cmd = windows_tesseract


# =========================================================
# ANALYZE
# =========================================================

if st.button(
    "🔍 Analyze Resume",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning("Please upload a resume PDF.")
        st.stop()

    if not job_description.strip():

        st.warning("Please paste a job description.")
        st.stop()


    # =====================================================
    # EXTRACT
    # =====================================================

    try:

        resume_text = extract_resume_text(
            uploaded_file.getvalue()
        )

    except Exception as e:

        st.error(
            f"Could not read the PDF: {e}"
        )

        st.stop()


    if not resume_text.strip():

        st.error(
            "Could not extract text from this PDF."
        )

        st.stop()


    # =====================================================
    # CLEAN TEXT
    # =====================================================

    resume_clean = re.sub(
        r"\s+",
        " ",
        resume_text.lower()
    ).strip()

    jd_clean = re.sub(
        r"\s+",
        " ",
        job_description.lower()
    ).strip()


    # =====================================================
    # TF-IDF SIMILARITY
    # =====================================================

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        vectors = vectorizer.fit_transform(
            [resume_clean, jd_clean]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        text_score = similarity * 100

    except Exception:

        text_score = 0


    # =====================================================
    # SKILLS
    # =====================================================

    resume_skills = [
        skill for skill in skills
        if contains_skill(resume_text, skill)
    ]

    required_skills = [
        skill for skill in skills
        if contains_skill(job_description, skill)
    ]

    missing_skills = [
        skill for skill in required_skills
        if skill not in resume_skills
    ]


    # =====================================================
    # SKILL SCORE
    # =====================================================

    if required_skills:

        matched_skills = [
            skill for skill in required_skills
            if skill in resume_skills
        ]

        skill_score = (
            len(matched_skills)
            / len(required_skills)
        ) * 100

    else:

        skill_score = 0


    # =====================================================
    # FINAL MATCH SCORE
    # =====================================================

    if required_skills:

        final_score = (
            (text_score * 0.60)
            + (skill_score * 0.40)
        )

    else:

        final_score = text_score


    final_score = round(
        min(max(final_score, 0), 100),
        2
    )


    # =====================================================
    # KEYWORD COVERAGE
    # =====================================================

    if required_skills:

        keyword_coverage = round(
            skill_score,
            2
        )

    else:

        keyword_coverage = 0


    # =====================================================
    # STATISTICS
    # =====================================================

    word_count = len(
        resume_text.split()
    )

    character_count = len(
        resume_text
    )


    # =====================================================
    # SECTIONS
    # =====================================================

    possible_sections = [
        "education",
        "experience",
        "skills",
        "projects",
        "certifications",
        "summary",
        "objective",
        "achievements",
        "contact"
    ]

    found_sections = [
        section.title()
        for section in possible_sections
        if section in resume_clean
    ]


    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Analysis Results</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # SCORE
    # =====================================================

    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">
                Resume Match Score
            </div>

            <div class="score-value">
                {final_score}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.progress(
        min(final_score / 100, 1.0)
    )


    # =====================================================
    # METRICS
    # =====================================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "📊 Match",
            f"{final_score}%"
        )

    with m2:

        st.metric(
            "🔍 Keyword Coverage",
            f"{keyword_coverage}%"
        )

    with m3:

        st.metric(
            "✅ Skills Found",
            len(resume_skills)
        )

    with m4:

        st.metric(
            "⚠️ Missing Skills",
            len(missing_skills)
        )


    # =====================================================
    # SKILLS FOUND
    # =====================================================

    st.markdown(
        '<div class="section-title">✅ Skills Found</div>',
        unsafe_allow_html=True
    )

    if resume_skills:

        html = ""

        for skill in resume_skills:

            html += (
                f'<span class="skill-found">'
                f'{skill}'
                f'</span>'
            )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No matching skills detected."
        )


    # =====================================================
    # REQUIRED SKILLS
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Required Skills</div>',
        unsafe_allow_html=True
    )

    if required_skills:

        html = ""

        for skill in required_skills:

            html += (
                f'<span class="skill-required">'
                f'{skill}'
                f'</span>'
            )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No predefined skills detected."
        )


    # =====================================================
    # MISSING
    # =====================================================

    st.markdown(
        '<div class="section-title">⚠️ Missing Skills</div>',
        unsafe_allow_html=True
    )

    if missing_skills:

        html = ""

        for skill in missing_skills:

            html += (
                f'<span class="skill-missing">'
                f'{skill}'
                f'</span>'
            )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    else:

        st.success(
            "🎉 No missing predefined skills detected."
        )


    # =====================================================
    # STATISTICS
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Resume Statistics</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3 = st.columns(3)

    with s1:

        st.metric(
            "📝 Word Count",
            word_count
        )

    with s2:

        st.metric(
            "🔤 Characters",
            character_count
        )

    with s3:

        st.metric(
            "📑 Sections",
            len(found_sections)
        )


    # =====================================================
    # SECTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">📑 Resume Sections</div>',
        unsafe_allow_html=True
    )

    if found_sections:

        st.write(
            " • ".join(found_sections)
        )

    else:

        st.info(
            "No standard sections detected."
        )


    # =====================================================
    # SUGGESTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">💡 Improvement Suggestions</div>',
        unsafe_allow_html=True
    )

    suggestions = []


    if final_score < 40:

        suggestions.append(
            "Improve alignment between your resume and the target job description."
        )

    elif final_score < 70:

        suggestions.append(
            "Your resume has moderate alignment. Add relevant keywords and practical projects."
        )

    else:

        suggestions.append(
            "Your resume has strong alignment with the target job description."
        )


    if missing_skills:

        suggestions.append(
            "Consider learning or demonstrating: "
            + ", ".join(missing_skills[:8])
            + "."
        )


    if word_count < 250:

        suggestions.append(
            "Consider adding relevant projects, achievements or experience."
        )


    if "projects" not in resume_clean:

        suggestions.append(
            "Add a dedicated Projects section if you have relevant projects."
        )


    for suggestion in suggestions:

        st.info(
            "💡 " + suggestion
        )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.markdown(
        '<div class="section-title">🧠 Overall Recommendation</div>',
        unsafe_allow_html=True
    )

    if final_score >= 70:

        st.success(
            "Strong resume-job alignment based on the current analysis."
        )

    elif final_score >= 40:

        st.info(
            "Moderate resume-job alignment. "
            "Consider improving missing skills and relevant keywords."
        )

    else:

        st.warning(
            "Low resume-job alignment. "
            "Consider improving relevant skills, keywords and project descriptions."
        )


    # =====================================================
    # REPORT
    # =====================================================

    report = f"""
AI-POWERED RESUME ANALYZER
==========================

Resume Match Score: {final_score}%

Keyword Coverage: {keyword_coverage}%

Skills Found:
{", ".join(resume_skills)}

Required Skills:
{", ".join(required_skills)}

Missing Skills:
{", ".join(missing_skills)}

Resume Statistics:
Word Count: {word_count}
Characters: {character_count}
Sections: {len(found_sections)}

Resume Sections:
{", ".join(found_sections)}

Improvement Suggestions:
{chr(10).join("- " + x for x in suggestions)}
"""


    st.download_button(
        "📥 Download Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    📄 AI-Powered Resume Analyzer
    <br>
    Python • Streamlit • NLP • TF-IDF • OCR
</div>
""", unsafe_allow_html=True)
