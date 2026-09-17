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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #ffffff 55%, #f5f3ff 100%);
}

/* Main header */
.hero {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 28px 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.18);
}

.hero h1 {
    margin: 0;
    font-size: 38px;
}

.hero p {
    margin: 8px 0 0 0;
    font-size: 16px;
    opacity: 0.92;
}

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

/* Score */
.score-card {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    border: 2px solid #c7d2fe;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    margin: 15px 0 25px 0;
}

.score-label {
    color: #4b5563;
    font-size: 16px;
    font-weight: 600;
}

.score-value {
    color: #4f46e5;
    font-size: 42px;
    font-weight: 800;
    margin-top: 5px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 28px;
    margin-bottom: 14px;
    color: #111827;
}

/* Skill pills */
.skill-found {
    display: inline-block;
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
    font-weight: 600;
}

.skill-missing {
    display: inline-block;
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
    font-weight: 600;
}

.skill-required {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    padding: 25px;
    margin-top: 35px;
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
        Analyze your resume against a job description,
        identify skill gaps and understand your resume-job match.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Analyzer")

    st.write(
        "Upload your resume and enter the target job description."
    )

    st.divider()

    st.write("### 🚀 Features")

    st.write("📊 Resume Match Score")
    st.write("🎯 Skill Gap Analysis")
    st.write("🔍 Keyword Coverage")
    st.write("📈 Resume Statistics")
    st.write("📑 Section Detection")
    st.write("💡 Improvement Suggestions")
    st.write("🖨️ OCR for Scanned PDFs")
    st.write("📥 Download Report")


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "📄 Upload Resume PDF",
        type=["pdf"]
    )

with col2:

    job_description = st.text_area(
        "💼 Paste Job Description",
        height=220,
        placeholder="Paste the complete job description here..."
    )


# =========================================================
# SKILLS DATABASE
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
# SKILL DETECTION
# =========================================================

def contains_skill(text, skill):

    text = text.lower()
    skill = skill.lower()

    pattern = r"\b" + re.escape(skill) + r"\b"

    return re.search(pattern, text) is not None


# =========================================================
# PDF + OCR
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
# TESSERACT CONFIG
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

        st.warning("⚠️ Please upload a resume PDF.")

        st.stop()

    if not job_description.strip():

        st.warning("⚠️ Please paste a job description.")

        st.stop()


    # -----------------------------------------------------
    # Extract text
    # -----------------------------------------------------

    try:

        pdf_bytes = uploaded_file.getvalue()

        resume_text = extract_resume_text(pdf_bytes)

    except Exception as e:

        st.error(
            f"❌ Could not read the PDF: {e}"
        )

        st.stop()


    if not resume_text.strip():

        st.error(
            "❌ Could not extract any text from this PDF."
        )

        st.stop()


    # -----------------------------------------------------
    # TF-IDF similarity
    # -----------------------------------------------------

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )

    score = round(
        similarity[0][0] * 100,
        2
    )


    # -----------------------------------------------------
    # Skills
    # -----------------------------------------------------

    resume_skills = [
        skill
        for skill in skills
        if contains_skill(resume_text, skill)
    ]

    required_skills = [
        skill
        for skill in skills
        if contains_skill(job_description, skill)
    ]

    missing_skills = [
        skill
        for skill in required_skills
        if skill not in resume_skills
    ]


    # -----------------------------------------------------
    # Keyword Coverage
    # -----------------------------------------------------

    if required_skills:

        matched_count = len(
            [
                skill
                for skill in required_skills
                if skill in resume_skills
            ]
        )

        keyword_coverage = round(
            (matched_count / len(required_skills)) * 100,
            2
        )

    else:

        keyword_coverage = 0


    # -----------------------------------------------------
    # Resume Statistics
    # -----------------------------------------------------

    words = resume_text.split()

    word_count = len(words)

    character_count = len(resume_text)


    # -----------------------------------------------------
    # Section Detection
    # -----------------------------------------------------

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
        if section in resume_text.lower()
    ]


    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Analysis Results</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Main Score
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">
                Resume Match Score
            </div>

            <div class="score-value">
                {score}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Advanced Metrics
    # -----------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "📊 Match",
            f"{score}%"
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


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    st.write("### 🎯 Match Progress")

    st.progress(
        min(score / 100, 1.0)
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
            "No predefined skills detected in the job description."
        )


    # =====================================================
    # MISSING SKILLS
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
    # RESUME STATISTICS
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
    # RESUME SECTIONS
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
            "No standard resume sections detected."
        )


    # =====================================================
    # IMPROVEMENT SUGGESTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">💡 Improvement Suggestions</div>',
        unsafe_allow_html=True
    )

    suggestions = []


    if score < 40:

        suggestions.append(
            "Improve alignment between your resume and the target job description."
        )

    elif score < 70:

        suggestions.append(
            "Your resume has moderate similarity. Add more relevant keywords and project experience."
        )

    else:

        suggestions.append(
            "Your resume has strong textual similarity with the target job description."
        )


    if missing_skills:

        suggestions.append(
            "Relevant missing skills include: "
            + ", ".join(missing_skills[:8])
            + "."
        )


    if word_count < 250:

        suggestions.append(
            "Your resume appears short. Consider adding relevant projects, achievements or practical experience."
        )


    if "projects" not in resume_text.lower():

        suggestions.append(
            "Consider adding a dedicated Projects section."
        )


    if "experience" not in resume_text.lower():

        suggestions.append(
            "If applicable, add internships or practical experience."
        )


    for suggestion in suggestions:

        st.info(
            "💡 " + suggestion
        )


    # =====================================================
    # OVERALL RECOMMENDATION
    # =====================================================

    st.markdown(
        '<div class="section-title">🧠 Overall Recommendation</div>',
        unsafe_allow_html=True
    )

    if score >= 70:

        st.success(
            "Your resume shows strong textual similarity with this job description."
        )

    elif score >= 40:

        st.info(
            "Your resume shows moderate similarity. "
            "Consider improving relevant keywords and missing skills."
        )

    else:

        st.warning(
            "Your resume currently shows low textual similarity. "
            "Consider improving relevant skills, keywords and project descriptions."
        )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = f"""
AI-POWERED RESUME ANALYZER
==========================

Resume Match Score: {score}%

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
Sections Detected: {len(found_sections)}

Resume Sections:
{", ".join(found_sections)}

Recommendations:
{chr(10).join("- " + x for x in suggestions)}
"""


    st.download_button(
        label="📥 Download Analysis Report",
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
    Built with Python • Streamlit • NLP • TF-IDF • OCR
</div>
""", unsafe_allow_html=True)
