import streamlit as st
import fitz
import pytesseract
import re
import os
import shutil
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #666;
    margin-bottom: 25px;
}

.card {
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #ddd;
    background: white;
    margin-bottom: 15px;
}

.metric-title {
    font-size: 15px;
    color: #666;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 5px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

.skill {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 15px;
    background: #f0f2f6;
    border: 1px solid #ddd;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.markdown(
    '<div class="main-title">📄 AI-Powered Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze your resume against a job description and identify skill gaps, '
    'similarity and improvement areas.'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("⚙️ Analyzer")

    st.write(
        "Upload your resume and paste the target job description "
        "to analyze the match."
    )

    st.divider()

    st.write("### Features")
    st.write("📊 Resume Match Score")
    st.write("🎯 Skill Gap Analysis")
    st.write("🔍 Keyword Coverage")
    st.write("📑 Resume Section Detection")
    st.write("💡 Improvement Suggestions")
    st.write("🖨️ OCR Support")
    st.write("📥 Download Report")


# =========================
# INPUTS
# =========================

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


# =========================
# SKILLS DATABASE
# =========================

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


# =========================
# SKILL DETECTION
# =========================

def contains_skill(text, skill):

    text = text.lower()
    skill = skill.lower()

    pattern = r"\b" + re.escape(skill) + r"\b"

    return re.search(pattern, text) is not None


# =========================
# PDF + OCR EXTRACTION
# =========================

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


# =========================
# TESSERACT CONFIG
# =========================

tesseract_path = shutil.which("tesseract")

windows_tesseract = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if tesseract_path:

    pytesseract.pytesseract.tesseract_cmd = tesseract_path

elif os.path.exists(windows_tesseract):

    pytesseract.pytesseract.tesseract_cmd = windows_tesseract


# =========================
# ANALYZE BUTTON
# =========================

if st.button("🔍 Analyze Resume", use_container_width=True):

    if uploaded_file is None:

        st.warning("Please upload a resume PDF.")

        st.stop()

    if not job_description.strip():

        st.warning("Please paste a job description.")

        st.stop()


    # =========================
    # EXTRACT RESUME
    # =========================

    try:

        pdf_bytes = uploaded_file.getvalue()

        resume_text = extract_resume_text(pdf_bytes)

    except Exception as e:

        st.error(
            f"Could not read the PDF: {e}"
        )

        st.stop()


    if not resume_text.strip():

        st.error(
            "Could not extract any text from this PDF."
        )

        st.stop()


    # =========================
    # TEXT SIMILARITY
    # =========================

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


    # =========================
    # SKILLS
    # =========================

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


    # =========================
    # KEYWORD COVERAGE
    # =========================

    if required_skills:

        matched_count = len(
            [
                skill
                for skill in required_skills
                if skill in resume_skills
            ]
        )

        keyword_coverage = round(
            matched_count / len(required_skills) * 100,
            2
        )

    else:

        keyword_coverage = 0


    # =========================
    # RESUME STATISTICS
    # =========================

    words = resume_text.split()

    word_count = len(words)

    character_count = len(resume_text)


    # =========================
    # SECTION DETECTION
    # =========================

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


    # =========================
    # DASHBOARD
    # =========================

    st.markdown(
        '<div class="section-title">📊 Analysis Dashboard</div>',
        unsafe_allow_html=True
    )


    # OLD STYLE SCORE — KEEPING YOUR 30.51% FORMAT

    st.subheader("📊 Analysis Results")

    st.metric(
        "Resume Match Score",
        f"{score}%"
    )


    # ADVANCED METRICS

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">
                    Resume Match
                </div>

                <div class="metric-value">
                    {score}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with m2:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">
                    Keyword Coverage
                </div>

                <div class="metric-value">
                    {keyword_coverage}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with m3:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">
                    Skills Found
                </div>

                <div class="metric-value">
                    {len(resume_skills)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with m4:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">
                    Missing Skills
                </div>

                <div class="metric-value">
                    {len(missing_skills)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =========================
    # PROGRESS
    # =========================

    st.subheader("🎯 Resume Match Score")

    st.progress(
        min(score / 100, 1.0)
    )


    # =========================
    # SKILLS FOUND
    # =========================

    st.markdown(
        '<div class="section-title">✅ Skills Found</div>',
        unsafe_allow_html=True
    )

    if resume_skills:

        html = ""

        for skill in resume_skills:

            html += (
                f'<span class="skill">{skill}</span>'
            )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No predefined matching skills detected."
        )


    # =========================
    # MISSING SKILLS
    # =========================

    st.markdown(
        '<div class="section-title">⚠️ Missing Skills</div>',
        unsafe_allow_html=True
    )

    if missing_skills:

        html = ""

        for skill in missing_skills:

            html += (
                f'<span class="skill">{skill}</span>'
            )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    else:

        st.success(
            "No missing predefined skills detected."
        )


    # =========================
    # JOB DESCRIPTION SKILLS
    # =========================

    st.markdown(
        '<div class="section-title">💼 Job Description Skills</div>',
        unsafe_allow_html=True
    )

    if required_skills:

        st.write(
            ", ".join(required_skills)
        )

    else:

        st.info(
            "No predefined skills detected in the job description."
        )


    # =========================
    # RESUME STATISTICS
    # =========================

    st.markdown(
        '<div class="section-title">📈 Resume Statistics</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3 = st.columns(3)

    with s1:

        st.metric(
            "Word Count",
            word_count
        )

    with s2:

        st.metric(
            "Characters",
            character_count
        )

    with s3:

        st.metric(
            "Sections Detected",
            len(found_sections)
        )


    # =========================
    # RESUME SECTIONS
    # =========================

    st.markdown(
        '<div class="section-title">📑 Resume Sections</div>',
        unsafe_allow_html=True
    )

    if found_sections:

        st.write(
            ", ".join(found_sections)
        )

    else:

        st.info(
            "No standard resume sections detected."
        )


    # =========================
    # IMPROVEMENT SUGGESTIONS
    # =========================

    st.markdown(
        '<div class="section-title">💡 Improvement Suggestions</div>',
        unsafe_allow_html=True
    )

    suggestions = []

    if score < 40:

        suggestions.append(
            "Increase alignment between your resume and the target job description."
        )

    elif score < 70:

        suggestions.append(
            "Your resume has moderate similarity. Add more relevant keywords and project experience."
        )

    else:

        suggestions.append(
            "Your resume has strong textual similarity with this job description."
        )


    if missing_skills:

        suggestions.append(
            "Consider learning or demonstrating these relevant skills: "
            + ", ".join(missing_skills[:8])
            + "."
        )


    if word_count < 250:

        suggestions.append(
            "Your resume appears short. Consider adding relevant projects, skills or achievements."
        )


    if "projects" not in resume_text.lower():

        suggestions.append(
            "Consider adding a dedicated Projects section."
        )


    if "experience" not in resume_text.lower():

        suggestions.append(
            "If applicable, add internships, practical experience or relevant work."
        )


    for suggestion in suggestions:

        st.info(
            "💡 " + suggestion
        )


    # =========================
    # RECOMMENDATION
    # =========================

    st.markdown(
        '<div class="section-title">🧠 Overall Recommendation</div>',
        unsafe_allow_html=True
    )

    if score >= 70:

        st.success(
            "Your resume shows strong similarity with the target job description."
        )

    elif score >= 40:

        st.info(
            "Your resume shows moderate similarity. "
            "Improving missing skills and relevant keywords may increase alignment."
        )

    else:

        st.warning(
            "Your resume currently shows low textual similarity. "
            "Consider improving relevant skills, keywords and project descriptions."
        )


    # =========================
    # DOWNLOAD REPORT
    # =========================

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


# =========================
# FOOTER
# =========================

st.markdown(
    """
    <div class="footer">
        AI-Powered Resume Analyzer • Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
