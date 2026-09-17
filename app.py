import streamlit as st
import fitz
import pytesseract
import re
import shutil
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------------- TESSERACT ----------------
windows_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
elif windows_tesseract:
    pytesseract.pytesseract.tesseract_cmd = windows_tesseract

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}

.metric-title {
    font-size: 14px;
    color: #6b7280;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
}

.skill {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    background: #eef2ff;
    color: #4338ca;
    font-size: 14px;
}

.missing {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    background: #fff1f2;
    color: #be123c;
    font-size: 14px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📄 AI-Powered Resume Analyzer</h1>
    <p>Analyze your resume against a job description and discover your skill gaps.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Analyzer Settings")
    st.write("Upload your resume and provide the target job description.")
    
    st.info(
        "💡 Tip: Use a complete job description for more meaningful analysis."
    )

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "📎 Upload Resume (PDF)",
        type=["pdf"]
    )

with col2:
    job_description = st.text_area(
        "💼 Paste Job Description",
        height=180,
        placeholder="Paste the complete job description here..."
    )

# ---------------- FUNCTIONS ----------------

def contains_skill(text, skill):
    text = text.lower()
    skill = skill.lower()

    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
    return re.search(pattern, text) is not None


def extract_resume_text(pdf_bytes):

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in pdf:

        page_text = page.get_text()

        if page_text.strip():

            text += page_text + "\n"

        else:

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            ocr_text = pytesseract.image_to_string(img)

            text += ocr_text + "\n"

    pdf.close()

    return text


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
    "docker",
    "linux",
    "flask",
    "django",
    "power bi",
    "excel"
]


def detect_sections(text):

    text_lower = text.lower()

    possible_sections = {
        "Contact Information": [
            "email", "phone", "linkedin"
        ],
        "Education": [
            "education", "b.tech", "btech", "degree", "university"
        ],
        "Experience": [
            "experience", "work experience", "internship"
        ],
        "Projects": [
            "projects", "project"
        ],
        "Skills": [
            "skills", "technical skills"
        ],
        "Certifications": [
            "certification", "certifications"
        ]
    }

    found = []

    for section, keywords in possible_sections.items():

        if any(keyword in text_lower for keyword in keywords):
            found.append(section)

    return found


def generate_suggestions(
    score,
    missing_skills,
    sections,
    word_count
):

    suggestions = []

    if score < 40:
        suggestions.append(
            "Improve alignment between your resume and the target job description."
        )

    if missing_skills:
        suggestions.append(
            "Consider learning or demonstrating relevant missing skills through projects or coursework."
        )

    if "Projects" not in sections:
        suggestions.append(
            "Add a dedicated Projects section with measurable achievements."
        )

    if "Experience" not in sections:
        suggestions.append(
            "If applicable, add internships, freelance work, or practical experience."
        )

    if word_count < 150:
        suggestions.append(
            "Your resume contains relatively little text. Consider adding relevant achievements and project details."
        )

    if not suggestions:
        suggestions.append(
            "Your resume has good basic alignment. Continue improving project depth and measurable achievements."
        )

    return suggestions


# ---------------- ANALYZE ----------------

if st.button("🔍 Analyze Resume", use_container_width=True):

    if uploaded_file is None:

        st.warning("⚠️ Please upload your resume PDF.")

    elif not job_description.strip():

        st.warning("⚠️ Please paste a job description.")

    else:

        with st.spinner("🔄 Analyzing your resume..."):

            try:

                pdf_bytes = uploaded_file.getvalue()

                resume_text = extract_resume_text(pdf_bytes)

            except Exception as e:

                st.error(f"❌ Could not read the PDF: {e}")

                st.stop()

            if not resume_text.strip():

                st.error(
                    "❌ Could not extract text from this PDF."
                )

                st.stop()

            # -------- TEXT SIMILARITY --------

            documents = [
                resume_text,
                job_description
            ]

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            vectors = vectorizer.fit_transform(documents)

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )

            score = round(
                similarity[0][0] * 100,
                2
            )

            # -------- SKILLS --------

            resume_skills = [
                skill
                for skill in skills
                if contains_skill(
                    resume_text,
                    skill
                )
            ]

            required_skills = [
                skill
                for skill in skills
                if contains_skill(
                    job_description,
                    skill
                )
            ]

            missing_skills = [
                skill
                for skill in required_skills
                if skill not in resume_skills
            ]

            # -------- ATS KEYWORDS --------

            keyword_coverage = 0

            if required_skills:

                keyword_coverage = round(
                    (
                        len(resume_skills)
                        / len(required_skills)
                    ) * 100,
                    1
                )

                keyword_coverage = min(
                    keyword_coverage,
                    100
                )

            # -------- RESUME STATS --------

            words = resume_text.split()

            word_count = len(words)

            character_count = len(resume_text)

            sections = detect_sections(
                resume_text
            )

            suggestions = generate_suggestions(
                score,
                missing_skills,
                sections,
                word_count
            )

            # ================= RESULTS =================

            st.markdown(
                '<div class="section-title">📊 Analysis Dashboard</div>',
                unsafe_allow_html=True
            )

            # -------- METRICS --------

            m1, m2, m3, m4 = st.columns(4)

            with m1:

                st.markdown(
                    f"""
                    <div class="card">
                    <div class="metric-title">Resume Match</div>
                    <div class="metric-value">{score}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m2:

                st.markdown(
                    f"""
                    <div class="card">
                    <div class="metric-title">Keyword Coverage</div>
                    <div class="metric-value">{keyword_coverage}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m3:

                st.markdown(
                    f"""
                    <div class="card">
                    <div class="metric-title">Skills Found</div>
                    <div class="metric-value">{len(resume_skills)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m4:

                st.markdown(
                    f"""
                    <div class="card">
                    <div class="metric-title">Missing Skills</div>
                    <div class="metric-value">{len(missing_skills)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -------- SCORE BAR --------

            st.subheader("🎯 Resume Match Score")

            st.progress(
                min(score / 100, 1.0)
            )

            if score >= 70:

                st.success(
                    "Strong textual alignment with the job description."
                )

            elif score >= 40:

                st.info(
                    "Moderate alignment. Review the missing skills and keywords."
                )

            else:

                st.warning(
                    "Low textual alignment. Consider improving relevant keywords and project experience."
                )

            # -------- TWO COLUMN RESULTS --------

            left, right = st.columns(2)

            with left:

                st.subheader("✅ Skills Found")

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

                    st.write(
                        "No predefined skills detected."
                    )

            with right:

                st.subheader("⚠️ Missing Skills")

                if missing_skills:

                    html = ""

                    for skill in missing_skills:

                        html += (
                            f'<span class="missing">{skill}</span>'
                        )

                    st.markdown(
                        html,
                        unsafe_allow_html=True
                    )

                else:

                    st.success(
                        "No missing predefined skills detected."
                    )

            # -------- REQUIRED SKILLS --------

            st.subheader("💼 Job Description Skills")

            if required_skills:

                st.write(
                    ", ".join(required_skills)
                )

            else:

                st.write(
                    "No predefined skills detected in the job description."
                )

            # -------- RESUME STATISTICS --------

            st.subheader("📈 Resume Statistics")

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
                    len(sections)
                )

            # -------- SECTIONS --------

            st.subheader("📑 Resume Sections")

            if sections:

                st.write(
                    " • ".join(sections)
                )

            else:

                st.warning(
                    "No common resume sections detected."
                )

            # -------- SUGGESTIONS --------

            st.subheader("💡 Improvement Suggestions")

            for suggestion in suggestions:

                st.info(
                    "💡 " + suggestion
                )

            # -------- DOWNLOAD REPORT --------

            report = f"""
AI-POWERED RESUME ANALYZER
==========================

Resume Match Score: {score}%

Keyword Coverage: {keyword_coverage}%

Skills Found:
{", ".join(resume_skills)}

Missing Skills:
{", ".join(missing_skills)}

Required Skills:
{", ".join(required_skills)}

Resume Word Count: {word_count}

Detected Sections:
{", ".join(sections)}

IMPROVEMENT SUGGESTIONS
-----------------------

"""

            for suggestion in suggestions:

                report += "- " + suggestion + "\n"

            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name="resume_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

            # -------- FOOTER --------

            st.markdown("---")

            st.caption(
                "AI-Powered Resume Analyzer • Built with Python, Streamlit, OCR and NLP techniques"
            )
