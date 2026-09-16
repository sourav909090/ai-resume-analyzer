import streamlit as st
import fitz
import pytesseract
import re
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tesseract location
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄"
)

st.title("📄 AI-Powered Resume Analyzer")
st.write(
    "Compare a resume with a job description and identify "
    "relevant and missing skills."
)

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=200
)


def contains_skill(text, skill):
    pattern = r"\b" + re.escape(skill.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def extract_resume_text(pdf_bytes):
    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        page_text = page.get_text()

        # Normal text PDF
        if page_text.strip():
            text += page_text + "\n"

        # Image/scanned PDF -> OCR
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


if st.button("🔍 Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload a resume PDF.")

    elif not job_description.strip():
        st.warning("Please paste a job description.")

    else:

        # Extract resume text
        try:
            pdf_bytes = uploaded_file.getvalue()
            resume_text = extract_resume_text(pdf_bytes)

        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

        if not resume_text.strip():
            st.error("Could not extract any text from this PDF.")
            st.stop()

        # Resume match score
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

        # Skills
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
            "pytorch"
        ]

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

        # Results
        st.subheader("📊 Analysis Results")

        st.metric(
            "Resume Match Score",
            f"{score}%"
        )

        st.subheader("✅ Skills Found")

        if resume_skills:
            st.write(", ".join(resume_skills))
        else:
            st.write("No matching skills detected.")

        st.subheader("🎯 Required Skills")

        if required_skills:
            st.write(", ".join(required_skills))
        else:
            st.write("No predefined skills detected.")

        st.subheader("⚠️ Missing Skills")

        if missing_skills:
            st.write(", ".join(missing_skills))
        else:
            st.success("No missing skills detected.")

        st.subheader("💡 Recommendation")

        if score >= 70:
            st.success(
                "Your resume has strong similarity with this job description."
            )

        elif score >= 40:
            st.info(
                "Your resume has moderate similarity. "
                "Consider improving the missing skills."
            )

        else:
            st.warning(
                "Your resume has low similarity. "
                "Consider adding relevant skills and projects."
            )
