import streamlit as st
import json
import PyPDF2
from ats_engine import process_resume

import os

st.set_page_config(page_title="ATS Evaluation Engine", layout="wide", page_icon="📄")

st.title("📄 SAAS ATS Evaluation & Resume Optimization Engine")
st.markdown("Upload your resume and a target job description to get a compatibility score, missing skills analysis, and an optimized LaTeX resume ready for ATS systems.")

from dotenv import load_dotenv

load_dotenv()

# Try to get API key from environment first, then secrets
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if not api_key:
    st.error("🚨 API Key not found! Please set the `GEMINI_API_KEY` in your `.env` file or environment variables to proceed.")
    st.stop()

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    resume_file = st.file_uploader("Upload your current resume (PDF or TXT)", type=["pdf", "txt"])

with col2:
    st.subheader("2. Job Description")
    job_desc_text = st.text_area("Paste the Target Job Description", height=250)

if st.button("Evaluate & Optimize Resume", type="primary", use_container_width=True):
    if not resume_file:
        st.error("Please upload a resume.")
    elif not job_desc_text.strip():
        st.error("Please provide a job description.")
    else:
        # Extract text from the uploaded resume
        try:
            resume_text = ""
            if resume_file.name.lower().endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(resume_file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        resume_text += extracted + "\n"
            else:
                resume_text = resume_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Failed to read resume file: {e}")
            st.stop()

        if not resume_text.strip():
            st.error("Resume file appears to be empty or unreadable.")
            st.stop()

        with st.spinner("🤖 Analyzing your resume with Gemini... This may take a moment."):
            try:
                # Call our ATS engine
                result_json_str = process_resume(resume_text, job_desc_text, api_key)
                
                # Parse the returned JSON
                try:
                    # Try to strip markdown code blocks if the model included them by mistake
                    if result_json_str.strip().startswith("```json"):
                        result_json_str = result_json_str.strip()[7:-3]
                    elif result_json_str.strip().startswith("```"):
                        result_json_str = result_json_str.strip()[3:-3]
                        
                    result_data = json.loads(result_json_str)
                except json.JSONDecodeError as e:
                    st.error("Engine failed to return valid JSON.")
                    with st.expander("Show Raw Engine Output"):
                        st.text(result_json_str)
                    st.stop()

                st.success("✅ Analysis Complete!")

                # Create tabs for the output
                tab1, tab2, tab3 = st.tabs(["📊 Scoring Report", "📝 Optimized Resume", "💻 LaTeX Source"])

                with tab1:
                    scoring_report = result_data.get("scoring_report", {})
                    
                    # Top-level score
                    match_score = scoring_report.get("match_score", 0)
                    st.metric(label="ATS Match Score", value=f"{match_score} / 100")
                    
                    st.divider()
                    
                    # Detailed sections
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Score Breakdown")
                        bd = scoring_report.get("score_breakdown", {})
                        for k, v in bd.items():
                            st.write(f"**{k.replace('_', ' ').title()}:** {v}")
                            
                        st.subheader("Missing Mandatory Skills")
                        missing_skills = scoring_report.get("missing_mandatory_skills", [])
                        if missing_skills:
                            for skill in missing_skills:
                                st.write(f"- 🔴 {skill}")
                        else:
                            st.write("✅ None identified!")

                    with c2:
                        st.subheader("Improvement Suggestions")
                        suggestions = scoring_report.get("improvement_suggestions", [])
                        if suggestions:
                            for sugg in suggestions:
                                st.write(f"- 💡 {sugg}")
                        else:
                            st.write("✅ No further suggestions.")

                with tab2:
                    st.subheader("Structured ATS-Friendly Resume Data")
                    st.json(result_data.get("optimized_resume", {}))

                with tab3:
                    st.subheader("Compiled LaTeX Code")
                    latex_code = result_data.get("latex_code", "")
                    
                    st.download_button(
                        label="⬇️ Download LaTeX (.tex)",
                        data=latex_code,
                        file_name="optimized_resume.tex",
                        mime="text/plain",
                    )
                    
                    st.code(latex_code, language="latex")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
