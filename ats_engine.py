import os
import json
import argparse
import PyPDF2
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Dict, Any

def parse_pdf(file_path: str) -> str:
    """Extracts text from a given PDF file."""
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def parse_text(file_path: str) -> str:
    """Reads text from a generic text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_resume(resume_text: str, job_description: str, api_key: str) -> str:
    """Calls Gemini with strict instructions to generate JSON."""
    genai.configure(api_key=api_key)
    
    # We enforce JSON output structure directly in the prompt
    prompt = f"""You are a strict ATS evaluation and resume optimization engine. Analyze the provided resume text and job description carefully. First, extract structured information from the resume, including name, contact details, summary, skills, experience (company, role, duration, bullet points), projects, and education. Then extract structured job requirements from the job description, including mandatory skills, optional skills, tools, required experience years, and key responsibilities. Compare both and calculate an ATS compatibility score out of 100 using this weighting: mandatory skills 40%, experience alignment 20%, responsibility similarity 20%, optional skills 10%, and clarity and relevance 10%. Be strict but fair, and do not inflate the score. Identify missing mandatory skills and provide clear improvement suggestions.

CRITICAL OPTIMIZATION STEP: You must rewrite the professional summary and ALL experience bullet points to specifically target and match the keywords, phrases, and critical responsibilities found in the Job Description. Maximize the ATS match score by strategically incorporating the exact job requirements naturally into the experience bullets, using strong action verbs and measurable impact where possible. Do not invent technologies, tools, or experience not present in the original resume.

Finally, output three sections: (1) an ATS report with score breakdown and missing skills, (2) an optimized structured resume in JSON format, and (3) ATS-friendly single-column LaTeX resume code without tables, images, icons, or decorative formatting. Ensure all outputs are valid, structured, and ready for production use.

Output MUST strictly match this JSON schema format (do not include markdown block ticks like ```json, just output the raw JSON):
{{
    "scoring_report": {{
        "match_score": 0,
        "score_breakdown": {{
            "mandatory_skills": 0,
            "experience_alignment": 0,
            "responsibility_similarity": 0,
            "optional_skills": 0,
            "clarity_and_relevance": 0
        }},
        "missing_mandatory_skills": ["skill1", "skill2"],
        "improvement_suggestions": ["suggestion1"]
    }},
    "optimized_resume": {{
        "name": "Full Name",
        "contact_info": {{"email": "...", "phone": "...", "linkedin": "..."}},
        "summary": "Professional summary...",
        "skills": ["skill_a", "skill_b"],
        "experience": [
            {{
                "company": "Company Name",
                "role": "Job Title",
                "duration": "Duration",
                "bullet_points": ["optimized bullet 1", "optimized bullet 2"]
            }}
        ],
        "projects": [
            {{
                "name": "Project Name",
                "description": "...",
                "technologies": ["tech1"]
            }}
        ],
        "education": [
            {{"degree": "...", "institution": "...", "year": "..."}}
        ]
    }},
    "latex_code": "\\\\documentclass{{article}}..."
}}

--- RESUME TEXT ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}
"""
    
    # Configure model to return JSON
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )
    
    return response.text

def main():
    parser = argparse.ArgumentParser(description="ATS Evaluation and Optimization Engine")
    parser.add_argument("--resume", required=True, help="Path to the user's current resume (PDF or TXT)")
    parser.add_argument("--job", required=True, help="Path to the target job description (TXT)")
    parser.add_argument("--outdir", default="output", help="Directory to save the ATS reports and updated resume")
    
    args = parser.parse_args()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set GEMINI_API_KEY environment variable. E.g., export GEMINI_API_KEY='your_key'")
        return
        
    print(f"Reading Resume: {args.resume}")
    if args.resume.lower().endswith('.pdf'):
        resume_text = parse_pdf(args.resume)
    else:
        resume_text = parse_text(args.resume)
        
    print(f"Reading Job Description: {args.job}")
    job_desc = parse_text(args.job)
    
    print("Sending data to the ATS Engine (Gemini) for evaluation and optimization...")
    # Get JSON output from Gemini
    result_json_str = process_resume(resume_text, job_desc, api_key)
    
    try:
        # Validate returned JSON
        result_data = json.loads(result_json_str)
        
        os.makedirs(args.outdir, exist_ok=True)
        
        # 1. Scoring Report
        scoring_path = os.path.join(args.outdir, "scoring_report.json")
        with open(scoring_path, "w", encoding="utf-8") as f:
            json.dump(result_data.get("scoring_report", {}), f, indent=4)
            
        # 2. Optimized Resume JSON
        opt_resume_path = os.path.join(args.outdir, "optimized_resume.json")
        with open(opt_resume_path, "w", encoding="utf-8") as f:
            json.dump(result_data.get("optimized_resume", {}), f, indent=4)
            
        # 3. LaTeX Code
        latex_path = os.path.join(args.outdir, "optimized_resume.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(result_data.get("latex_code", ""))
            
        print(f"\nSuccess! Reports generated in '{args.outdir}' directory:")
        print(f"  - {scoring_path}")
        print(f"  - {opt_resume_path}")
        print(f"  - {latex_path}")
        
        score_val = result_data.get('scoring_report', {}).get('match_score', 'N/A')
        print(f"\n>>> Final ATS Match Score: {score_val} / 100 <<<")
        
    except json.JSONDecodeError as e:
        print("Error: The engine failed to return valid JSON.")
        print(f"Details: {e}")
        print("Raw Output:")
        print(result_json_str)

if __name__ == "__main__":
    main()
