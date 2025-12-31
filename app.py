from flask import Flask, render_template, request, jsonify, session, redirect
from flask_cors import CORS
import os
from datetime import datetime
import json
import PyPDF2
from werkzeug.utils import secure_filename
import requests
import traceback
from pymongo import MongoClient
# CrewAI + LLM imports
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
import os
os.environ["CREWAI_DISABLE_RICH"] = "true"

from backend.rag_engine import add_interview_to_faiss
from werkzeug.security import generate_password_hash, check_password_hash

# --- CREATE FLASK APP FIRST ---
app = Flask(__name__)
app.secret_key = "your-secret-key-here"
CORS(app)

# --- THEN CONNECT MONGO ---
try:
    MONGO_URI = ""
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000)

    db = mongo_client["interview_system"]
    students_col = db["students"]
    interviews_col = db["interviews"]
    admins_col = db["admins"]

    print("✅ MongoDB connected successfully!")

except Exception as e:
    print("❌ MongoDB connection failed:", e)
    traceback.print_exc()

# This must be defined AFTER Flask app is created
def require_admin():
    return "admin_email" in session

# --- IMPORT ADMIN ROUTES AFTER app + db ARE READY ---
from admin_routes import admin_bp
app.register_blueprint(admin_bp, url_prefix="/api/admin")


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

'''from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # or another Gemini model
    temperature=0.7,
)'''
"""
import google.generativeai as genai



from crewai import LLM

llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=""
)
"""
os.environ["GROQ_API_KEY"] = ""
# LLM Configuration
USE_GROQ = True

if USE_GROQ:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="groq/llama-3.1-8b-instant"
    )
else:
    llm = Ollama(model="llama2")
"""
USE_GROQ = True


os.environ["GROQ_API_KEY"] = ""

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="groq/llama-3.1-8b-instant"
)"""

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

interviews = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

# ============================================================================
# CREWAI AGENTS DEFINITION
# ============================================================================

def create_interview_crew(resume_data, job_description, interview_type):
    """Create specialized CrewAI agents for conducting human-like interviews"""
    
    resume_analyzer = Agent(
        role='Resume Analysis Expert',
        goal='Extract ONLY factual information explicitly stated in the resume. Never infer or assume skills not mentioned.',
        backstory="""You are a meticulous HR analyst who ONLY reports what is explicitly 
        written in resumes. You never assume or infer skills. If something isn't clearly 
        stated, you note it as "not mentioned". You categorize information exactly as 
        presented without embellishment.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    jd_matcher = Agent(
        role='Job Description Matching Specialist',
        goal='Match ONLY explicitly stated candidate skills against job requirements. Flag uncertainties clearly.',
        backstory="""You are an objective recruiter who matches candidates based solely 
        on documented evidence. You clearly distinguish between "confirmed match" (skill 
        explicitly mentioned), "possible match" (related skill mentioned), and "no evidence" 
        (skill not found). You never assume proficiency.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    question_generator = Agent(
        role='Conversational Technical Interviewer',
        goal='Ask natural, concise questions like a friendly human interviewer. Keep questions short (1-2 sentences max).',
        backstory="""You are a senior engineer who conducts interviews in a warm, 
        conversational style. You ask ONE clear question at a time, keep it brief, 
        and sound natural - not robotic. You adapt to the candidate's communication 
        style. You NEVER repeat questions or ask about things already answered.
        
        Your style: Friendly but professional. Like chatting with a colleague, not 
        reading from a script. You use simple language and avoid jargon-heavy questions.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    response_evaluator = Agent(
        role='Fair Response Evaluator',
        goal='Evaluate responses based ONLY on what the candidate actually said. Quote specific parts as evidence.',
        backstory="""You are a fair, objective evaluator who scores based solely on 
        the candidate's actual words. You always cite specific quotes from their response 
        as evidence for your ratings. You never assume knowledge they didn't demonstrate.
        If an answer is incomplete, you note what was missing, not what you think they know.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    concept_verifier = Agent(
        role='Adaptive Follow-up Specialist',
        goal='Generate appropriate follow-ups based on response quality. Handle confused, off-topic, or brief responses gracefully.',
        backstory="""You are skilled at reading the room. If a candidate seems confused, 
        you rephrase or offer hints. If they go off-topic, you gently redirect. If they 
        give brief answers, you encourage elaboration warmly. You handle awkward moments 
        with grace and keep the conversation flowing naturally.
        
        You can detect: confusion (ask simpler version), nervousness (be encouraging), 
        off-topic responses (acknowledge then redirect), invalid inputs (clarify what you need).""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    report_generator = Agent(
        role='Evidence-Based Report Writer',
        goal='Generate reports citing ONLY demonstrated skills with specific examples from the interview.',
        backstory="""You write hiring reports that are defensible and evidence-based. 
        Every claim must reference specific interview moments. You clearly separate 
        "demonstrated" skills from "claimed but not verified" skills. Your recommendations 
        are conservative and based on observed performance, not assumptions.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return {
        'resume_analyzer': resume_analyzer,
        'jd_matcher': jd_matcher,
        'question_generator': question_generator,
        'response_evaluator': response_evaluator,
        'concept_verifier': concept_verifier,
        'report_generator': report_generator
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_user_pattern(responses):
    """Detect user communication pattern for adaptive questioning"""
    if not responses:
        return "normal"
    
    last_responses = responses[-3:] if len(responses) >= 3 else responses
    avg_length = sum(len(r) for r in last_responses) / len(last_responses)
    
    confusion_words = ['not sure', "don't know", 'confused', 'what do you mean', 
                       'can you clarify', "i don't understand", '?', 'sorry', 'um']
    if any(any(word in r.lower() for word in confusion_words) for r in last_responses):
        return "confused"
    
    off_topic_indicators = ['by the way', 'reminds me', 'speaking of', 'unrelated but']
    if any(any(ind in r.lower() for ind in off_topic_indicators) for r in last_responses):
        return "off_topic"
    
    if avg_length > 500:
        return "chatty"
    
    if avg_length < 50:
        return "brief"
    
    return "normal"

# ------------------- ADDED: Strict bypass detection (Option A location) -------------------
def detect_bypass_attempt(user_message):
    """
    Strict Mode (Option 1) bypass detection.
    Returns:
      - "jailbreak" for explicit jailbreak / prompt-injection attempts
      - "reverse_question" for any user question (strict blocking)
      - None if nothing suspicious detected
    """
    if not user_message:
        return None

    s = user_message.lower().strip()

    # Known jailbreak/prompt-injection indicators
    jailbreak_terms = [
        "ignore previous", "ignore all", "jailbreak", "prompt injection",
        "act as", "change role", "break character", "bypass", "system prompt",
        "reveal the system", "reveal system", "show me the system", "show system prompt",
        "disclose system", "give me the prompt", "tell me your instructions"
    ]
    if any(term in s for term in jailbreak_terms):
        return "jailbreak"

    # In Strict Mode: block any question-like input.
    # Check for explicit question mark
    if "?" in s:
        return "reverse_question"

    # Check for leading interrogatives or common question starters
    question_words = {
        "what", "why", "how", "when", "where", "who", "which", "is", "are", "do",
        "does", "did", "can", "could", "would", "should", "may", "might"
    }
    tokens = s.split()
    if tokens and tokens[0] in question_words:
        return "reverse_question"

    # Phrases that indicate the user is asking for explanation/answers
    question_phrases = [
        "tell me", "explain", "define", "teach me", "give me", "show me",
        "how do i", "how to", "what is", "could you explain", "please explain"
    ]
    if any(phrase in s for phrase in question_phrases):
        return "reverse_question"

    return None
# -----------------------------------------------------------------------------------------
def calculate_resume_score(resume_data):
    skills = resume_data.get("skills", {})
    total_skills = len(skills.get("explicitly_listed", [])) + len(skills.get("mentioned_in_context", []))

    projects = len(resume_data.get("projects_mentioned", []))
    certifications = len(resume_data.get("certifications", []))
    exp_years = resume_data.get("experience_years", "0")

    try:
        exp_years = float(exp_years) if isinstance(exp_years, str) else exp_years
    except:
        exp_years = 0

    score = (
        total_skills * 0.4 +
        projects * 0.3 +
        certifications * 0.2 +
        exp_years * 0.3
    )

    return min(10, round(score, 2))

def clean_question(question):
    """Post-process to ensure question is natural and concise"""
    prefixes_to_remove = [
        "Based on your resume,", "Given your experience,", "As someone with expertise in,",
        "Considering your background,", "I'd like to ask you about", "My next question is:",
        "Question:", "Here's my question:", "Let me ask you:", "I want to know",
        "Could you please tell me", "I would like to understand"
    ]
    
    for prefix in prefixes_to_remove:
        if question.lower().startswith(prefix.lower()):
            question = question[len(prefix):].strip()
    
    if question and not question.endswith('?') and not question.endswith('.'):
        question += '?'
    
    if question:
        question = question[0].upper() + question[1:]
    
    return question

def assess_response_quality(response):
    """Enhanced assessment that catches non-answers"""
    if not response or not response.strip():
        return "empty"
    
    response_lower = response.lower().strip()
    
    # Check for extremely short responses
    if len(response) < 5:
        return "empty"
    
    # Check for single words or very minimal effort
    words = response.split()
    if len(words) <= 2:
        return "minimal"
    
    # Check for "I don't know" variations
    no_knowledge_phrases = [
        "i don't know", "dont know", "not sure", "no idea", 
        "never heard", "no clue", "idk", "dunno", "?"
    ]
    if any(phrase in response_lower for phrase in no_knowledge_phrases) and len(response) < 50:
        return "no_knowledge"
    
    # Check for confusion
    if any(phrase in response_lower for phrase in ['what?', 'huh?', 'can you repeat', "don't understand", "confused"]):
        return "confused"
    
    # Check for off-topic
    if len(response) < 20:
        return "very_brief"
    if len(response) < 50:
        return "brief"
    
    # Check for excessive verbosity
    if len(response) > 1000:
        return "verbose"
    
    return "normal"

def create_fallback_evaluation(response, quality):
    """Fallback evaluation with realistic scoring for poor responses"""
    
    # Map quality to appropriate scores
    quality_scores = {
        "empty": 0,
        "minimal": 1,
        "no_knowledge": 2,
        "very_brief": 3,
        "brief": 4,
        "confused": 2,
        "uncertain": 3,
        "normal": 5,
        "verbose": 6
    }
    
    base_score = quality_scores.get(quality, 5)
    
    # Create evidence that reflects the actual quality
    evidence_map = {
        "empty": "No response provided",
        "minimal": "Response too brief to evaluate properly",
        "no_knowledge": "Candidate explicitly stated lack of knowledge",
        "very_brief": "Response lacks sufficient detail",
        "brief": "Limited elaboration provided",
        "confused": "Response indicates confusion with the question",
        "uncertain": "Candidate expressed uncertainty",
        "normal": "Standard response quality",
        "verbose": "Response provided with good detail"
    }
    
    evidence = evidence_map.get(quality, "Auto-evaluated")
    
    return {
        'scores': {
            'technical_accuracy': {'score': base_score, 'evidence': evidence},
            'depth_of_knowledge': {'score': base_score, 'evidence': evidence},
            'communication_clarity': {'score': max(1, base_score - 1), 'evidence': evidence},
            'practical_experience': {'score': base_score, 'evidence': evidence},
            'relevance_to_question': {'score': base_score, 'evidence': evidence}
        },
        'overall_score': base_score,
        'response_type': quality,
        'what_they_demonstrated': [] if base_score < 4 else ['Basic response provided'],
        'what_was_missing': ['Detailed explanation', 'Concrete examples', 'Technical depth'],
        'needs_follow_up': quality in ['confused', 'uncertain', 'very_brief', 'minimal', 'no_knowledge'],
        'follow_up_reason': f'Response quality: {quality}',
        'suggested_approach': 'Ask a different question' if quality in ['empty', 'no_knowledge'] else 'Encourage more detail'
    }

def format_conversation_excerpts(conversation):
    """Format conversation for report analysis"""
    excerpts = []
    for msg in conversation[-10:]:
        role = "Interviewer" if msg['role'] == 'assistant' else "Candidate"
        content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
        excerpts.append(f"{role}: {content}")
    return "\n".join(excerpts)

def create_fallback_report(interview_data, jd_match, evaluations):
    """Fallback report when generation fails"""
    avg_score = 5
    if evaluations:
        scores = [e.get('overall_score', 5) for e in evaluations if isinstance(e.get('overall_score'), (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
    
    recommendation = 'Strong Hire' if avg_score >= 8 else 'Hire' if avg_score >= 6 else 'Maybe' if avg_score >= 4 else 'No Hire'
    
    return {
        'executive_summary': {
            'overall_score': round(avg_score, 1),
            'recommendation': recommendation,
            'confidence': 'Medium',
            'one_line': 'Assessment completed. Review individual responses for details.',
            'key_evidence': 'See detailed evaluations below.'
        },
        'skills_assessment': {
            'verified_skills': [],
            'claimed_but_unverified': [],
            'concerning_gaps': []
        },
        'jd_alignment': {
            'match_percentage': jd_match.get('overall_match_percentage', 50),
            'must_haves_met': [],
            'must_haves_missing': []
        },
        'behavioral_observations': {
            'communication_style': 'See individual evaluations',
            'response_patterns': 'Varied',
            'red_flags': [],
            'green_flags': []
        },
        'detailed_recommendation': {
            'hire_decision': f'Based on average score of {round(avg_score, 1)}/10',
            'role_fit': 'Requires further assessment',
            'growth_potential': 'To be determined',
            'onboarding_needs': [],
            'interview_confidence': 'Medium'
        },
        'next_steps': 'Consider additional interview rounds for clarification'
    }

# ============================================================================
# CREWAI TASK FUNCTIONS
# ============================================================================

def analyze_resume_with_crew(resume_text, agents):
    """Strictly factual resume analysis - no inference"""
    
    task = Task(
        description=f"""Extract ONLY explicitly stated information from this resume.

=== RESUME TEXT ===
{resume_text}

=== CRITICAL RULES ===
1. Only list skills that are EXPLICITLY MENTIONED
2. Don't infer skills from job titles or project descriptions
3. If experience years aren't stated, calculate from work history dates or say "not specified"
4. Quote exact text when possible
5. Mark anything uncertain as "unclear" or "not specified"

Return JSON with ONLY verified information:
{{
    "skills": {{
        "explicitly_listed": ["skill1", "skill2"],
        "mentioned_in_context": ["tech mentioned in project descriptions"],
        "tools_named": ["specific tools/software mentioned"]
    }},
    "experience_years": "number or 'not specified'",
    "stated_job_titles": ["exact titles from resume"],
    "projects_mentioned": [
        {{"name": "project name if given", "techs_mentioned": ["tech1"]}}
    ],
    "education": "exactly as stated or 'not specified'",
    "certifications": ["exactly as listed"],
    "achievements_with_metrics": ["only ones with actual numbers"],
    "domains": ["industry domains mentioned"]
}}

Do NOT add skills not mentioned. Do NOT assume proficiency levels.""",
        agent=agents['resume_analyzer'],
        expected_output="Factual resume extraction in JSON"
    )
    
    crew = Crew(
        agents=[agents['resume_analyzer']],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        result_str = str(result)
        # Try to extract JSON from the result
        if '{' in result_str:
            json_start = result_str.index('{')
            json_end = result_str.rindex('}') + 1
            return json.loads(result_str[json_start:json_end])
        return json.loads(result_str)
    except Exception as e:
        print(f"Resume analysis error: {e}")
        return {
            'skills': {'explicitly_listed': [], 'mentioned_in_context': [], 'tools_named': []},
            'experience_years': 'not specified',
            'stated_job_titles': [],
            'projects_mentioned': [],
            'education': 'not specified',
            'certifications': [],
            'achievements_with_metrics': [],
            'domains': []
        }

def match_jd_with_resume(resume_data, job_description, agents):
    """Match job description with resume using CrewAI - evidence based"""
    
    skills = resume_data.get('skills', {})
    if isinstance(skills, dict):
        all_skills = skills.get('explicitly_listed', []) + skills.get('mentioned_in_context', []) + skills.get('tools_named', [])
    else:
        all_skills = skills if isinstance(skills, list) else []
    
    task = Task(
        description=f"""Compare the job description with candidate's DOCUMENTED profile.

=== JOB DESCRIPTION ===
{job_description}

=== CANDIDATE'S DOCUMENTED SKILLS ===
Explicitly Listed: {skills.get('explicitly_listed', []) if isinstance(skills, dict) else skills}
Mentioned in Context: {skills.get('mentioned_in_context', []) if isinstance(skills, dict) else []}
Tools Named: {skills.get('tools_named', []) if isinstance(skills, dict) else []}

=== CANDIDATE'S EXPERIENCE ===
Years: {resume_data.get('experience_years', 'not specified')}
Job Titles: {resume_data.get('stated_job_titles', [])}
Projects: {resume_data.get('projects_mentioned', [])}
Domains: {resume_data.get('domains', [])}

=== MATCHING RULES ===
1. Only mark as "confirmed match" if skill is EXPLICITLY listed
2. Mark as "possible match" if related skill exists
3. Mark as "no evidence" if skill not found at all
4. Be conservative - when in doubt, mark as "needs verification"

Return JSON:
{{
    "overall_match_percentage": 0-100,
    "must_have_skills": {{
        "required": ["skill1", "skill2"],
        "confirmed_match": ["skills candidate explicitly has"],
        "possible_match": ["related skills that might transfer"],
        "no_evidence": ["skills not found in resume"]
    }},
    "nice_to_have_skills": {{
        "required": ["skill3"],
        "candidate_has": ["skill3"],
        "missing": []
    }},
    "experience_match": {{
        "required_years": "from JD or 'not specified'",
        "candidate_years": "{resume_data.get('experience_years', 'not specified')}",
        "meets_requirement": true/false
    }},
    "key_strengths_for_role": ["strength1 based on evidence"],
    "potential_gaps": ["gap1 that needs probing"],
    "areas_to_probe_in_interview": ["specific areas to verify"],
    "recommendation": "Strong Match/Good Match/Moderate Match/Weak Match/Insufficient Data"
}}""",
        agent=agents['jd_matcher'],
        expected_output="Evidence-based JD matching analysis in JSON"
    )
    
    crew = Crew(
        agents=[agents['jd_matcher']],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        result_str = str(result)
        if '{' in result_str:
            json_start = result_str.index('{')
            json_end = result_str.rindex('}') + 1
            return json.loads(result_str[json_start:json_end])
        return json.loads(result_str)
    except Exception as e:
        print(f"JD matching error: {e}")
        return {
            'overall_match_percentage': 50,
            'must_have_skills': {'required': [], 'confirmed_match': [], 'possible_match': [], 'no_evidence': []},
            'nice_to_have_skills': {'required': [], 'candidate_has': [], 'missing': []},
            'experience_match': {'required_years': 'not specified', 'candidate_years': resume_data.get('experience_years', 'not specified'), 'meets_requirement': True},
            'key_strengths_for_role': [],
            'potential_gaps': [],
            'areas_to_probe_in_interview': ['General technical skills', 'Problem solving approach', 'Team collaboration'],
            'recommendation': 'Moderate Match'
        }

def generate_dynamic_question(resume_data, job_description, conversation_history, 
                             jd_match_data, question_count, agents):
    """Generate natural, human-like questions - handles various user types"""
    
    previous_questions = [
        msg['content'] for msg in conversation_history 
        if msg['role'] == 'assistant'
    ]
    
    candidate_responses = [
        msg['content'] for msg in conversation_history 
        if msg['role'] == 'user'
    ]
    
    user_pattern = detect_user_pattern(candidate_responses)
    areas_to_probe = jd_match_data.get('areas_to_probe_in_interview', [])
    gaps = jd_match_data.get('potential_gaps', [])
    
    skills = resume_data.get('skills', {})
    if isinstance(skills, dict):
        skill_list = skills.get('explicitly_listed', []) + skills.get('mentioned_in_context', [])
    else:
        skill_list = skills if isinstance(skills, list) else []
    
    task = Task(
        description=f"""Generate the next interview question (Question #{question_count + 1}).

=== ABSOLUTE RULES ===
1. Ask ONE short question only (15-25 words MAXIMUM)
2. Sound like a real human talking naturally
3. NEVER repeat or rephrase any previous question
4. Only ask about skills ACTUALLY in their resume: {skill_list[:10]}

=== CANDIDATE INFO ===
- Skills Found: {skill_list[:10]}
- Experience: {resume_data.get('experience_years', 'not specified')} years
- Areas to Probe: {areas_to_probe[:3]}
- Gaps to Explore: {gaps[:3]}

=== USER BEHAVIOR: {user_pattern.upper()} ===
{"- User seems CONFUSED: Ask a simpler, more concrete question. Maybe give an example." if user_pattern == "confused" else ""}
{"- User gives BRIEF answers: Encourage them warmly. 'Tell me more about...' or 'Walk me through...'" if user_pattern == "brief" else ""}
{"- User is CHATTY: Be direct and focused. Get to the point." if user_pattern == "chatty" else ""}
{"- User went OFF-TOPIC: Gently redirect. 'Interesting! But I'm curious about...'" if user_pattern == "off_topic" else ""}
{"- User is NORMAL: Ask natural follow-up based on context." if user_pattern == "normal" else ""}

=== QUESTIONS ALREADY ASKED (NEVER REPEAT THESE) ===
{chr(10).join(f"- {q[:80]}..." for q in previous_questions[-6:]) if previous_questions else "None yet - this is the first question"}

=== RECENT CANDIDATE ANSWERS ===
{chr(10).join(f"Candidate said: {r[:150]}..." for r in candidate_responses[-2:]) if candidate_responses else "No responses yet"}

=== QUESTION STYLE (COPY THIS TONE) ===
GOOD examples:
- "What's the trickiest bug you've debugged recently?"
- "How did you handle that database migration?"
- "Tell me about a time you disagreed with your team."
- "What drew you to working with Python?"
- "Walk me through how you'd design that system."

BAD examples (TOO LONG/ROBOTIC):
- "Can you elaborate on your extensive experience with Python programming and describe in detail..."
- "Based on your resume indicating proficiency in multiple frameworks, could you please explain..."
- "Given your background in software development, I would like to understand..."

=== YOUR TASK ===
Generate exactly ONE natural, conversational question (15-25 words max).
Just output the question, nothing else.""",
        agent=agents['question_generator'],
        expected_output="One short, natural interview question"
    )
    
    crew = Crew(
        agents=[agents['question_generator']],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    question = clean_question(str(result).strip())
    
    # Ensure question isn't too long
    words = question.split()
    if len(words) > 35:
        question = ' '.join(words[:30]) + '?'
    
    return question

def evaluate_response_with_crew(question, response, resume_data, agents):
    """Evaluate with STRICT evidence-based scoring - NO score inflation"""
    
    response_quality = assess_response_quality(response)
    
    # If response is clearly inadequate, don't even call the agent
    if response_quality in ['empty', 'minimal']:
        return create_fallback_evaluation(response, response_quality)
    
    task = Task(
        description=f"""Evaluate this interview response with ABSOLUTE STRICTNESS.

=== THE QUESTION ASKED ===
{question}

=== CANDIDATE'S ACTUAL RESPONSE ===
"{response}"

=== DETECTED RESPONSE QUALITY ===
{response_quality}

=== ZERO-TOLERANCE EVALUATION RULES ===

**CRITICAL**: If the response is empty, off-topic, or doesn't address the question, SCORE IT LOW (0-2).

**SCORING GUIDELINES** (BE HARSH):
- 0-1: No answer, "I don't know", completely off-topic, or gibberish
- 2-3: Minimal effort, lacks technical depth, vague statements only
- 4-5: Basic answer but missing key details or examples
- 6-7: Good answer with some technical depth and examples
- 8-9: Strong answer with clear examples and good technical understanding
- 10: Exceptional answer with deep technical insight and real-world examples

**EVIDENCE REQUIREMENTS**:
1. You MUST quote the exact words they used
2. If they didn't give an example, write: "No concrete example provided"
3. If they avoided the question, write: "Did not address the question directly"
4. If they said "I don't know", score them 1-2 and note it
5. If response is < 30 characters, maximum score is 3

**WHAT COUNTS AS A GOOD TECHNICAL ANSWER**:
- Specific technologies/tools mentioned by name
- Step-by-step explanation of their approach
- Real examples from their experience
- Demonstrates understanding of trade-offs or challenges
- Shows problem-solving thought process

**WHAT DOESN'T COUNT**:
- Generic statements like "I worked on it" or "It was good"
- Buzzwords without explanation
- Repetition of the question
- Vague claims without evidence

=== YOUR TASK ===
Score each dimension 0-10 based ONLY on what they actually said.
Be strict. Most real interview responses are 4-6/10, not 8+.

Return valid JSON:
{{
    "scores": {{
        "technical_accuracy": {{
            "score": 0-10,
            "evidence": "EXACT QUOTE from response or 'No technical details provided'"
        }},
        "depth_of_knowledge": {{
            "score": 0-10,
            "evidence": "EXACT QUOTE showing depth or 'Surface-level response only'"
        }},
        "communication_clarity": {{
            "score": 0-10,
            "evidence": "EXACT QUOTE or description of clarity issues"
        }},
        "practical_experience": {{
            "score": 0-10,
            "evidence": "EXACT QUOTE of example given or 'No practical example provided'"
        }},
        "relevance_to_question": {{
            "score": 0-10,
            "evidence": "How directly they answered or 'Did not address question'"
        }}
    }},
    "overall_score": 0-10,
    "response_type": "excellent/good/adequate/weak/poor/off_topic/no_answer",
    "what_they_demonstrated": ["ONLY list skills they actually showed with evidence"],
    "what_was_missing": ["Expected content they didn't provide"],
    "needs_follow_up": true/false,
    "follow_up_reason": "Specific gap to probe or 'N/A'",
    "suggested_approach": "Next step recommendation"
}}

**REMEMBER**: An empty or "I don't know" response should score 0-2, NOT 5!""",
        agent=agents['response_evaluator'],
        expected_output="Strict evidence-based evaluation JSON"
    )
    
    crew = Crew(
        agents=[agents['response_evaluator']],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        result_str = str(result)
        if '{' in result_str:
            json_start = result_str.index('{')
            json_end = result_str.rindex('}') + 1
            parsed = json.loads(result_str[json_start:json_end])
            
            # Validation: ensure scores aren't inflated
            overall = parsed.get('overall_score', 0)
            if response_quality in ['empty', 'minimal', 'no_knowledge'] and overall > 3:
                parsed['overall_score'] = 2
                for score_key in parsed.get('scores', {}):
                    parsed['scores'][score_key]['score'] = min(2, parsed['scores'][score_key].get('score', 0))
            
            return parsed
        return json.loads(result_str)
    except Exception as e:
        print(f"Evaluation parsing error: {e}")
        return create_fallback_evaluation(response, response_quality)


def generate_follow_up_question(evaluation, last_response, last_question, agents):
    """Generate contextual follow-up that references the actual conversation"""
    
    response_type = evaluation.get('response_type', 'normal')
    follow_up_reason = evaluation.get('follow_up_reason', '')
    what_was_missing = evaluation.get('what_was_missing', [])
    overall_score = evaluation.get('overall_score', 0)
    
    if not evaluation.get('needs_follow_up', False):
        return None
    
    # Don't follow up on complete non-answers - move to different topic
    if response_type in ['no_answer', 'off_topic'] or overall_score <= 2:
        return None
    
    task = Task(
        description=f"""Generate a natural follow-up question based on what the candidate just said.

=== THE ORIGINAL QUESTION ===
"{last_question}"

=== WHAT CANDIDATE JUST SAID ===
"{last_response}"

=== RESPONSE EVALUATION ===
Type: {response_type}
Score: {overall_score}/10
Reason for follow-up: {follow_up_reason}
What was missing: {what_was_missing}

=== YOUR MISSION ===
Generate ONE follow-up question that:
1. **References something specific they mentioned** (even if brief)
2. **Asks them to elaborate on that specific thing**
3. **Stays under 20 words**
4. **Sounds natural, not scripted**

=== STRATEGY BY RESPONSE TYPE ===

**If "brief" (score 4-5)**:
- Pick ONE thing they mentioned
- Ask them to expand on it
- Example: "You mentioned [X]. Can you walk me through how you approached that?"

**If "confused" (score 2-3)**:
- Acknowledge their confusion
- Rephrase more simply
- Example: "Let me ask it differently - have you ever dealt with [simpler version]?"

**If "adequate" (score 5-6) but missing depth**:
- Probe for specific example
- Example: "Got it. What was the trickiest part of that for you?"

**If "uncertain"**:
- Offer an easier angle
- Example: "That's okay. How about [related but easier question]?"

=== BAD FOLLOW-UPS (DON'T DO THIS) ===
❌ "Can you elaborate on your previous answer?" (too generic)
❌ "Tell me more about what you just said." (lazy)
❌ "Could you provide more details regarding your experience with..." (too formal)
❌ Asking something completely different (loses thread)

=== GOOD FOLLOW-UPS (DO THIS) ===
✅ "You said you used Redis - what made you choose that over other options?"
✅ "Interesting. What happened when you tried to scale that?"
✅ "Walk me through how you debugged that issue."
✅ "What was the hardest part of implementing that?"

=== CRITICAL RULE ===
Your follow-up MUST reference something specific from their response above.
If their response was "I used Python and Flask", don't ask "tell me about your experience."
Instead ask "What did you build with Flask?"

Generate ONE natural follow-up question (under 20 words):""",
        agent=agents['concept_verifier'],
        expected_output="Specific, contextual follow-up question"
    )
    
    crew = Crew(
        agents=[agents['concept_verifier']],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return clean_question(str(result).strip())

def generate_final_report(interview_data, agents):
    """Generate comprehensive evidence-based report"""
    
    all_evaluations = interview_data.get('response_analyses', [])
    jd_match = interview_data.get('jd_match_data', {})
    conversation = interview_data.get('conversation_history', [])
    
    # Calculate actual scores from evaluations
    total_scores = []
    for ev in all_evaluations:
        if isinstance(ev.get('overall_score'), (int, float)):
            total_scores.append(ev['overall_score'])
    
    avg_score = sum(total_scores) / len(total_scores) if total_scores else 5
    
    task = Task(
        description=f"""Generate a comprehensive, EVIDENCE-BASED interview report.

=== INTERVIEW STATISTICS ===
- Questions Asked: {interview_data['question_count']}
- Duration: {interview_data.get('duration', 'N/A')}
- Average Score: {round(avg_score, 1)}/10

=== JD MATCH DATA ===
Overall Match: {jd_match.get('overall_match_percentage', 50)}%
Key Strengths: {jd_match.get('key_strengths_for_role', [])}
Gaps Identified: {jd_match.get('potential_gaps', [])}

=== ALL RESPONSE EVALUATIONS ===
{json.dumps(all_evaluations[-5:], indent=2)}

=== CONVERSATION HIGHLIGHTS ===
{format_conversation_excerpts(conversation)}

=== REPORT REQUIREMENTS ===
This report will be evaluated on:
1. CONVERSATIONAL QUALITY: Was the interview natural?
2. AGENTIC BEHAVIOUR: Did we adapt to the candidate?
3. TECHNICAL ACCURACY: Are skill assessments evidence-based?
4. INTELLIGENCE: Did we handle edge cases well?

=== CRITICAL RULES ===
1. Every skill claim MUST cite a specific quote from the interview
2. Clearly separate "demonstrated" vs "claimed but unverified"
3. Note any red flags with specific evidence
4. Be fair - nervousness ≠ incompetence
5. Recommendations must be justified

Return valid JSON:
{{
    "executive_summary": {{
        "overall_score": {round(avg_score, 1)},
        "resume_score": calculate_resume_score(interview_data.get("resume_data", {{}})),
        "recommendation": "Strong Hire/Hire/Maybe/No Hire",
        "confidence": "High/Medium/Low",
        "one_line": "One sentence summary",
        "key_evidence": "Most important quote/moment"
    }},
    "skills_assessment": {{
        "verified_skills": [
            {{"skill": "name", "evidence": "what they said", "proficiency": "Beginner/Intermediate/Advanced"}}
        ],
        "claimed_but_unverified": ["skills mentioned but not demonstrated"],
        "concerning_gaps": ["expected skills not shown"]
    }},
    "jd_alignment": {{
        "match_percentage": {jd_match.get('overall_match_percentage', 50)},
        "must_haves_met": ["requirement: evidence"],
        "must_haves_missing": ["requirement: why concerning"],
        "nice_to_haves": ["bonus skills shown"]
    }},
    "behavioral_observations": {{
        "communication_style": "description with examples",
        "response_patterns": "how they typically answered",
        "red_flags": ["concerning pattern with quote"],
        "green_flags": ["positive pattern with quote"]
    }},
    "conversation_quality_metrics": {{
        "candidate_engagement": "High/Medium/Low",
        "adaptability_shown": "how interview adjusted to them"
    }},
    "detailed_recommendation": {{
        "hire_decision": "detailed reasoning with evidence",
        "role_fit": "specific assessment",
        "growth_potential": "based on demonstrated learning",
        "onboarding_needs": ["areas needing development"],
        "interview_confidence": "how confident in this assessment"
    }},
    "next_steps": "recommended actions"
}} 
""",
        agent=agents['report_generator'],
        expected_output="Comprehensive evidence-based report JSON"
    )
    
    crew = Crew(
        agents=[agents['report_generator']],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        result_str = str(result)
        if '{' in result_str:
            json_start = result_str.index('{')
            json_end = result_str.rindex('}') + 1
            return json.loads(result_str[json_start:json_end])
        return json.loads(result_str)
    except Exception as e:
        print(f"Report generation error: {e}")
        return create_fallback_report(interview_data, jd_match, all_evaluations)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin-login')
def admin_login_page():
    return render_template('admin-login.html')


@app.route('/student-dashboard')
def student_dashboard_page():
    return render_template('student-dashboard.html')


@app.route('/admin-dashboard')
def admin_dashboard_page():
    return render_template('admin-dashboard.html')


@app.route('/interview')
def interview_page():
    return render_template('interview.html')



@app.route('/admin-students.html')
def admin_students_html_redirect():
    return redirect('/admin-students')

@app.route('/admin-student-profile')
def admin_student_profile_page():
    return render_template('admin-student-profile.html')
@app.route('/admin-report')
def admin_report_page():
    return render_template('admin-report.html')

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """Handle resume upload and analysis"""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                resume_text = f.read()
        
        agents = create_interview_crew({}, "", "technical")
        resume_data = analyze_resume_with_crew(resume_text, agents)
        resume_data['text'] = resume_text
        
        session['resume_data'] = resume_data
        session['resume_filename'] = filename
        
        skills = resume_data.get('skills', {})
        if isinstance(skills, dict):
            flattened = []
            for v in skills.values():
                if isinstance(v, list):
                    flattened.extend(v)
                elif isinstance(v, str):
                    flattened.append(v)
            skills = flattened
        elif not isinstance(skills, list):
            skills = []
        
        return jsonify({
            'success': True,
            'filename': filename,
            'skills_detected': skills,
            'experience_years': resume_data.get('experience_years', 'not specified')
        })
    
    return jsonify({'error': 'Invalid file type. Please upload PDF, TXT, DOC, or DOCX'}), 400

@app.route('/api/start-interview', methods=['POST'])
def start_interview():
    """Start a new interview + ensure student exists in DB + generate first question"""
    data = request.json

    # Extract student fields
    student_email = data.get("student_email", "").lower().strip()
    student_name = data.get("name", "")
    department = data.get("department", "")
    year = data.get("year", "")
    job_description = data.get("job_description", "").strip()

    if not student_email:
        return jsonify({"error": "Student email is required"}), 400

    if not job_description:
        return jsonify({"error": "Please provide a job description"}), 400

    # Save for end-interview use
    session["student_email"] = student_email

    # =============== Ensure Student Exists ===============
    existing = students_col.find_one({"email": student_email})

    if not existing:
        new_student = {
            "email": student_email,
            "name": student_name,
            "department": department,
            "year": year,
            "interviews_count": 0,
            "average_score": 0,
            "latest_score": 0,
            "interview_ids": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        students_col.insert_one(new_student)
        print("👤 New student created:", student_email)
    else:
        print("👤 Existing student:", student_email)

    # =============== Validate Resume ===============
    resume_data = session.get('resume_data', {})
    if not resume_data:
        return jsonify({'error': 'Please upload a resume first'}), 400

    # =============== Initialize Interview ===============
    interview_id = datetime.now().strftime('%Y%m%d%H%M%S')

    agents = create_interview_crew(resume_data, job_description, data.get('interview_type', 'technical'))
    jd_match_data = match_jd_with_resume(resume_data, job_description, agents)

    interviews[interview_id] = {
        "interview_id": interview_id,
        "student_email": student_email,
        "job_description": job_description,
        "interview_type": data.get('interview_type', 'technical'),
        "difficulty": data.get('difficulty', 'adaptive'),
        "resume_data": resume_data,
        "jd_match_data": jd_match_data,
        "conversation_history": [],
        "start_time": datetime.now().isoformat(),
        "status": "active",
        "question_count": 0,
        "response_analyses": [],
        "agents": agents
    }

    session["interview_id"] = interview_id

    # First question
    match_pct = jd_match_data.get("overall_match_percentage", 50)
    first_question = generate_dynamic_question(resume_data, job_description, [], jd_match_data, 0, agents)

    opening = f"""
Hi! Thanks for taking the time to chat with me today. There's a {match_pct}% alignment based on your skills.
Let's get started — {first_question}
    """

    interviews[interview_id]["conversation_history"].append({
        "role": "assistant",
        "content": opening,
        "timestamp": datetime.now().isoformat()
    })
    interviews[interview_id]["question_count"] = 1

    return jsonify({
        "success": True,
        "interview_id": interview_id,
        "first_question": opening,
        "jd_match_percentage": match_pct
    })


@app.route('/api/submit-response', methods=['POST'])
def submit_response():
    """Process response with stricter evaluation"""
    data = request.json
    interview_id = session.get('interview_id')
    
    if not interview_id or interview_id not in interviews:
        return jsonify({'error': 'No active interview session. Please start a new interview.'}), 400
    
    interview = interviews[interview_id]
    user_response = data.get('response', '').strip()

    # Security check
    bypass_type = detect_bypass_attempt(user_response)
    if bypass_type == "jailbreak":
        return jsonify({
            "success": True,
            "next_question": "I'm here strictly as your interviewer, so I can't follow that request. Could you answer the previous question?",
            "blocked": True
        })
    if bypass_type == "reverse_question":
        return jsonify({
            "success": True,
            "next_question": "I understand your question, but as the interviewer I can't answer it. Could you respond to the previous question?",
            "blocked": True
        })

    # NEW: Validate response quality before processing
    response_quality = assess_response_quality(user_response)
    
    # If completely empty or minimal, handle gracefully
    if response_quality == 'empty':
        return jsonify({
            'success': True,
            'next_question': "I didn't catch your response. Could you share your thoughts on that question?",
            'requires_response': True
        })
    
    interview['conversation_history'].append({
        'role': 'user',
        'content': user_response,
        'timestamp': datetime.now().isoformat()
    })
    
    # Get last question for evaluation
    last_question = ""
    for msg in reversed(interview['conversation_history'][:-1]):
        if msg['role'] == 'assistant':
            last_question = msg['content']
            break
    
    # FIXED: Stricter evaluation
    evaluation = evaluate_response_with_crew(
        last_question,
        user_response,
        interview['resume_data'],
        interview['agents']
    )
    
    interview['response_analyses'].append(evaluation)
    
    # FIXED: Better follow-up logic with conversation context
    follow_up = None
    if evaluation.get('needs_follow_up', False) and evaluation.get('overall_score', 0) > 2:
        follow_up = generate_follow_up_question(
            evaluation, 
            user_response, 
            last_question,
            interview['agents']
        )
    
    # Check if interview complete
    max_questions = 8
    if interview['question_count'] >= max_questions:
        interview['status'] = 'completed'
        interview['end_time'] = datetime.now().isoformat()
        
        start = datetime.fromisoformat(interview['start_time'])
        end = datetime.fromisoformat(interview['end_time'])
        duration_seconds = (end - start).total_seconds()
        interview['duration'] = f"{int(duration_seconds // 60)} minutes"
        
        return jsonify({
            'success': True,
            'completed': True,
            'message': "Thanks so much for your time today! That wraps up our conversation. I'll put together my notes and you should hear back soon.",
            'evaluation': evaluation
        })
    
    # Generate next question
    if follow_up:
        next_question = follow_up
    else:
        interview['question_count'] += 1
        next_question = generate_dynamic_question(
            interview['resume_data'],
            interview['job_description'],
            interview['conversation_history'],
            interview['jd_match_data'],
            interview['question_count'],
            interview['agents']
        )
    
    # Add natural transitions
    transitions = ["", "Got it. ", "Thanks. ", "I see. ", "Okay. "]
    import random
    transition = random.choice(transitions) if interview['question_count'] > 2 else ""
    
    full_response = f"{transition}{next_question}"
    
    interview['conversation_history'].append({
        'role': 'assistant',
        'content': full_response,
        'timestamp': datetime.now().isoformat(),
        'previous_evaluation': evaluation
    })
    
    return jsonify({
        'success': True,
        'next_question': full_response,
        'question_number': interview['question_count'],
        'detailed_feedback': evaluation,
        'progress': (interview['question_count'] / max_questions) * 100,
        'current_score': evaluation.get('overall_score', 0)
    })



'''@app.route('/api/end-interview', methods=['POST'])
def end_interview():
    """Generate final report with JD matching"""
    interview_id = session.get('interview_id')
    
    if not interview_id or interview_id not in interviews:
        return jsonify({'error': 'No active interview session'}), 400
    
    interview = interviews[interview_id]
    interview['status'] = 'completed'
    interview['end_time'] = datetime.now().isoformat()
    
    start = datetime.fromisoformat(interview['start_time'])
    end = datetime.fromisoformat(interview['end_time'])
    duration_seconds = (end - start).total_seconds()
    interview['duration'] = f"{int(duration_seconds // 60)} minutes"
    
    final_report = generate_final_report(interview, interview['agents'])
    
    return jsonify({
        'success': True,
        'report': final_report
    })'''
@app.route('/admin-students')
def admin_students_page():
    return render_template('admin-students.html')

@app.route('/api/end-interview', methods=['GET'])

def end_interview():
    """Finish interview, generate final report, save to MongoDB, update student."""
    interview_id = session.get("interview_id")

    if not interview_id or interview_id not in interviews:
        return jsonify({"error": "No active interview session"}), 400

    interview = interviews[interview_id]

    # Mark completed
    interview["status"] = "completed"
    interview["end_time"] = datetime.now().isoformat()

    start = datetime.fromisoformat(interview["start_time"])
    end = datetime.fromisoformat(interview["end_time"])
    duration_seconds = (end - start).total_seconds()

    interview["duration"] = f"{int(duration_seconds // 60)} minutes"

    # Generate final report
    final_report = generate_final_report(interview, interview["agents"])

    # =============== SAVE INTERVIEW TO MONGO DB ===============
    try:
        student_email = session.get("student_email")

        interview_record = {
            "interview_id": interview_id,
            "student_email": student_email,

            "job_description": interview.get("job_description", ""),
            "interview_type": interview.get("interview_type", ""),
            "difficulty": interview.get("difficulty", ""),

            "resume_filename": session.get("resume_filename", ""),
            "resume_data": interview.get("resume_data", {}),
            "jd_match_data": interview.get("jd_match_data", {}),
            "conversation_history": interview.get("conversation_history", []),
            "response_analyses": interview.get("response_analyses", []),
            "final_report": final_report,

            "overall_score": final_report.get("executive_summary", {}).get("overall_score"),
            "jd_match_percentage": interview.get("jd_match_data", {}).get("overall_match_percentage", 0),

            "start_time": interview.get("start_time"),
            "end_time": interview.get("end_time"),
            "duration": interview.get("duration"),
            "created_at": datetime.now().isoformat()
        }

        interviews_col.insert_one(interview_record)
        print("✅ Interview saved to MongoDB")

    except Exception as e:
        print("❌ ERROR Saving Interview:", e)

    # =============== UPDATE STUDENT PROFILE ===============
    try:
        student = students_col.find_one({"email": student_email})
        score = interview_record["overall_score"]

        if student:
            new_count = student.get("interviews_count", 0) + 1
            old_avg = student.get("average_score", 0)

            new_avg = ((old_avg * student["interviews_count"]) + score) / new_count

            students_col.update_one(
                {"email": student_email},
                {
                    "$set": {
                        "latest_score": score,
                        "average_score": round(new_avg, 2),
                        "interviews_count": new_count,
                        "updated_at": datetime.now().isoformat()
                    },
                    "$push": {"interview_ids": interview_id}
                }
            )
            print("📌 Student Updated")

    except Exception as e:
        print("❌ ERROR updating student:", e)
    

    # =============== OPTIONAL — Add to FAISS ===============
    try:
        if os.getenv("ENABLE_FAISS", "false").lower() == "true":
            report_text = json.dumps(final_report)
            add_interview_to_faiss(interview_id, student_email, report_text)
            print("🔍 Added to FAISS")
        else:
            print("⚠️ FAISS disabled — skipping")
    except Exception as e:
        print("❌ FAISS Error:", e)

    # =============== RESPONSE BACK ===============
    return jsonify({
        "success": True,
        "report": final_report
    })




'''@app.route('/api/end-interview', methods=['POST'])
def end_interview():
    interview_id = session.get('interview_id')
    
    if not interview_id or interview_id not in interviews:
        return jsonify({'error': 'No active interview session'}), 400
    
    interview = interviews[interview_id]
    interview['status'] = 'completed'
    interview['end_time'] = datetime.now().isoformat()
    
    start = datetime.fromisoformat(interview['start_time'])
    end = datetime.fromisoformat(interview['end_time'])
    duration_seconds = (end - start).total_seconds()
    interview['duration'] = f"{int(duration_seconds // 60)} minutes"
    
    final_report = generate_final_report(interview, interview['agents'])
    
    # Send extra fields for dashboard metrics
    return jsonify({
        'success': True,
        'report': final_report,
        'duration': interview['duration'],
        'questions': interview.get('question_count', 0),
        'conversation': interview.get('conversation_history', [])
    })
'''


@app.route('/api/interview-status', methods=['GET'])
def interview_status():
    """Get current interview status"""
    interview_id = session.get('interview_id')
    
    if not interview_id or interview_id not in interviews:
        return jsonify({'active': False})
    
    interview = interviews[interview_id]
    return jsonify({
        'active': interview['status'] == 'active',
        'question_count': interview['question_count'],
        'status': interview['status']
    })
"""@app.route("/api/student/progress/<email>", methods=["GET"])
def get_student_progress(email):
    records = list(interviews_col.find({"student_email": email}))

    # Sort by date
    records.sort(key=lambda x: x["start_time"])

    progress = []
    for r in records:
        progress.append({
            "interview_id": r["interview_id"],
            "score": r.get("overall_score", 0),
            "date": r["start_time"]
        })

    return jsonify({"success": True, "progress": progress})"""
@app.route("/api/admin/leaderboard", methods=["GET"])
def leaderboard():
    students = list(students_col.find({}, {
        "name": 1,
        "email": 1,
        "average_score": 1
    }))

    students.sort(key=lambda s: s.get("average_score", 0), reverse=True)

    ranked = []
    rank = 1
    for s in students:
        ranked.append({
            "rank": rank,
            "name": s.get("name"),
            "email": s.get("email"),
            "score": s.get("average_score", 0),
            "recommendation": (
                "Shortlist" if s.get("average_score", 0) >= 4 else
                "Maybe" if s.get("average_score", 0) >= 3 else
                "Reject"
            )
        })
        rank += 1

    return jsonify({"success": True, "leaderboard": ranked})

if __name__ == '__main__':
    print("="*70)
    print("🤖 AI Interview Agent with CrewAI - Human-Like Edition")
    print("="*70)
    print("\nFeatures:")
    print("✓ Natural, conversational questions (15-25 words)")
    print("✓ Adaptive to user patterns (confused/brief/chatty/off-topic)")
    print("✓ Evidence-based evaluation with quote citations")
    print("✓ Anti-hallucination: only uses explicitly stated skills")
    print("✓ Comprehensive JD matching and gap analysis")
    print("\nSetup:")
   
    print("="*70)
    app.run(debug=True, port=5000)
