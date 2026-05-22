# ---------------- GENERAL IMPORTS ---------------- #
import os
import json
from typing import TypedDict
from transformers import CLIPTokenizer

# ---------------- LANGCHAIN IMPORTS ---------------- #
# ChatGroq: LangChain wrapper around Groq-hosted LLMs
from langchain_groq import ChatGroq

# PubMed tool: allows retrieval of biomedical literature/context
from langchain_community.tools.pubmed.tool import PubmedQueryRun

# Message abstractions used by LangChain chat models
from langchain_core.messages import SystemMessage, HumanMessage

# LangGraph workflow engine
from langgraph.graph import StateGraph, END


# ---------------- TOKENIZER ---------------- #
# CLIP tokenizer used to enforce the diffusion-model token constraint.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


# ---------------- PUBMED TOOL ---------------- #
# PubMed search utility used for retrieving real biomedical context.
pubmed = PubmedQueryRun()


# ---------------- LANGCHAIN GROQ CLIENT ---------------- #
# Primary LLM used throughout the workflow.
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"), temperature=0.0)


# ---------------- STATE ---------------- #
class AgentState(TypedDict):
    """
    Shared state propagated across LangGraph nodes.

    The workflow follows this general pattern:
        biological auditor → validate → generate → validate → ...

    The state stores:
    - current working prompt
    - best prompts discovered so far
    - token statistics
    - validator feedback
    - medical metadata/context
    """

    # ---------------- LOOP CONTROL ---------------- #
    iterations: int  # number of optimization/refinement cycles completed

    # ---------------- MEDICAL CONTEXT ---------------- #
    anatomy: str  # target anatomical structure or organ (e.g. lung, brain)
    disease: str  # target pathology or disease being represented
    image_type: str  # medical imaging modality/domain (CT, MRI, Histopathology, etc.)

    # ---------------- LITERATURE GROUNDING---------------- #
    errors: list[str]  # structured mappings of baseline medical prompt errors
    medical_requirements: str
    medical_details: str  # PubMed-derived biomedical context used for grounding

    # ---------------- VALIDATOR OUTPUT ---------------- #
    feedback: str  # validator-generated critique and improvement suggestions

    # ---------------- CURRENT WORKING PROMPT ---------------- #
    current_prompt: str  # actively evolving prompt used in current iteration
    current_token_count: int  # CLIP token length of current prompt
    current_score: float  # validator score assigned to current prompt

    # ---------------- BEST OVERALL PROMPT ---------------- #
    best_prompt: str  # highest-scoring prompt discovered so far
    best_token_count: int  # CLIP token count of best overall prompt
    best_score: float  # highest score achieved across all prompts

    # ---------------- BEST VALID PROMPT ---------------- #
    best_valid_prompt: str  # best prompt satisfying diffusion token constraint
    best_valid_token_count: int  # CLIP token count of best valid prompt
    best_valid_score: float  # highest score among token-valid prompts

# # ---------------- BIOLOGICAL AUDITOR NODE ---------------- #
def biological_audior_node(state: AgentState):
    """
        Retrieves biomedical context from PubMed, synthesizes visual features,
        and updates the state with structured clinical descriptions.

        Purpose:
        - Ground the system in verified medical literature.
        - Improve anatomical and pathological accuracy in generations.
        - Minimize prompt hallucination through evidence-based conditioning.
    """

    # # ---------------- PUBMED QUERY ---------------- #
    # # Formulate a targeted search string using state variables for disease, anatomy, and imaging modality
    # query = f"{state['disease']} appearance in {state['anatomy']} {state['image_type']}"
    #
    # # ---------------- PUBMED SEARCH ---------------- #
    # # Execute retrieval and sanitize results for LLM processing
    # search_results = pubmed.invoke(query)
    # cleaned_results = search_results.strip() if isinstance(search_results, str) else ""

    # ---------------- SUMMARIZATION PROMPT ---------------- #
    # Direct the LLM to filter literature for visually concrete, diffusion-model compatible features
    summary_prompt = f"""
    REFERENCE DATA:

    QUERY CONTEXT:
    Anatomy: {state['anatomy']}
    Disease: {state['disease']}
    Image Type: {state['image_type']}


    INSTRUCTIONS:

    Summarize the key clinically relevant imaging features from the PubMed results relevant to the query. Extract
    only visually observable and medically relevant descriptors that would help a diffusion model generate clinically
    realistic medical images.


    OUTPUT FORMAT:

    You must use the following format for every item:
    **Term**: characteristic description,

    Provide ONLY the JSON list. Do not include any introductory text, concluding remarks, explanations, or
    conversational filler.
    """

    # ---------------- LLM SUMMARIZATION ---------------- #
    # Generate structured, bulleted feature list based on PubMed evidence
    summary = llm.invoke([HumanMessage(content=summary_prompt)])

   # Split raw string by newlines, clean up markdown markers/whitespace, and extract lines
   # containing ':' to build a clean list of full terms and definitions for loop iteration.
    parsed_list = [line.replace("**", "").strip(" ,[]\t\r") for line in summary.content.split("\n") if ":" in line]

    # ---------------- STATE UPDATE ---------------- #
    # Return the distilled clinical context to the state machine
    return {
        "medical_details": parsed_list
    }


# ---------------- FILTER NODE ---------------- #
def filter_node(state: AgentState):
    """
    Audits the master medical details list against the current prompt draft.

    Iterates through each biomedical detail using a native Python loop to bypass
    LLM contextual laziness, evaluating presence on a single-item basis to filter
    out redundant or already-described terms.
    """

    # Extract the current details list and fallback to an empty array if missing
    medical_details = state.get("medical_details", [])

    # Tracking list for terms that are NOT yet included in the user's prompt
    missing_details = []

    # ---------------- SYSTEM PROMPT CONFIGURATION ---------------- #
    # Keep strings clean and left-aligned to prevent trailing whitespace tokens
    system_prompt = (
        "You are a precise clinical data assistant. Your only job is to check if "
        "a single medical detail is already explicitly mentioned or described in a prompt."
    )

    # Execute a deterministic serial loop in Python to guarantee 100% list coverage
    for detail in medical_details:

        # ---------------- USER PROMPT CONFIGURATION ---------------- #
        # FIX: Wrapped {detail} in curly braces so the actual data string resolves
        user_prompt = f"""
        REFERENCE DATA:

        DETAIL TO CHECK:
        {detail}

        PROMPT TO SCAN:
        {state.get('current_prompt', '')}


        INSTRUCTIONS:
        Scan the PROMPT TO SCAN to see if the DETAIL TO CHECK keyword or concept is already written, mentioned, or explicitly included in the text.

        OUTPUT FORMAT:
        - If the concept or keyword is written or mentioned, return exactly: PRESENT
        - If the concept or keyword is completely absent, return exactly: MISSING

        Do not include any conversational text, JSON structures, or explanations. Just return the single word.
        """

        # Invoke the language model for a single-element evaluation pass
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # Normalize the string response to ensure bulletproof logic comparison
        result = response.content.strip().upper()

        # If the detail is missing from the prompt, retain it for the downstream generation nodes
        if "MISSING" in result:
            missing_details.append(detail)

    # ---------------- STATE UPDATE ---------------- #
    # Update the graph state with the trimmed, non-redundant details array
    return {
        "medical_details": missing_details
    }
