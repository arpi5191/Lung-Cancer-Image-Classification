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
# llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"), temperature=0.0)


# ---------------- STATE ---------------- #
class AgentState(TypedDict):
    """
    Shared state propagated across LangGraph nodes.

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
    medical_requirements: list[str]  # Structured clinical requirements generated to steer prompt optimization
    medical_details: list[str]  # Combined array of knowledge-synthesized and PubMed-retrieved visual features

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


# ---------------- BIOLOGICAL AUDITOR NODE (PHASE 1) ---------------- #
def biological_auditor_node_knowledge(state: AgentState):
    """
    Synthesizes pre-trained biomedical context from internal LLM weights.

    Purpose:
    - Establish an initial baseline of visually concrete anatomical/pathological features.
    - Provide raw generation material to act as a fallback/comparison for literature searches.
    """

    # ---------------- SUMMARIZATION PROMPT ---------------- #
    # Direct the LLM to synthesize pre-trained knowledge for visually concrete, diffusion-model compatible features
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

    Ensure no duplicate descriptors are stored.

    Provide ONLY the raw text list items matching the format above. Do not include any introductory text, concluding
    remarks, explanations, or conversational filler.
    """

    # ---------------- LLM SUMMARIZATION ---------------- #
    # Generate structured, bulleted feature list based on internal weights
    summary = llm.invoke([HumanMessage(content=summary_prompt)])

    # Split raw string by newlines, clean up markdown markers/whitespace, and extract lines
    # containing ':' to build a clean list of full terms and definitions for loop iteration.
    parsed_list = [line.replace("**", "").strip(" ,[]\t\r") for line in summary.content.split("\n") if ":" in line]

    # ---------------- STATE UPDATE ---------------- #
    # Return the distilled clinical context to the state machine
    return {
        "medical_details": parsed_list
    }


# # ---------------- BIOLOGICAL AUDITOR NODE (PHASE 2) ---------------- #
# def biological_auditor_node_pubmed(state: AgentState):
#     """
#     Retrieves dynamic biomedical context from live PubMed literature searches.
#
#     Purpose:
#     - Ground the system in verified, peer-reviewed medical literature.
#     - Enhance accuracy and supplement pre-existing baseline knowledge with specific descriptors.
#     """
#
#     # Extract the current details list from Phase 1 and fallback to an empty array if missing
#     medical_details = state.get("medical_details", [])
#
#     # ---------------- PUBMED QUERY ---------------- #
#     # Formulate a targeted search string using state variables for disease, anatomy, and imaging modality
#     query = f"{state['disease']} appearance in {state['anatomy']} {state['image_type']}"
#
#     # ---------------- PUBMED SEARCH ---------------- #
#     # Execute retrieval and sanitize results for LLM processing
#     search_results = pubmed.invoke(query)
#     # FIX: Uncommented to define cleaned_results for prompt template execution
#     cleaned_results = search_results.strip() if isinstance(search_results, str) else ""
#
#     # ---------------- SUMMARIZATION PROMPT ---------------- #
#     # Direct the LLM to filter literature for visually concrete, diffusion-model compatible features
#     summary_prompt = f"""
#     REFERENCE DATA:
#
#     LITERATURE SEARCH RESULTS FROM PUBMED:
#     {cleaned_results}
#
#     QUERY CONTEXT:
#     Anatomy: {state['anatomy']}
#     Disease: {state['disease']}
#     Image Type: {state['image_type']}
#
#
#     INSTRUCTIONS:
#
#     Summarize the key clinically relevant imaging features from LITERATURE SEARCH RESULTS FROM PUBMED relevant to the
#     query. Extract only visually observable and medically relevant descriptors that would help a diffusion model
#     generate clinically realistic medical images.
#
#
#     OUTPUT FORMAT:
#
#     You must use the following format for every item:
#     **Term**: characteristic description,
#
#     Provide ONLY the raw text list items matching the format above. Do not include any introductory text, concluding
#     remarks, explanations, or conversational filler.
#     """
#
#     # ---------------- LLM SUMMARIZATION ---------------- #
#     # Generate structured, bulleted feature list based on PubMed evidence
#     summary = llm.invoke([HumanMessage(content=summary_prompt)])
#
#     # Split raw string by newlines, clean up markdown markers/whitespace, and extract lines
#     # containing ':' to build a clean list of full terms and definitions for loop iteration.
#     parsed_list = [line.replace("**", "").strip(" ,[]\t\r") for line in summary.content.split("\n") if ":" in line]
#
#     # ---------------- STATE UPDATE ---------------- #
#     # FIX: Combined current knowledge list (medical_details) with fresh PubMed items (parsed_list)
#     return {
#         "medical_details": medical_details + parsed_list
#     }


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
        user_prompt = f"""
        REFERENCE DATA:

        DETAIL TO CHECK:
        {detail}

        PROMPT TO SCAN:
        {state.get('current_prompt', '')}


        INSTRUCTIONS:
        Scan the PROMPT TO SCAN to see if the DETAIL TO CHECK keyword or concept is already written, mentioned, or
        explicitly included in the text.

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


# ---------------- CLINICAL PRIORITIZATION NODE ---------------- #
def clinical_prioritization_node(state: AgentState):
    """
    Ranks extracted medical details based on their visual salience for image generation.

    This node processes a consolidated array of medical descriptors by executing a
    deterministic serial evaluation loop. Each feature is contextually scored by an LLM
    acting as a clinical imaging specialist. Features are tagged based on relevance
    thresholds and sorted in descending order to guarantee that downstream prompt optimization
    prioritizes the most pathognomonic and visually critical radiological elements.
    """

    # Extract the current details list from state, fallback to an empty array if missing
    medical_details = state.get("medical_details", [])

    # Map each medical term to its calculated visual priority score
    detail_scores = dict()

    # ---------------- SYSTEM PROMPT CONFIGURATION ---------------- #
    # Configure the structural persona; keep text clean to avoid trailing token padding
    system_prompt = (
        "You are a clinical imaging specialist. Your only job is to provide a ranking on how relevant the detail "
        "is for medical image generation."
    )

    # Execute a deterministic serial loop in Python to guarantee 100% list coverage
    for detail in medical_details:

        # ---------------- USER PROMPT CONFIGURATION ---------------- #
        # Format a highly specific evaluation prompt context for the LLM
        user_prompt = f"""
        REFERENCE DATA:
        Anatomy: {state['anatomy']}
        Disease: {state['disease']}
        Image Type: {state['image_type']}

        DETAIL TO EVALUATE:
        "{detail}"

        INSTRUCTIONS:
        Rate how important this feature is for a diffusion model to generate a realistic medical image.
        You must classify this feature into exactly one of the four categories below. Be highly critical.

        [4] = CORE SHAPE & LAYER STRUCTURE
        - Mandatory macro-geometry.
        - Without this, the structural backbone of the image is wrong.

        [3] = KEY VISUAL HALLMARKS
        - High-yield diagnostic features.
        - Highly prominent and visually striking, but sits inside the primary macro-geometry.

        [2] = BACKGROUND MICRO-FEATURES
        - Micro-level details.
        - Visually present, but does not dictate the major regional layout of the target image.

        [1] = GENERIC / VISUALLY NEGLIGIBLE
        - Vague or common details.
        - Extremely non-specific, faint, or has low visual distinction on an image scan.

        OUTPUT FORMAT:
        Return exactly a single integer: 1, 2, 3, or 4.
        Do not include any conversational text, decimals, letters, punctuation, or markdown code fences.
        """

        # Invoke the language model for a single-element scoring pass
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # Clean up output string text before attempting numeric translation
        score_str = response.content.strip()

        # CRITICAL FIX: Safe float casting to protect against arbitrary LLM string returns
        try:
            score_val = float(score_str)
        except ValueError:
            score_val = 0.0  # Safe fallback default value for non-numeric parser failures

        # Append visual categorization metadata based on the numerical cutoff threshold
        if score_val >= 3.0:
            final_detail = f"{detail} [HIGH RELEVANCE]"
        else:
            final_detail = f"{detail} [LOW RELEVANCE]"

        # Populate the mapping structure using the clean string as the access key
        detail_scores[final_detail] = score_val

        # # Iteration Diagnositics
        # print(f"\n[Evaluating Detail]: {detail}")
        # print(f" -> Assigned Score : {score_val:.1f}/10.0")
        # print(f" -> Tagged Output  : {final_detail}")
        # print("-" * 50)

    # Sort the dictionary by scores in descending order
    sorted_detail_scores = dict(sorted(detail_scores.items(), key=lambda item: item[1], reverse=True))

    # Extract just the sorted string keys to overwrite the state's requirement list
    ranked_medical_details = list(sorted_detail_scores.keys())

    # ---------------- STATE UPDATE ---------------- #
    # Return the prioritized array back to the graph state
    return {
        "medical_details": ranked_medical_details
    }


# ---------------- MEDICAL PROMPT OPTIMIZATION NODE ---------------- #
def medical_prompt_optimization_node(state: AgentState):
        """
        Extracts explicit visual and structural clinical requirements from reference data.

        Acts as an Information Extraction Agent that parses dense biomedical literature
        and baseline user prompts. It isolates diagnostic criteria (e.g., tissue structures,
        pathological anomalies) and converts them into a clean JSON array of strings
        for downstream generation and evaluation layers.
        """

        # ---------------- SYSTEM PROMPT CONFIGURATION ---------------- #
        # Configures a triple-quoted f-string to inject the current state data.
        # Explicitly instructs the model to return a raw line-by-line list
        # without conversational filler or markdown bolding tags.
        system_prompt = f"""
            REFERENCE DATA:

            PROMPT TO PARSE:
            {state['current_prompt']}


            INSTRUCTIONS:

            Scan the PROMPT TO PARSE to extract specific requirements that the diffusion model must express to
            generate accurate medical images.

            OUTPUT FORMAT:

            You must use the following format for every item:
            requirement,

            Provide ONLY the raw text list items matching the format above. Do not include any introductory text,
            concluding remarks, explanations, or conversational filler.
        """

        # ---------------- LLM SUMMARIZATION ---------------- #
        # Invokes the model to extract structural and visual medical features.
        summary = llm.invoke([HumanMessage(content=system_prompt)])

        # ---------------- CONVERT TO LIST OF STRINGS ---------------- #
        # Split the string line-by-line using splitlines() to avoid backslash
        # syntax issues inside enclosing f-strings or complex execution environments.
        raw_lines = summary.content.splitlines()
        parsed_list = []

        for line in raw_lines:
            # Strip markdown artifacts, formatting quotes, and trailing commas.
            # Casing is left natural to preserve clinical acronyms (e.g., N/C ratio)
            # since diffusion text encoders (CLIP/T5) are fundamentally case-insensitive.
            if line:
                parsed_list.append(line)

        # ---------------- STATE UPDATE ---------------- #
        # Pass the isolated clinical criteria array back to the state machine.
        return {
            "medical_requirements": parsed_list
        }
