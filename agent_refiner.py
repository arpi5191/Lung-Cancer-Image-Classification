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
# llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"), temperature=0.0)
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"), temperature=0.0)


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
    medical_requirements: list[str]  # Structured clinical requirements generated to steer prompt optimization
    medical_details: list[str]  # Combined array of knowledge-synthesized and PubMed-retrieved visual features

    # ---------------- AUDIT FINDINGS ---------------- #
    present_medical_requirements: list[str]  # Successfully validated clinical requirements found in the prompt
    missing_medical_requirements: list[str]  # Unmet clinical requirements requiring generator attention
    present_medical_details: list[str]  # Validated PubMed-derived features present in the current prompt
    missing_medical_details: list[str]  # Omitted PubMed-derived features that should be integrated

    # ---------------- CURRENT WORKING PROMPT ---------------- #
    current_prompt: str  # actively evolving prompt used in current iteration
    current_token_count: float  # CLIP token length of current prompt
    current_score: float  # validator score assigned to current prompt

    # ---------------- BEST OVERALL PROMPT ---------------- #
    best_prompt: str  # highest-scoring prompt discovered so far
    best_token_count: float  # CLIP token count of best overall prompt
    best_score: float  # highest score achieved across all prompts

    # ---------------- BEST VALID PROMPT ---------------- #
    best_valid_prompt: str  # best prompt satisfying diffusion token constraint
    best_valid_token_count: float  # CLIP token count of best valid prompt
    best_valid_score: float  # highest score among token-valid prompts


# ---------------- MEDICAL PROMPT OPTIMIZATION NODE ---------------- #
def medical_prompt_optimization_node(state: AgentState):
        """
        Extracts explicit visual and structural clinical requirements from reference data.

        Acts as an Information Extraction Agent that parses dense biomedical literature
        and baseline user prompts. It isolates diagnostic criteria (e.g., tissue structures,
        pathological anomalies) and converts them into a clean JSON array of strings
        for downstream generation and evaluation layers.

        Args:
            state (AgentState): The current graph state containing 'current_prompt'
                                as well as any global clinical identifiers.

        Returns:
            dict: A state update dictionary containing 'medical_requirements', a clean
                  list of parsed raw string criteria extracted from the target text.
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


# ---------------- BIOLOGICAL AUDITOR NODE ---------------- #
def biological_auditor_node_knowledge(state: AgentState):
    """
    Synthesizes pre-trained biomedical context from internal LLM weights.

    Purpose:
    - Establish an initial baseline of visually concrete anatomical/pathological features.
    - Provide raw generation material to act as a fallback/comparison for literature searches.

    Args:
        state (AgentState): The current graph state containing 'anatomy',
                            'disease', and 'image_type'.

    Returns:
        dict: A state update dictionary containing 'medical_details', populated with
              a clean list of parsed, non-duplicate clinical term and description strings.
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


# ---------------- FILTER NODE ---------------- #
def filter_node(state: AgentState):
    """
    Audits the master medical details list against the current prompt draft.

    Iterates through each biomedical detail using a native Python loop to bypass
    LLM contextual laziness, evaluating presence on a single-item basis to filter
    out redundant or already-described terms.

    Args:
        state (AgentState): The current graph state containing 'medical_details'
                            and 'current_prompt'.

    Returns:
        dict: A state update dictionary containing 'medical_details', updated to
              include only the subset of detail strings that are completely
              absent or missing from the current prompt draft.
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


# ---------------- VALIDATION NODE ---------------- #
def validation_node(state: AgentState):
    """
    Performs a granular verification of the current prompt against clinical ground truth.

    Iterates through medical requirements and PubMed-derived details to determine
    their presence in the current prompt. Acts as an independent 'audit' layer
    to provide the validator node with categorized grounding data.

    Args:
        state (AgentState): The current graph state containing:
            - medical_requirements (list): List of required clinical features.
            - medical_details (list): List of PubMed-derived diagnostic context.
            - current_prompt (str): The active prompt to be audited.
            - current_token_count (float): Current token usage for the prompt.

    Returns:
        dict: A dictionary containing categorized audit results and updated scoring:
            - present_medical_requirements (list): Successfully included requirements.
            - missing_medical_requirements (list): Unmet clinical requirements.
            - present_medical_details (list): Successfully included PubMed details.
            - missing_medical_details (list): Omitted relevant clinical details.
            - current_score (float): Calculated quality score.
            - best_prompt (str): The best prompt found so far.
            - best_token_length (int): Token length of the best prompt.
            - best_score (float): The highest score achieved so far.
            - best_valid_prompt (str): The best prompt within token constraints.
            - best_valid_token_length (int): Token length of the best valid prompt.
            - best_valid_score (float): The highest score within token constraints.
    """

    # ---------------- INITIALIZE STATE DATA ---------------- #
    # Fetch ground truth requirements and PubMed details from the agent state
    medical_requirements = state.get("medical_requirements", [])
    medical_details = state.get("medical_details", [])

    # Group characteristics to enable logical iteration by category
    characteristics_dict = {
        "medical_requirements": medical_requirements,
        "medical_details": medical_details
    }

    # ---------------- TRACKING BUFFERS ---------------- #
    # Initialize lists to categorize presence/absence for the scoring logic
    present_medical_requirements = []
    missing_medical_requirements = []
    present_medical_details = []
    missing_medical_details = []

    # Defines the persona and objective for the LLM audit call
    system_prompt = (
        "You are a precise clinical data assistant. Your only job is to check if "
        "a single medical detail is already explicitly mentioned or described in a prompt."
    )

    # ---------------- VERIFICATION LOOP ---------------- #
    # Iterate through each clinical category and evaluate the prompt's adherence
    for category, characteristics in characteristics_dict.items():
        for characteristic in characteristics:

            # Construct the per-item evaluation prompt for the LLM audit
            user_prompt = f"""
            REFERENCE DATA:

            CHARACTERISTIC TO CHECK:
            {characteristic}

            PROMPT TO SCAN:
            {state.get('current_prompt', '')}

            INSTRUCTIONS:
            Scan the PROMPT TO SCAN to see if the CHARACTERISTIC TO CHECK keyword or concept
            is already written, mentioned, or explicitly included in the text.

            OUTPUT FORMAT:
            - If the concept or keyword is written or mentioned, return exactly: PRESENT
            - If the concept or keyword is completely absent, return exactly: MISSING

            Do not include any conversational text, JSON structures, or explanations.
            Just return the single word.
            """

            # Execute the audit call to the LLM and capture the presence status
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            # Normalize the result to ensure consistent boolean string comparison
            result = response.content.strip().upper()

            # Map the audit result to the corresponding tracking buffer
            if "PRESENT" in result:
                if category == "medical_requirements":
                    present_medical_requirements.append(characteristic)
                else:
                    present_medical_details.append(characteristic)
            else:
                if category == "medical_requirements":
                    missing_medical_requirements.append(characteristic)
                else:
                    missing_medical_details.append(characteristic)

    # ---------------- COUNTS AND SCORING ---------------- #
    # Calculate the cardinality of requirements and details for quantitative assessment
    present_medical_requirements_count = float(len(present_medical_requirements))
    missing_medical_requirements_count = float(len(missing_medical_requirements))
    present_medical_details_count = float(len(present_medical_details))
    missing_medical_details_count = float(len(missing_medical_details))

    # Log the counts for debugging to ensure the math logic is receiving correct inputs
    print(f"Present Requirements Count: {present_medical_requirements_count}")
    print(f"Missing Requirements Count: {missing_medical_requirements_count}")
    print(f"Present Details Count: {present_medical_details_count}")
    print(f"Missing Details Count: {missing_medical_details_count}")

    # ---------------- SCORING CALCULATION ---------------- #
    # Calculate weighted metrics for requirements, details, and token constraints
    req_score = (present_medical_requirements_count /
                 (present_medical_requirements_count + missing_medical_requirements_count)) * 4.0
    det_score = (present_medical_details_count /
                 (present_medical_details_count + missing_medical_details_count)) * 3.0
    token_bonus = 5.0 if state.get('current_token_count', 0.0) <= 77 else 0.0
    current_score = req_score + det_score + token_bonus

    print(f"Req Score: {req_score}")
    print(f"Det Score: {det_score}")
    print(f"Token Bonus: {token_bonus}")
    print(f"Current Total Score: {current_score}")

    # ---------------- STATE PRESERVATION ---------------- #
    # Compile current audit findings and carry forward historical 'best' records for state persistence
    updates = {
        "present_medical_requirements": present_medical_requirements,
        "missing_medical_requirements": missing_medical_requirements,
        "present_medical_details": present_medical_details,
        "missing_medical_details": missing_medical_details,
        "current_score": current_score,
        "best_prompt": state.get("best_prompt", ""),
        "best_token_count": state.get("best_token_count", 0),
        "best_score": state.get("best_score", 0.0),
        "best_valid_prompt": state.get("best_valid_prompt", ""),
        "best_valid_token_count": state.get("best_valid_token_count", 0),
        "best_valid_score": state.get("best_valid_score", 0.0)
    }

    # ---------------- BEST OVERALL PROMPT TRACKING ---------------- #
    # Update global record if the current iteration achieves a higher total score
    if current_score > updates["best_score"]:
        updates["best_prompt"] = state["current_prompt"]
        updates["best_score"] = current_score
        updates["best_token_count"] = state["current_token_count"]

    # ---------------- BEST VALID PROMPT TRACKING ---------------- #
    # Update constraint-satisfied record if the prompt meets token limits and scores higher
    if state["current_token_count"] <= 77:
        if current_score > updates["best_valid_score"]:
            updates["best_valid_prompt"] = state["current_prompt"]
            updates["best_valid_score"] = current_score
            updates["best_valid_token_count"] = state["current_token_count"]

    # ---------------- STATE UPDATE ---------------- #
    # Return the aggregated audit results and updated historical markers to the graph
    return updates


# ---------------- VALIDATION NODE ---------------- #
def generator_node(state: AgentState):
    """
    Executes a conditional prompt revision loop based on token constraints.

    This function analyzes the current prompt's list structure against a hard
    token limit (77). Depending on whether the count exceeds the limit, it
    instructs the LLM to either prune a semantically similar characteristic
    or integrate a new one, ensuring that the resulting comma-separated list
    maintains syntactic integrity.

    Args:
        state: The current agent state containing the prompt, token metrics,
               and lists of medical requirements/details.

    Returns:
        dict: An updated state containing the revised prompt, new token count,
              and incremented iteration counter.
    """

    # ---------------- INITIALIZE STATE DATA ---------------- #
    # Retrieve current progress from the agent state to compute the performance metrics.
    present_medical_requirements = state.get("present_medical_requirements", [])
    missing_medical_requirements = state.get("missing_medical_requirements", [])
    present_medical_details = state.get("present_medical_details", [])
    missing_medical_details = state.get("missing_medical_details", [])

    # ---------------- SELECT CHARACTERISTIC TO ADD ---------------- #
    # Prioritize missing requirements over details to ensure baseline
    # clinical criteria are met before refining specific diagnostic details.
    characteristic_to_add = None

    if missing_medical_requirements:
        # If requirements are missing, select the first requirement to address.
        characteristic_to_add = missing_medical_requirements[0]
    elif missing_medical_details:
        # If all requirements are met but specific details are still missing,
        # select the first detail to enrich the prompt.
        characteristic_to_add = missing_medical_details[0]

    # ---------------- SELECT CHARACTERISTIC TO REMOVE ---------------- #
    # If the token limit is exceeded, identify a low-priority characteristic
    # currently in the prompt to remove, starting with clinical details
    # before sacrificing core medical requirements.
    characteristic_to_remove = None

    if present_medical_details:
        # If details are present, remove the first one to reduce token usage
        # while preserving essential clinical requirements.
        characteristic_to_remove = present_medical_details[0]

    elif present_medical_requirements:
        # If no optional details are present, begin removing core requirements
        # to ensure the prompt stays within the hard token limit.
        characteristic_to_remove = present_medical_requirements[0]

    # ---------------- SYSTEM ROLE DEFINITION ---------------- #
    # Define the agent's persona and objective as a specialist
    # focusing on descriptive accuracy within strict token constraints.
    system_prompt = (
        "You are an expert clinical imaging specialist. Your objective is to "
        "refine and revise prompts to be highly descriptive and accurate, "
        "while strictly adhering to the specified token limit."
    )

    # ---------------- USER ROLE DEFINITION ---------------- #
    # Provide the revision agent with the list state and edit instructions.
    # The agent's focus is on maintaining valid list syntax (commas/conjunctions)
    # when adding or removing elements to meet the 77-token constraint.
    user_prompt = f"""
    REFERENCE DATA:

    PROMPT TO REVISE:
    {state.get('best_prompt', '')}

    CHARACTERISTIC TO ADD:
    {characteristic_to_add}

    CHARACTERISTIC TO REMOVE:
    {characteristic_to_remove}

    TOKEN LENGTH:
    {state['best_token_count']}

    TOKEN LIMIT:
    77

    INSTRUCTIONS:
    - IF TOKEN LENGTH: > TOKEN LIMIT:
        1) Locate the item in the list semantically similar to '{characteristic_to_remove}'.
        2) Remove it.
        3) Reformat the list: ensure there are no trailing commas, double commas, or orphaned conjunctions (like an
          'and' left without a partner).

    - IF TOKEN LENGTH <= TOKEN LIMIT:
        1) Insert a reworded version of '{characteristic_to_add}' into the list, ensuring it matches the tone of the
           existing items.
        2) Ensure it is preceded by a comma if necessary.
        3) Ensure the final list has proper comma placement and no duplicate conjunctions.

    OUTPUT FORMAT:
    Return ONLY the final revised prompt.
    Do not include explanations, labels, units, or any additional text.
    """

    # Invoke the language model to perform the structured quality assessment
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    # Clean the response
    current_prompt = response.content.strip()

    # Calculate the token length using the CLIP tokenizer
    current_token_count = len(tokenizer.encode(current_prompt))

    # Increment the iteration counter to track progress through the revision loop
    iterations = state.get("iterations") + 1

    # ---------------- RETURN STATE UPDATE ---------------- #
    return {
        "current_prompt": current_prompt,
        "current_token_count": current_token_count,
        "Iterations": iterations
    }
