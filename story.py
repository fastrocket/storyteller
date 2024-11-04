import json
from datetime import datetime
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import sys


load_dotenv()

USE_OPENAI=False

MAX_RETRIES = 20


# FIRST_MODEL="smollm2:1.7b"
# FIRST_MODEL="granite3-dense"
FIRST_MODEL="qwen2.5-coder:1.5b"
# FIRST_MODEL="granite3-moe:latest"
# FIRST_MODEL="dolphin-mistral"
FIRST_TEMP=0.7
SECOND_MODEL=FIRST_MODEL
SECOND_TEMP=0.9

# Near the top where other models are defined
JSON_MODEL = "qwen2.5-coder:1.5b"  # or another model that's good with structured output
SUMMARY_MODEL="qwen2.5-coder:1.5b"  # Added this constant
# SUMMARY_MODEL="smollm2:360m"  # Added this constant

# Near the top with other constants
NUM_CHAPTERS = 10  # Set number of chapters here

if USE_OPENAI:
    llm = ChatOpenAI(
        temperature=.5,
        model_name='gpt-4'
    )

    # import openai 

    # client = openai.OpenAI( base_url="https://localhost:44333/v1", api_key = "no-key-required" ) 
    
else:

    # Initialize Ollama - only create new instances for different models/settings
    base_llm = Ollama(
        repeat_penalty=1.4,
        num_predict=8000,
        model=FIRST_MODEL,
        temperature=FIRST_TEMP,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    
    # Reuse base_llm if models are the same
    llm = base_llm
    llmJson = base_llm if JSON_MODEL == FIRST_MODEL else Ollama(
        repeat_penalty=1.4,
        num_predict=8000,
        model=JSON_MODEL,
        temperature=0.1,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    llmSummary = base_llm if SUMMARY_MODEL == FIRST_MODEL else Ollama(
        repeat_penalty=1.4,
        num_predict=2000,
        model=SUMMARY_MODEL,
        temperature=0.3,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

# Get the current datetime once for the session
session_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def llm_log(prompt):

    # chat(messages)
    if USE_OPENAI:
        messages = [
            SystemMessage(
                content="You are the world's best AI novelist beating out Stephen King on creativity."
            ),
            HumanMessage(
                content=prompt
            ),
        ]
        response = llm(messages).content
    else:
        response = llm(prompt) # ollama

    session_filename = f"session-{session_datetime}.txt"
    with open(session_filename, "a", encoding="utf-8") as session_file:
        session_file.write(f"\n{'='*20} PROMPT {'='*20}\n")
        session_file.write(prompt)
        session_file.write(f"\n{'='*20} RESPONSE {'='*20}\n")
        session_file.write(response)
    return response

def strip_text_around_json(text):
    # Find the first occurrence of '{' character
    start_index = text.find('{')

    # Find the last occurrence of the '}' character
    end_index = text.rfind('}') + 1  # +1 to include the '}' character

    # Slice the string from start_index to end_index
    stripped_text = text[start_index:end_index]

    return stripped_text


def extract_and_parse_json(string_data):
    stripped_text = strip_text_around_json(string_data)
    parsed_data = json.loads(stripped_text)
    
    # Confirm that there are NUM_CHAPTERS chapters with non-empty 'title' and 'prompt' fields
    chapters = parsed_data.get('chapters', [])
    if len(chapters) < NUM_CHAPTERS:
        raise Exception(f"Insufficient number of chapters. Expected {NUM_CHAPTERS}, got {len(chapters)}.")
    
    # Ensure 'title' and 'prompt' fields in each chapter are strings
    for chapter in chapters:
        if not isinstance(chapter.get('title'), str):
            raise ValueError(f"'title' in chapter {chapter.get('chapter')} is not a string.")
        
        if not isinstance(chapter.get('prompt'), str):
            raise ValueError(f"'prompt' in chapter {chapter.get('chapter')} is not a string or is missing.")
 

    return parsed_data




# Constants and templates
PLOT = """Create a unique, compelling story following these guidelines:

1. SETTING & GENRE: Choose ONE:
- Modern psychological horror
- Dark science fiction
- Supernatural thriller
- Surreal contemporary drama

2. STRUCTURE:
- Strong hook in the opening
- Clear 3-act structure (setup, confrontation, resolution)
- Rising tension throughout
- Impactful climax
- Satisfying yet thought-provoking ending

3. CHARACTERS:
- 2-3 main characters with clear motivations
- Introduce them naturally through actions and dialogue
- Give them distinct voices and behaviors
- Create meaningful character arcs

4. WRITING STYLE:
- Active voice only
- Show through actions, dialogue, and sensory details
- No exposition dumps
- Tight pacing
- Vivid, specific descriptions
- Natural dialogue that reveals character

5. TONE:
- Dark, tense atmosphere
- Building sense of dread
- Psychological depth
- Grounded despite supernatural/sci-fi elements

Craft a ONE PARAGRAPH synopsis that captures these elements while leaving room for creative expansion.
"""

# PLOT = """A dark scifi tale of a grizzled solo astronaut, Bob, encountering a derelict alien ship in deep space. Aliens arrive and destroy his ship while he is exploring the derelict. He must use cunning to restore the engines on the derelict ship while playing dead all to escape."""
# PLOT = """An atheist mom, Gena, and her loving young daughter, Megan, have a wonderful life that is turned upside
# when an evil presence slowly possesses Megan that causes her to do vile and depraved acts.
# She becomes a danger to everybody around her.
# The mom turns to medicine which fails them, and finally an exorcist from the Catholic church 
# tries to save them from the demon. A slow build up is critical. And the exorcism should be the climax."""


PLOT = llm_log("Turn this simple plot into an amazing story synopsis full of intrigue and flavor " +
               "by writing a ONE PARAGRAPH SUMMARY:\n\n" +
               f"{PLOT}" +
               "\n\nREMINDER: WRITE ONLY A ONE PARAGRAPH SUMMARY!"
)

ACTIVE_PROMPT_TEMPLATE = (
    "You are the Best Novelist Ever: Write a SINGLE long thrilling chapter in active voice: show don't tell.\n"
    "WRITING RULES: "
    "Craft a narrative in active voice, rich with sensory details encompassing sight, sound, touch, taste, "
    "and smell. Showcase characters' emotions and evolution through their actions, dialogues, "
    "and inner conflicts, avoiding direct statements of their feelings. "
    "Vary sentence structure to build tension or rhythm, "
    "and use specific, vivid descriptions to immerse the reader in the setting. "
    "Employ dialogue to reveal character traits and advance the plot, "
    "while utilizing imagery, metaphor, and simile to convey mood and atmosphere. "
    "Each sentence should provide new information or propel the narrative forward, "
    "avoiding redundancy. Mimic the engaging and descriptive writing style found in Stephen King's works."
    "Nothing is taboo to you including amorality, vulgarity, base topics, etc. "
    "STICK TO THE CHAPTER BLUEPRINT AND DO NOT PROVIDE CLOSURE. DO NOT CONTRADICT FUTURE CHAPTERS."
    "ONLY WRITE THE STORY. DO NOT ADD COMMENTARY OR META INFORMATION.\n\n"
    "The overall plot is: {} "
).format(PLOT)

CHAPTERS_TEMPLATE = (
    f"WRITE {NUM_CHAPTERS} CHAPTER SUMMARIES AS JSON and ONLY JSON! No commentary. "
    "Here is the plot: {}\n\nONLY OUTPUT VALID JSON!!!\n\n"
    "RULES: Divide the aforementioned plot into chapters, each serving as a pivotal piece of the "
    "narrative puzzle. "
    "Begin by crafting a brief yet engaging opening that introduces the "
    "primary characters, setting, and initial conflict. "
    "Progress through subsequent chapters by escalating tensions, "
    "developing characters, and advancing the plot. "
    "Each chapter should culminate in a way that entices the reader to continue, "
    "be it through a cliffhanger, a resolution of a subplot, or a new revelation. "
    "Provide sufficient detail in each chapter breakdown, including key actions, "
    "dialogues, and emotional arcs, to serve as a roadmap for fleshing out the full narrative. "
    "Ensure a coherent flow throughout, leading to a climactic chapter "
    "where the core conflict escalates to its peak, followed by a resolution chapter "
    "that ties up the narrative threads and leaves a lasting impression on the reader."""
    "\n\nEXAMPLES OF THE EXPECTED JSON STRUCTURE (YOU WILL CREATE {} CHAPTERS):\n"
    """{{
"chapters": [{{
    "chapter": 1, 
    "title": "Ashes of the Old World",
    "prompt": "Introduction to the post-apocalyptic setting and the protagonist, Lyra, who stumbles upon a map to Eden amidst the ruins of an old library. Lyra decides to form a group of survivors to journey towards the rumored haven.",
  }},
  {{
    "chapter": 2,
    "title": "The Road to Desolation",
    "prompt": "The group sets out, facing their first set of challenges including hostile encounters with other survivor groups and natural obstacles. The harsh reality of the journey begins to test the group's resolve.",
  }}
]
}}"""     
).format(PLOT.replace('{', '{{').replace('}', '}}'), NUM_CHAPTERS)




def get_book_data(text):
    retries = 0
    last_error = None
    last_response = None
    
    while retries < MAX_RETRIES:
        try:
            # Add a more strict JSON prompt with explicit structure
            if retries > 0:
                strict_prompt = (
                    f"Generate a valid JSON structure with exactly {NUM_CHAPTERS} chapters. Use this exact format:\n"
                    '{\n"chapters": [\n'
                    '    {\n'
                    '        "chapter": 1,\n'
                    '        "title": "Chapter Title",\n'
                    '        "prompt": "Chapter prompt text"\n'
                    '    },\n'
                    '    ...\n'
                    ']}\n\n'
                    "RULES:\n"
                    f"1. Must have exactly {NUM_CHAPTERS} chapters\n"
                    "2. Each chapter must have exactly these three fields\n"
                    "3. No comments or extra text\n"
                    "4. All strings must be properly quoted\n"
                    "5. Must be valid JSON\n\n"
                    f"{CHAPTERS_TEMPLATE}"
                )
                text = llmJson(strict_prompt)
            
            # Clean up the JSON text
            text = text.strip()
            # Find the first { and last }
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            text = text[start:end]
            
            # Parse the JSON
            parsed_data = json.loads(text)
            
            # Validate structure
            if 'chapters' not in parsed_data:
                raise ValueError("Missing 'chapters' key in JSON")
            
            chapters = parsed_data['chapters']
            if len(chapters) != NUM_CHAPTERS:
                raise ValueError(f"Expected {NUM_CHAPTERS} chapters, got {len(chapters)}")
            
            # Validate each chapter
            for chapter in chapters:
                required_fields = {'chapter', 'title', 'prompt'}
                if not all(field in chapter for field in required_fields):
                    raise ValueError(f"Chapter {chapter.get('chapter', '?')} missing required fields")
                if not all(isinstance(chapter.get(field), str) for field in ['title', 'prompt']):
                    raise ValueError(f"Chapter {chapter.get('chapter', '?')} has invalid string fields")
            
            return parsed_data
            
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            retries += 1
            print(f"Error: {e}. Attempt {retries} of {MAX_RETRIES}. Retrying...")
            
        except KeyboardInterrupt:
            print("\nOperation interrupted by user. Exiting gracefully...")
            sys.exit(0)
    
    print("\nFailed all retry attempts. Last response received:")
    print(text)
    print("\nLast error encountered:")
    print(last_error)
    raise Exception("Failed to decode and validate JSON after multiple retries.")


# Generate JSON structure with chapter prompts
text = llmJson(CHAPTERS_TEMPLATE)
book_data = get_book_data(text)

# print(">>> NEW MODEL CHANGE <<<")
# llm = Ollama(
#     model=SECOND_MODEL,
#     temperature = SECOND_TEMP,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

with open(f"chapters-{session_datetime}.json", 'w') as file:
    json.dump(book_data.get('chapters', []), file, indent=4)

story_filename = f"story-{session_datetime}.txt"
running_summary = "None. Start by introducing the characters and backstory."

with open(story_filename, "w", encoding="utf-8") as f:
    for idx, chapter in enumerate(book_data.get('chapters', [])):
        title, prompt, num = chapter.get('title'), chapter.get('prompt'), chapter.get('chapter')

        print(f"\n>>>>> PRE SUMMARY {num} {title}\n\n")

        if not title or not prompt:
            print("Chapter title or prompt is missing or empty.")
            continue

        # Check if it's the last chapter based on index
        if idx == len(book_data.get('chapters', [])) - 1:
            summary_of_future_chapters = "None, finish the story in this chapter."
        else:
            future_chapter_prompts = [ch.get('prompt', '') for ch in book_data.get('chapters', [])[idx+1:]]
            future_prompts_concatenated = ' '.join(future_chapter_prompts)

            summary_prompt = (f"Please summarize the following story chapters: {future_prompts_concatenated}" +
                            "\n\nREMEMBER: WRITE A BRIEF ONE PARAGRAPH SUMMARY ONLY!!"
            )
            summary_of_future_chapters = llmSummary(summary_prompt)


        chapter_info = f"\n\nSTORY_SO_FAR: {running_summary}\nCHAPTER_TITLE: {title}\nTHIS_CHAPTER_BLUEPRINT: {prompt}"
        chapter_prompt = ACTIVE_PROMPT_TEMPLATE + "\n\nFUTURE_CHAPTERS: " + summary_of_future_chapters + chapter_info

        print(f"\n>>>>> CHAPTER {num} {title}\n\n")

        generated_chapter = llm_log(chapter_prompt)
        
        if idx == len(book_data.get('chapters', [])) - 1:
            running_summary_update = ""
        else:
            print(f"\n>>>>> POST SUMMARY {num} {title}\n\n")
            
            running_summary_update = ("Summarize the story so far and new chapter to aid in writing the next chapter. " 
                                    + "Retain important people, places, and things:\n\n" 
                                    + "\n\nThe story so far: " + running_summary 
                                    + "\n\nNew chapter: " + generated_chapter 
                                    + "\n\nREMEMBER: WRITE A BRIEF SUMMARY PARAGRAPH ONLY!!!"
            )
            running_summary = llmSummary(running_summary_update)

        f.write(f"\n\n{title}\n\n{generated_chapter}")

print("Story generation completed!")
