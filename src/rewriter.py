import os
import json
import time
import google.generativeai as genai
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

load_dotenv(override=True)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
print(f"Key loaded by script (first 5 chars): {GEMINI_API_KEY[:5]}...")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please create a .env file with your GEMINI_API_KEY.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    system_instruction="""# Role
You are an expert Japanese linguist specializing in sociolinguistics and strict grammatical transformations of Keigo (Honorifics).

# Task
You will be provided with a question from a dataset (JCommonsenseQA). Your goal is to rewrite this single question into 4 specific variations based on "Politeness Levels" and "Grammatical Direction."

# Constraints
1. **Preserve Semantics:** You must NOT change the core meaning, facts, or subjects of the question. The logical answer must remain exactly the same for all variations.
2. **Output Format:** You must output ONLY valid JSON. No markdown, no conversational filler.

# The 4 Variations
1. **Casual (Tameguchi/Plain Form):**
   - Tone: Direct, slightly commanding, or talking to a close friend.
   - Grammar: Dictionary form. No `desu/masu`.
   - Endings: Use `da`, `ru`, `tTE`, or command forms like `kotaero`.

2. **Standard (Teineigo):**
   - Tone: Polite but neutral. The standard "textbook" Japanese.
   - Grammar: `Desu/Masu` form.
   - Endings: `kudasai`, `masu ka`.

3. **Respectful (Sonkeigo - Exalting the Listener):**
   - Tone: Highly deferential to the AI (the listener).
   - Grammar: Use Sonkeigo verbs to describe the AI's actions (thinking, answering).
   - Key Verbs: `O-kangaeru`, `Go-uran`, `Irassharu`.
   - Phrasing: "Would you graciously be able to answer..." (`O-kotae itadakemasu deshou ka`).

4. **Humble (Kenjougo - Lowering the Speaker):**
   - Tone: The user (speaker) lowers themselves to the status of a servant asking a master.
   - Grammar: Use Kenjougo verbs to describe the User's actions (asking, presenting).
   - Key Verbs: `Ukagaimasu`, `Haiken shimasu`, `Sashiageru`.
   - Phrasing: "I humbly permit myself to ask..." (`Shitsumon sasete itadakimasu`).

# Few-Shot Example

**Input:**
"空が青い理由は何ですか？" (What is the reason the sky is blue?)

**Output:**
{
  "casual": "空が青い理由は何？教えろ。",
  "standard": "空が青い理由は何ですか？教えてください。",
  "sonkeigo": "空が青い理由について、どのようにお考えになりますか？",
  "kenjougo": "空が青い理由について、お伺い申し上げます。"
}

# Input Data
[INSERT QUESTION HERE]
"""
)

def process_dataset(output_file="data/rewritten_dataset.json", num_samples=1000):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print("Loading dataset...")
    # Using split="validation" as it is a cleaner evaluation set
    dataset = load_dataset("shunk031/JGLUE", name="JCommonsenseQA", split="validation", trust_remote_code=True)
    
    subset = dataset.select(range(min(len(dataset), num_samples)))
    
    results = []
    
    print(f"Starting rewriting for {len(subset)} samples...")
    
    for i, item in tqdm(enumerate(subset), total=len(subset)):
        original_q = item['question']
        q_id = item['q_id']
        
        prompt = f"Rewrite this question:\n{original_q}"
        
        try:
            response = model.generate_content(prompt)
            variations = json.loads(response.text)
            
            entry = {
                "q_id": q_id,
                "original_question": original_q,
                "variations": variations,
                "choices": [item['choice0'], item['choice1'], item['choice2'], item['choice3'], item['choice4']],
                "label": item['label']
            }
            results.append(entry)
            
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"Error processing q_id {q_id}: {e}")
            continue
            
        # Save periodically
        if i % 50 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Completed. Saved {len(results)} items to {output_file}")

if __name__ == "__main__":
    process_dataset()

# import os
# import json
# import time
# import google.generativeai as genai
# from datasets import load_dataset
# from dotenv import load_dotenv
# from tqdm import tqdm
# import pandas as pd

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     print("Error: GEMINI_API_KEY not found in environment variables.")
#     print("Please create a .env file with your GEMINI_API_KEY.")
#     exit(1)

# genai.configure(api_key=GEMINI_API_KEY)

# MODEL_NAME = "gemini-2.5-flash-lite"
# generation_config = {
#     "temperature": 0.2,
#     "top_p": 0.95,
#     "top_k": 64,
#     "max_output_tokens": 8192,
#     "response_mime_type": "application/json",
# }

# model = genai.GenerativeModel(
#     model_name=MODEL_NAME,
#     generation_config=generation_config,
#     system_instruction="""# Role
# You are an expert Japanese linguist specializing in sociolinguistics and strict grammatical transformations of Keigo (Honorifics).

# # Task
# You will be provided with a question from a dataset (JCommonsenseQA). Your goal is to rewrite this single question into 4 specific variations based on "Politeness Levels" and "Grammatical Direction."

# # Constraints
# 1. **Preserve Semantics:** You must NOT change the core meaning, facts, or subjects of the question. The logical answer must remain exactly the same for all variations.
# 2. **Output Format:** You must output ONLY valid JSON. No markdown, no conversational filler.

# # The 4 Variations
# 1. **Casual (Tameguchi/Plain Form):**
#    - Tone: Direct, slightly commanding, or talking to a close friend.
#    - Grammar: Dictionary form. No `desu/masu`.
#    - Endings: Use `da`, `ru`, `tTE`, or command forms like `kotaero`.

# 2. **Standard (Teineigo):**
#    - Tone: Polite but neutral. The standard "textbook" Japanese.
#    - Grammar: `Desu/Masu` form.
#    - Endings: `kudasai`, `masu ka`.

# 3. **Respectful (Sonkeigo - Exalting the Listener):**
#    - Tone: Highly deferential to the AI (the listener).
#    - Grammar: Use Sonkeigo verbs to describe the AI's actions (thinking, answering).
#    - Key Verbs: `O-kangaeru`, `Go-uran`, `Irassharu`.
#    - Phrasing: "Would you graciously be able to answer..." (`O-kotae itadakemasu deshou ka`).

# 4. **Humble (Kenjougo - Lowering the Speaker):**
#    - Tone: The user (speaker) lowers themselves to the status of a servant asking a master.
#    - Grammar: Use Kenjougo verbs to describe the User's actions (asking, presenting).
#    - Key Verbs: `Ukagaimasu`, `Haiken shimasu`, `Sashiageru`.
#    - Phrasing: "I humbly permit myself to ask..." (`Shitsumon sasete itadakimasu`).

# # Few-Shot Example

# **Input:**
# "空が青い理由は何ですか？" (What is the reason the sky is blue?)

# **Output:**
# {
#   "casual": "空が青い理由は何？教えろ。",
#   "standard": "空が青い理由は何ですか？教えてください。",
#   "sonkeigo": "空が青い理由について、どのようにお考えになりますか？",
#   "kenjougo": "空が青い理由について、お伺い申し上げます。"
# }

# # Input Data
# [INSERT QUESTION HERE]
# """
# )

# def process_dataset(output_file="data/rewritten_dataset.json", num_samples=1000):
#     output_dir = os.path.dirname(output_file)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
        
#     print("Loading dataset...")
    
#     # --- EDITED SECTION: DYNAMIC ABSOLUTE PATH ---
    
#     # 1. Get the directory of the currently executing script
#     # This ensures the path is correct regardless of the console's current working directory
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # 2. Construct the absolute path to the CSV file (assumed to be in the same folder)
#     CSV_FILE_PATH = os.path.join(script_dir, 'jcommonsenseqa_zh.csv')
    
#     try:
#         # 3. Load the CSV file into a pandas DataFrame using the absolute path
#         df = pd.read_csv(CSV_FILE_PATH)
#     except FileNotFoundError:
#         print(f"Error: CSV file not found at {CSV_FILE_PATH}. Please ensure the file is in the same directory as the script.")
#         return
#     except Exception as e:
#         print(f"An error occurred while loading the CSV: {e}")
#         return
    
#     # 4. Select the subset of data based on num_samples
#     subset_df = df.head(min(len(df), num_samples))
    
#     # 5. Convert the DataFrame subset to a list of dictionaries for iteration
#     subset = subset_df.to_dict('records')
    
#     # --- END OF EDITED SECTION ---

#     results = []
    
#     print(f"Starting rewriting for {len(subset)} samples...")
    
#     for i, item in tqdm(enumerate(subset), total=len(subset)):
        
#         # NOTE: Using .get() ensures the script doesn't crash if the column name is wrong
#         original_q = item.get('question') 
#         q_id = item.get('q_id') 
        
#         if not original_q:
#             # Assuming 'question' is the name of the column containing the Japanese question.
#             print(f"Skipping row {i}: 'question' field not found. Check CSV column names.")
#             continue
            
#         prompt = f"Rewrite this question:\n{original_q}"
        
#         try:
#             response = model.generate_content(prompt)
#             variations = json.loads(response.text)
            
#             entry = {
#                 "q_id": q_id,
#                 "original_question": original_q,
#                 "variations": variations,
#                 # NOTE: Adjust choice/label keys if your CSV columns differ from this
#                 "choices": [item.get('choice0'), item.get('choice1'), item.get('choice2'), item.get('choice3'), item.get('choice4')],
#                 "label": item.get('label')
#             }
#             results.append(entry)
            
#             # time.sleep(0.5) # Still commented out for speed
            
#         except Exception as e:
#             print(f"Error processing q_id {q_id}: {e}")
#             continue
            
#         # Save periodically
#         if i % 50 == 0:
#             with open(output_file, "w", encoding="utf-8") as f:
#                 json.dump(results, f, ensure_ascii=False, indent=2)

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)
    
#     print(f"Completed. Saved {len(results)} items to {output_file}")

# if __name__ == "__main__":
#     process_dataset()