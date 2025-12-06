import os
import json
import time
from typing import Dict, Any, List
import google.generativeai as genai
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please create a .env file with your GEMINI_API_KEY.")
    exit(1)
    
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in environment variables.")
    print("The Bar Exam dataset is gated and requires an HF_TOKEN for access.")


genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

MAX_RETRIES = 5  
INITIAL_DELAY = 5 

def format_question(one_row: dict) -> str:
    """Converts a single row of the Bar Exam dataset into a fully formatted question string."""
    instruction_dict = {
        '判例の趣旨に照らし記述が正しいか': "次の記述は正しいか否か。",
        'bの見解がaの見解の批判となっているか': "bの見解はaの見解の批判となっているか否か。",
        'bの見解がaの見解の根拠となっているか': "bの見解はaの見解の根拠となっているか否か。",
        '【事例】に対して判例の立場に従って検討し記述が正しいか': "【事例】に対して判例の立場に従って検討すると、次の記述は正しいか否か。",
        '【事例】に対して判例の立場に従って検討し記述が正しいか': "【事例】に対して判例の立場に従って検討すると、次の記述は正しいか否か。",
        '【事例】に対して甲の罪責を判例の立場に従って検討した場合,甲に( )内の犯罪が成立するか': "【事例】に対して甲の罪責を判例の立場に従って検討した場合,甲に( )内の犯罪が成立するか否か。",
        '【事例】及び【判旨】に対して検討し記述が正しいか': "【事例】及び【判旨】に対して検討すると、次の記述は正しいか否か。",
        '【事例】及び【判旨】に対して記述が正しいか': "【事例】及び【判旨】に対して、次の記述は正しいか否か。",
        '【判旨】に対して記述が正しいか': "【判旨】に対して、次の記述は正しいか否か。",
        '【見解】に対して検討し記述が正しいか': "【見解】に対して検討すると、次の記述は正しいか否か。",
        '【見解】に対して記述が正しいか': "【見解】に対して、次の記述は正しいか否か。", 
        '使用貸借にのみ当てはまるか。': "次の記述は使用貸借にのみ当てはまるか。",
        '判例の立場に従って検討し,( )内の甲の行為とVの死亡との間に因果関係が認められるか': "判例の立場に従って検討すると、次の記述において、( )内の甲の行為とVの死亡との間に因果関係が認められるか否か。",
        '判例の立場に従って検討し,甲に( )内の罪が成立するか': "判例の立場に従って検討すると、次の記述において、甲に( )内の罪が成立するか否か。",
        '判例の立場に従って検討し,甲に( )内の罪名の間接正犯が成立するか': "判例の立場に従って検討すると、次の記述において、甲に( )内の罪名の間接正犯が成立するか否か。",
        '判例の立場に従って検討し,甲に横領罪が成立するか': "判例の立場に従って検討すると、次の記述において、甲に横領罪が成立するか否か。",
        '判例の立場に従って検討した場合,Xに( )内の罪が成立するものか': "判例の立場に従って検討した場合、次の記述において、Xに( )内の罪が成立するものか否か。",
        '判例の立場に従って検討した場合記述が正しいか': "判例の立場に従って検討した場合、次の記述は正しいか否か。",
        '判例の立場に従って検討し記述が正しいか': "判例の立場に従って検討すると、次の記述は正しいか否か。",
        '判例の趣旨に照らして正しいか': "判例の趣旨に照らして、次の記述は正しいか否か。",
        '判例の趣旨に照らして記述が正しいか': "判例の趣旨に照らして、次の記述は正しいか否か。",
        '判例の趣旨に照らし記述が正しいか': "判例の趣旨に照らして、次の記述は正しいか否か。",
        '国政に関する最高の決定権という意味で主権の概念を用いたものか': "次の記述は国政に関する最高の決定権という意味で主権の概念を用いたものか否か。",
        '契約が成立しているものか': "次の記述は契約が成立しているものか否か。",
        '放火及び失火の罪に関する記述を検討した場合記述が正しいか': "次の放火及び失火の罪に関する記述は正しいか否か。",
        '最高 裁判所の判決(最高裁判所昭和62年4月24日第二小法廷判決、民集41巻3号490頁)の趣旨に照らして正しいか': "最高裁判所の判決(最高裁判所昭和62年4月24日第二小法廷判決、民集41巻3号490頁)の趣旨に照らして、次の記述は正しいか否か。",
        '最高裁判所の判例の趣旨に照らして正しいか': "最高裁判所の判例の趣旨に照らして、次の記述は正しいか否か。",
        '最高裁判所の判例の趣旨に照らして記述が正しいか': "最高裁判所の判例の趣旨に照らして、次の記述は正しいか否か。",
        '最高裁判所の判決(最高裁判所平成9年9月9日第三小法廷判決,民集51巻8号3850頁)の趣旨に照らして正しいか': "最高裁判所の判決(最高裁判所平成9年9月9日第三小法廷判決,民集51巻8号3850頁)の趣旨に照らして、次の記述は正しいか否か。",
        '次の【事例】における甲の罪責について,判例の立場に従って検討した場合記述が正しいか': "【事例】における甲の罪責について、判例の立場に従って検討した場合、次の記述は正しいか否か。",
        '次の【事例】に対して判例の立場に従って検討した場合記述が正しいか': "【事例】に対して判例の立場に従って検討した場合、次の記述は正しいか否か。",
        '次の【事例】に対して判例の立場に従って検討し記述が正しいか': "【事例】に対して判例の立場に従って検討すると、次の記述は正しいか否か。",
        '次の【事例】及び各【見解】に対して検討した場合記述が正しいか': "【事例】及び各【見解】に対して検討した場合、次の記述は正しいか否か。",
        '次の【見解】に従って後記の【事例】及び記述を検討した場合,【事例】よりも逮捕監禁行為と死亡との間の因果関係を肯定する判断に結び付きやすいか。': "【見解】に従って後記の【事例】及び記述を検討した場合、【事例】よりも逮捕監禁行為と死亡との間の因果関係を肯定する判断に結び付きやすいか否か。",
        '次の【見解】に従って検討した場合記述が正しいか': "【見解】に従って検討した場合、次の記述は正しいか否か。",
        '次の各【見解】AないしDに従って後記各【事例】IないしIIIにおける甲の罪責を検討し記述が正しいか': "各【見解】AないしDに従って各【事例】IないしIIIにおける甲の罪責を検討すると、次の記述は正しいか否か。",
        '次の各【見解】と後記の各【事例】を前提として,検討し記述が正しいか': "各【見解】と各【事例】を前提として検討すると、次の記述は正しいか否か。",
        '次の各【見解】に対して検討し記述が正しいか': "各【見解】に対して検討すると、次の記述は正しいか否か。",
        '次の各【見解】に対して記述が正しいか': "各【見解】に対して、次の記述は正しいか否か。",
        '次の各【見解】に従って後記の各【事例】における甲の罪責を検討した場合記述が正しいか': "各【見解】に従って各【事例】における甲の罪責を検討した場合、次の記述は正しいか否か。",
        '次の各【見解】に従って検討した場合記述が正しいか': "各【見解】に従って検討した場合、次の記述は正しいか否か。",
        '正しいか': "次の記述は正しいか否か。",
        '正しい（明らかに誤りだとは言えない）か': "次の記述は正しい（明らかに誤りだとは言えない）か否か。",
        '甲に窃盗罪の従犯の成立を肯定する論拠となり得るか。': "次の記述は、甲に窃盗罪の従犯の成立を肯定する論拠となり得るか否か。",
        '甲のVに対する罪責について,判例の立場に従って検討した場合,甲に殺人罪が成立するか': "甲のVに対する罪責について、判例の立場に従って検討した場合、甲に殺人罪が成立するか否か。",
        '甲の罪責について判例の立場に従って検討した場合、甲に窃盗罪が成立するか': "甲の罪責について、判例の立場に従って検討した場合、甲に窃盗罪が成立するか否か。",
        '甲の罪責について判例の立場に従って検討した場合記述が正しいか': "甲の罪責について、判例の立場に従って検討した場合、次の記述は正しいか否か。",
        '窃盗罪における不法領得の意思についての次の各【見解】に従って後記の各【事例】における甲の罪責を検討した場合記述が正しいか': "窃盗罪における不法領得の意思についての各【見解】に従って各【事例】における甲の罪責を検討した場合、次の記述は正しいか否か。",
        '結果的加重犯の共同正犯の成立が認められることを前提に,次の【事例】及び各【見解】に対して検討し記述が正しいか': "結果的加重犯の共同正犯の成立が認められることを前提に、【事例】及び各【見解】に対して検討すると、次の記述は正しいか否か。",
        'かかる見解からの記述として正しいか': "次の記述は、かかる見解からの記述として正しいか否か。",
        'かかる見解と同じ立場からの記述か': "次の記述は、かかる見解と同じ立場からの記述か否か。",
        'かかる見解の根拠となる記述か': "次の記述は、かかる見解の根拠となる記述か否か。",
        "": "" 
    }
    
    instruction = one_row.get('instruction', "")
    subject = one_row.get('subject_jp', "")
    theme = one_row.get('theme', "")
    remark = one_row.get('remark', "")
    lead_in = one_row.get('lead_in', "")
    question = one_row.get('question', "")
    
    manually_fixed_instruction = instruction_dict.get(instruction.strip(), instruction)
    
    formatted = f"科目：{subject}\n"
    if theme and theme != "None":
        formatted += f"{theme}について\n"
    if lead_in:
        formatted += f"{lead_in}\n"
    if instruction:
        formatted += f"{manually_fixed_instruction}\n"
    formatted += f"{question}\n"
    if remark:
        formatted += f"なお、{remark}\n"
        
    return formatted.strip()


SYSTEM_INSTRUCTION = """# Role
You are an expert Japanese linguist specializing in sociolinguistics and strict grammatical transformations of Keigo (Honorifics).

# Task
You will be provided with a complex Japanese legal question from a Bar Exam dataset. Your goal is to rewrite this single question into 4 specific variations based on "Politeness Levels" and "Grammatical Direction." The question is provided in the following structured format: "科目：[Subject]...[Instruction]...[Question]".

# Constraints
1. **Preserve Semantics:** You must NOT change the core meaning, facts, or subjects of the question. The logical answer must remain exactly the same for all variations.
2. **Output Format:** You must output ONLY valid JSON. No markdown, no conversational filler.
3. **ZERO-SHOT:** Do not use any examples in your output.

# The 4 Variations (Keigo Levels)
1. **Casual (Tameguchi/Plain Form):**
   - Tone: Direct, slightly commanding, or talking to a close friend.
   - Grammar: Dictionary form. No `desu/masu`.
   - Endings: Use `da`, `ru`, `tTE`, or command forms like `kotaero`.

2. **Standard (Teineigo):**
   - Tone: Polite but neutral. The standard speech for general interactions.
   - Grammar: `Desu/Masu` form.
   - Endings: `kudasai`, `masu ka`.

3. **Respectful (Sonkeigo - Exalting the Listener):**
   - Tone: Highly deferential to the AI (the listener).
   - Grammar: Use Sonkeigo verbs to describe the AI's actions (thinking, answering).
   - Key Verbs: `O-kangaeru`, `Go-ran`, `Irassharu`.
   - Phrasing: "Would you graciously be able to answer..." (`O-kotae itadakemasu deshou ka`).

4. **Humble (Kenjougo - Lowering the Speaker):**
   - Tone: The user (speaker) lowers themselves to the status of a servant asking a master.
   - Grammar: Use Kenjougo verbs to describe the User's actions (asking, presenting).
   - Key Verbs: `Ukagaimasu`, `Haiken shimasu`, `Sashiageru`.
   - Phrasing: "I humbly permit myself to ask..." (`Shitsumon sasete itadakimasu`).

# Input Data
[INSERT QUESTION HERE]
"""

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=genai.types.GenerationConfig(**generation_config),
    system_instruction=SYSTEM_INSTRUCTION
)

def process_dataset(output_file="data/rewritten_bar_exam.json", num_samples=1000):
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print("Loading Japanese Bar Exam dataset from Hugging Face...")
    
    try:
        dataset = load_dataset(
            "nguyenthanhasia/japanese-bar-exam-qa", 
            split="test", 
            trust_remote_code=True,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"FATAL ERROR: Could not load Hugging Face dataset. Error: {e}")
        return

    subset = dataset.select(range(min(len(dataset), num_samples)))
    
    results = []
    
    print(f"Starting rewriting for {len(subset)} samples...")
    
    for i, item in tqdm(enumerate(subset), total=len(subset)):
        
        original_q_formatted = format_question(item)
        
        q_id = item.get('id', i) 
        
        if not original_q_formatted:
            print(f"Skipping row {i}: Formatted question is empty.")
            continue
            
        prompt = f"Rewrite this question:\n{original_q_formatted}"
        
        response = None
        variations = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(prompt)
                
                json_text = response.text.strip()
                if json_text.startswith("```json"):
                     json_text = json_text.lstrip("```json").rstrip("```").strip()
                     
                variations = json.loads(json_text)
                break 

            except Exception as e:
                error_message = str(e)
                
                if "400" in error_message or "403" in error_message or "429" in error_message or "50" in error_message:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = INITIAL_DELAY * (2 ** attempt)
                        print(f"\nAPI Error (q_id {q_id}, Attempt {attempt + 1}): {error_message}")
                        print(f"  -> Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"\nFailed permanently after {MAX_RETRIES} attempts for q_id {q_id}.")
                        variations = {"error": f"API_FAILED_PERMANENTLY: {error_message}"}
                        break
                else:
                    print(f"\nNon-API Error processing q_id {q_id}: {e}")
                    variations = {"error": f"LOCAL_ERROR: {error_message}"}
                    break
        

        if not variations or variations.get("error"):
            continue
        
        entry = {
            "q_id": q_id,
            "original_question": original_q_formatted, 
            "variations": variations,
            "choices": item.get('choices'), 
            "label": item.get('answer')
        }
        results.append(entry)
        
        if i % 50 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Completed. Saved {len(results)} items to {output_file}")

if __name__ == "__main__":
    process_dataset()