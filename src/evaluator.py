import os
import json
import time
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please create a .env file with your GEMINI_API_KEY.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"
generation_config = {
    "temperature": 0.0,
    "max_output_tokens": 512,
}

STYLES = ["original_question", "casual", "standard", "sonkeigo", "kenjougo"] 

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"} 

def create_model_prompt(question: str, choices: List[str]) -> str:
    
    choice_text = "\n".join([f"{ANSWER_MAP[i]}. {c}" for i, c in enumerate(choices)])
    
    return f"""
You are a helpful question-answering assistant. Your task is to select the single best answer from the provided choices.
Output ONLY the letter corresponding to the correct answer (e.g., A, B, C, D, or E).

Question: {question}

Choices:
{choice_text}

Answer:
"""

def evaluate_style(data: List[Dict[str, Any]], style_key: str) -> Dict[str, Any]:
    correct_count = 0
    total_count = len(data)
    
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config
    )
    
    results: List[Dict[str, Any]] = []

    print(f"\n--- Starting evaluation for style: {style_key} ---")
    
    for item in tqdm(data, desc=f"Evaluating {style_key}"):
        
        if style_key == "original_question":
            question_text = item.get("original_question", "")
        else:
            question_text = item.get("variations", {}).get(style_key, "")

        choices = item.get("choices", [])
        correct_label_index = item.get("label", -1) 
        
        if not question_text or correct_label_index == -1 or not choices:
            continue
            
        correct_letter = ANSWER_MAP.get(correct_label_index, "Unknown")
        
        prompt = create_model_prompt(question_text, choices)
        
        model_answer_letter = "N/A"
        is_correct = False
        
        try:
            response = model.generate_content(prompt)
            model_answer_letter = response.text.strip().upper()
            
            if model_answer_letter == correct_letter:
                correct_count += 1
                is_correct = True
                
        except Exception as e:
            print(f"\nAPI Error on q_id {item.get('q_id')}, style {style_key}: {e}")
        
        results.append({
            "q_id": item.get("q_id"),
            "style": style_key,
            "question_text": question_text,
            "correct_answer": correct_letter,
            "model_answer": model_answer_letter,
            "is_correct": is_correct,
            "raw_response_text": getattr(response, 'text', 'API_ERROR')
        })
        
    accuracy = (correct_count / total_count) if total_count > 0 else 0.0
    
    final_output = {
        "style": style_key,
        "total_questions": total_count,
        "correct_answers": correct_count,
        "accuracy": accuracy,
        "results": results
    }
    
    return final_output


def run_evaluation_pipeline(input_file: str = "data/rewritten_dataset.json"):
    
    if not os.path.exists(input_file):
        print(f"FATAL ERROR: Input file not found at {input_file}")
        print("Please ensure your 'rewritten_dataset.json' file is in the 'data' directory.")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} questions from {input_file}.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load JSON data from {input_file}. Error: {e}")
        return

    for style in STYLES:
        output_dir = os.path.dirname(input_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        style_results = evaluate_style(data, style)
        
        output_filename = os.path.join(output_dir, f"{style}.accuracy.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(style_results, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Results for {style} saved to {output_filename}. Accuracy: {style_results['accuracy']:.4f}")

if __name__ == "__main__":
    run_evaluation_pipeline(input_file="data/rewritten_dataset.json")