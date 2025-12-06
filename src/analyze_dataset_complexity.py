import json
import MeCab
import unidic_lite
import statistics
from collections import defaultdict
import os

def analyze():
    tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")

    with open(os.path.join(os.path.dirname(__file__), "..", "data", "jcommonsense", "rewritten_dataset.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        
    stats = defaultdict(lambda: {"pos_counts": defaultdict(int), "token_counts": [], "jaccard_scores": []})
    
    def get_analysis(text):
        node = tagger.parseToNode(text)
        tokens = [] # lemmas
        pos_list = []
        while node:
            if node.surface: # skip BOS/EOS
                features = node.feature.split(",")
                pos = features[0]
                # idx 7 is lemma in unidic
                lemma = features[7] if len(features) > 7 and features[7] != "*" else node.surface
                
                tokens.append(lemma)
                pos_list.append(pos)
            node = node.next
        return tokens, pos_list

    for item in data:
        orig_text = item.get('original_question', "")
        if not orig_text: continue
        
        orig_tokens, orig_pos = get_analysis(orig_text)
        orig_set = set(orig_tokens)
        
        if not orig_tokens: continue

        stats["original_question"]["token_counts"].append(len(orig_tokens))
        for pos in orig_pos:
            stats["original_question"]["pos_counts"][pos] += 1
        stats["original_question"]["jaccard_scores"].append(1.0)

        variations = item.get('variations', {})
        for v_type, v_text in variations.items():
            if not v_text: continue
            
            v_tokens, v_pos = get_analysis(v_text)
            v_set = set(v_tokens)
            
            # Jaccard Similarity
            if not v_set:
                jaccard = 0.0
            else:
                intersection = len(orig_set.intersection(v_set))
                union = len(orig_set.union(v_set))
                jaccard = intersection / union if union > 0 else 0.0
            
            stats[v_type]["jaccard_scores"].append(jaccard)
            stats[v_type]["token_counts"].append(len(v_tokens))
            
            for pos in v_pos:
                stats[v_type]["pos_counts"][pos] += 1

    categories = ["original_question", "casual", "standard", "sonkeigo", "kenjougo"]
    
    # pretty!
    print(f"{'Category':<20} | {'Avg Len':<8} | {'Jaccard':<8} | {'Func/Cont':<9} | {'Top POS Distribution'}")
    print("-" * 100)
    
    for cat in categories:
        s = stats[cat]
        if not s["token_counts"]:
            print(f"{cat:<12} | N/A")
            continue
            
        avg_len = statistics.mean(s["token_counts"])
        
        # Handle Jaccard for original_question
        if cat == "original_question":
            avg_jaccard_str = f"{'N/A':<8}"
        else:
            avg_jaccard_str = f"{statistics.mean(s['jaccard_scores']):<8.3f}"
        
        total_pos = sum(s["pos_counts"].values())
        
        func_count = s["pos_counts"].get("助詞", 0) + s["pos_counts"].get("助動詞", 0)
        cont_count = s["pos_counts"].get("名詞", 0) + s["pos_counts"].get("動詞", 0) + s["pos_counts"].get("形容詞", 0) + s["pos_counts"].get("副詞", 0)
        
        ratio = func_count / cont_count if cont_count > 0 else 0.0
        
        # top 5 POS
        pos_translation = {
            "助詞": "Particle",
            "助動詞": "Auxiliary Verb",
            "名詞": "Noun",
            "動詞": "Verb",
            "形容詞": "Adjective",
            "副詞": "Adverb",
            "補助記号": "Auxiliary Symbol",
            "代名詞": "Pronoun",
            "感動詞": "Interjection",
            "記号": "Symbol",
            "形状詞": "Adjectival Noun",
            "接続詞": "Conjunction",
            "接頭辞": "Prefix",
            "語:動詞": "Verb", # MeCab specific, sometimes verb is `語:動詞`
            "語:名詞": "Noun", # MeCab specific, sometimes noun is `語:名詞`
        }
        top_pos = sorted(s["pos_counts"].items(), key=lambda x: x[1], reverse=True)[:5]
        top_pos_str = ", ".join([f"{pos_translation.get(p, p)}({c/total_pos:.2f})" for p, c in top_pos])
        
        print(f"{cat:<20} | {avg_len:<8.2f} | {avg_jaccard_str} | {ratio:<9.2f} | {top_pos_str}")

if __name__ == "__main__":
    print()
    analyze()
    print()

