from datasets import load_dataset

def inspect_data():
    print("Loading JCommonsenseQA...")
    dataset = load_dataset("shunk031/JGLUE", name="JCommonsenseQA", split="validation", trust_remote_code=True)
    
    print(f"Dataset loaded. Size: {len(dataset)}")
    print("First example:")
    print(dataset[0])

if __name__ == "__main__":
    inspect_data()
