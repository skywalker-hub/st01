from datasets import load_dataset
import json

ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

data = []
for example in ds["test"]:
    choices_label = example["choices"]["label"]
    choices_text = example["choices"]["text"]
    formatted_choices = "\n".join(
        [f"{label}. {text}" for label, text in zip(choices_label, choices_text)]
    )
    question_with_choices = f"{example['question']}\n\n{formatted_choices}"

    data.append({
        "prompt": [
            {
                "from": "user",
                "value": question_with_choices
            }
        ],
        "final_answer": example["answerKey"]
    })

with open("arc_challenge.json", "w") as f:
    json.dump(data, f, indent=4)

print(f"Saved {len(data)} ARC-Challenge examples to arc_challenge.json")
