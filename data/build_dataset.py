from datasets import load_dataset
import json
import random

prompts = []

print("Loading OpenAssistant dataset...")
oasst = load_dataset("OpenAssistant/oasst1", split="train")

for ex in oasst:
    if "text" in ex and ex["text"]:
        prompts.append(ex["text"])

print("Loading Alpaca dataset...")
alpaca = load_dataset("yahma/alpaca-cleaned", split="train")

for ex in alpaca:
    prompts.append(ex["instruction"])

print("Loading GSM8K dataset...")
gsm = load_dataset("gsm8k", "main", split="train")

for ex in gsm:
    prompts.append(ex["question"])

print("Total prompts collected:", len(prompts))

random.shuffle(prompts)

n = len(prompts)

train = prompts[:int(0.8*n)]
valid = prompts[int(0.8*n):int(0.9*n)]
test = prompts[int(0.9*n):]

with open("data/train.json","w") as f:
    json.dump(train, f)

with open("data/valid.json","w") as f:
    json.dump(valid, f)

with open("data/test.json","w") as f:
    json.dump(test, f)

print("Dataset sizes:")
print("Train:", len(train))
print("Valid:", len(valid))
print("Test:", len(test))