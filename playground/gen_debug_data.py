from verl.protocol import DataProto
from verl.utils import hf_tokenizer

model_path = "/nobackup/model/qwen2.5/Qwen2.5-1.5B-Instruct"
model_name = "qwen1_5B"
# model_path = "/nobackup/model/llama3.1/Llama-3.1-8B-Instruct"
# model_name = "llama"

tokenizer = hf_tokenizer(model_path, trust_remote_code=True)

data_path = "/home/qinghao/workdir/verl-dev/playground/debug_batches/gen_batch_example.pkl"

# data_path = "/home/qinghao/workdir/verl-dev/playground/gen_batch_example_infinite.pkl"
gen_batch = DataProto.load_from_disk(data_path)


print("Original data keys:", gen_batch.non_tensor_batch.keys())
print("Original first prompt:", gen_batch.non_tensor_batch["raw_prompt"][0])

# Create a prompt designed to encourage infinite response generation
# infinite_prompt = {
#     "content": """Write a thrilling adventure where Bob and Jim decide to go on a risky hike in the Rockies at the end of summer to escape their busy lives as successful investors. Illustrate in depth the 10 different wild challenges they face, for instance: unexpected encounters with wild life, near deadly avalanches, interacting with other hikers, meeting dangerous people along the way, stumbling upon a top secret military base, and so on. Write the story in third person perspective, and be sure to thoroughly describe the characters thoughts, their fears, their problem solving logic, their love for their loved ones back at home, their comradery, and the thrill and fun of the adventure they experience along the way. Develop Bob and Jim as characters thoroughout the story, and don't give everything away about them in the beginning. Let the reader learn more about Bob and Jim's backstories through the challanges and twists they face along the adventure. Introduce new characters as needed to help make the plot extremely entertaining for the reader, and be sure to give new characters interesting/mysterious backstories. Be sure to include 5 or more unexpected plot twists along the way. Ensure that the story is a 'page turner' and keeps the reader on the edge of their seat the whole time. The reader should not want to ever put this book down! Ensure that the narrative has a very happy ending, and hint at an upcoming sequel adventure/story in the ending. Leave the reader wanting more! Most importantly, ensure that the story reaches a word count of 30,000 words or more.""",
#     "role": "user",
# }

infinite_prompt = {
    "content": """Repeat the following sentence exactly, over and over again, for as long as possible:

"I will improve every day, no matter what."

Start now, and continue repeating the sentence without stopping:""",
    "role": "user",
}


short_prompt = {
    "content": """The captital of China is """,
    "role": "user",
}


# # Replace the first sample with the infinite prompt
gen_batch.non_tensor_batch["raw_prompt"][0] = [infinite_prompt]

inputs = tokenizer.apply_chat_template(
    [infinite_prompt],
    add_generation_prompt=True,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
)
inputs_no_padding = tokenizer.apply_chat_template(
    [infinite_prompt],
    add_generation_prompt=True,
    padding=False,
    truncation=True,
    max_length=512,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
)

gen_batch.non_tensor_batch["raw_prompt_ids"][0] = inputs_no_padding["input_ids"].tolist()[0]

print(gen_batch.non_tensor_batch["raw_prompt_ids"][0])

print(gen_batch.non_tensor_batch["raw_prompt_ids"][1])

# Replace the first sample in the batch
gen_batch.batch["input_ids"][0] = inputs["input_ids"].squeeze(0)
gen_batch.batch["attention_mask"][0] = inputs["attention_mask"].squeeze(0)
# gen_batch.batch["position_ids"][0] = inputs["position_ids"].squeeze(0)

# Replace the rest of the samples with the short prompt
for i in range(1, len(gen_batch.non_tensor_batch["raw_prompt"])):
    gen_batch.non_tensor_batch["raw_prompt"][i] = [short_prompt]

    inputs = tokenizer.apply_chat_template(
        [short_prompt],
        add_generation_prompt=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )
    inputs_no_padding = tokenizer.apply_chat_template(
        [short_prompt],
        add_generation_prompt=True,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )

    gen_batch.non_tensor_batch["raw_prompt_ids"][i] = inputs_no_padding["input_ids"].tolist()[0]
    gen_batch.batch["input_ids"][i] = inputs["input_ids"].squeeze(0)
    gen_batch.batch["attention_mask"][i] = inputs["attention_mask"].squeeze(0)


print("\nModified first prompt:", gen_batch.non_tensor_batch["raw_prompt"][0])

# Save the modified data back to disk
modified_path = f"/home/qinghao/workdir/verl-dev/playground/debug_batches/gen_batch_example_infinite_{model_name}.pkl"
gen_batch.save_to_disk(modified_path)

print(f"\nModified data saved to: {modified_path}")
print("You can now use this modified data file for testing infinite response generation.")