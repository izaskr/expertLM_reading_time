"""
Calculate perplexity of a model given an independent test set (text that was not included in the training data)
Usage: provide the full path to a model checkpoint as a command line argument, e.g.

Usage: 
python calculate_perplexity.py /home/AK/skrjanec/tmp/bio_steps/checkpoint-1500

"""
import os
import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm



model_id = sys.argv[1] 

files = ["generic2_independent_test.txt", "bio_independent_test.txt", "phy_independent_test.txt", "phy2_independent_test.txt"]

potec_text_folder = "/home/AK/skrjanec/finetune_gpt2-de/potec_plain_texts" # TODO: change this accordingly
potec_file_names = []
for domain_letter in ["b", "p"]:
	for i in range(0,6):
		f = domain_letter + str(i) + ".txt"
		_file_path = os.path.join(potec_text_folder, f)
		files.append(_file_path)

files.append(os.path.join(potec_text_folder, "all_bio.txt"))
files.append(os.path.join(potec_text_folder, "all_phys.txt"))

print("Files ", files)		



device = "cpu"  # "cuda"
#model_id = "dbmdz/german-gpt2"
##model_id = "/home/AK/skrjanec/tmp/phy/checkpoint-56000"  # PHYS 1
#model_id = "/home/AK/skrjanec/tmp/phy_steps/checkpoint-50000"  # PHYS 2

#model_id = "/home/AK/skrjanec/tmp/bio/checkpoint-80000"  # BIO 1
#model_id = "/home/AK/skrjanec/tmp/bio_steps/checkpoint-1500"  # BIO 2  40,000
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


def read_in_test_file(path_to_file):
	all_text_one_line = ""
	with open(path_to_file) as f:
		for line in f:
			all_text_one_line += " " + line.strip()
	return all_text_one_line


def calculate_perplexity(seq_len, stride, max_length):
	nlls = []
	prev_end_loc = 0
	for begin_loc in tqdm(range(0, seq_len, stride)):
		end_loc = min(begin_loc + max_length, seq_len)
		trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
		input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
		target_ids = input_ids.clone()
		target_ids[:, :-trg_len] = -100

		with torch.no_grad():
			outputs = model(input_ids, labels=target_ids)
			# loss is calculated using CrossEntropyLoss which averages over input tokens.
			# Multiply it with trg_len to get the summation instead of average.
			# We will take average over all the tokens to get the true average
			# in the last step of this example.
			neg_log_likelihood = outputs.loss * trg_len
		nlls.append(neg_log_likelihood)

		prev_end_loc = end_loc
		if end_loc == seq_len:
			break
	ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
	return ppl


for path_to_file in files:

	text = read_in_test_file(path_to_file)
	encodings = tokenizer(text, return_tensors="pt")
	max_length = model.config.n_positions
	stride = 1024 # 512 # 1024
	seq_len = encodings.input_ids.size(1)

	perplexity = calculate_perplexity(seq_len, stride, max_length)
	print("Perplexity, model id, text", model_id, path_to_file, perplexity)



