"""
Load the texts from PoTeC.
For each domain, calculate the per-word surprisal. The context for calculating surprisal: entire text.

Save the surprisals to a csv file for further statistical analysis.

Usage: three command line arguments: full path to model checkpoint, domain of model, checkpoint step

Usage:
python main_surprisal_cli.py /home/AK/skrjanec/tmp/bio/checkpoint-500 bio_models 500

The result will be saved as variant_steps/bio_models/bioTexts_surprisals500.csv
"""
import os
import sys
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead
from surprisal_per_word_from_scratch import surprisal_per_word_from_scratch


#model_id = sys.argv[1]
#model_domain_folder = sys.argv[2]
#checkpoint = sys.argv[3] 

# load PoTeC: has GPT-2 surprisal
with open("data/potec_different_features/potec_linguistic_features.json", "r") as f:
	potec = json.load(f)


# load PoTeC: original linguistic features: Technical Term inidicator, including SSTS pos tag 
with open("data/potec_different_features/potec_sentences_noncomputational_ling_features.json", "r") as f:
	potec_basic = json.load(f)


# load the computational linguistic features of POTEC, relevant: universal POS tag from spacy
with open("data/potec_different_features/potec_sentences_computational_features.json", "r") as f2:
	potec_comp = json.load(f2)


#models = ["dbmdz/german-gpt2", "", ""]
models = ["/home/AK/skrjanec/tmp/bio/checkpoint-80000", "/home/AK/skrjanec/tmp/phy_steps/checkpoint-50000"]
device = "cpu"

#surprisals_biotexts, surprisals_phystexts = {"gpt": [], "bio-gpt": [], "phys-gpt": [], "techterm": []}, {"gpt": [], "bio-gpt": [], "phys-gpt": [], "techterm": []}

general_model_id = "dbmdz/german-gpt2"
general_tok = AutoTokenizer.from_pretrained(general_model_id)
general_lm = AutoModelWithLMHead.from_pretrained(general_model_id).to(device)

#surprisals_all = {"gpt": [], "domain-spec-gpt": [], "techterm": []}
#for domain_text_id in ["p0", "p1", "p2", "p3", "p4", "p5"]:  # CHANGE HERE
for domain_text_id in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']:
	surprisals_all = {"gpt": [], "bio-gpt": [], "phys-gpt":[]}
	all_pos_tags = []
	all_ssts_tags = []
	all_dependencies, n_rights, n_lefts, dep_distances = [], [], [], []
	# join all sentences into a text, a single string
	entire_text = ""
	for sentence_id, features in potec[domain_text_id].items():
		sentence_str = " ".join(features["words"])
		sentence_str = sentence_str + ". "
		entire_text += sentence_str
		# retrieve the universal POS tags as well
		all_pos_tags.extend(potec_comp[domain_text_id][sentence_id]["uni_pos_tags"])
		all_dependencies.extend(potec_comp[domain_text_id][sentence_id]["deps"])
		n_rights.extend(potec_comp[domain_text_id][sentence_id]["n_rights"])
		n_lefts.extend(potec_comp[domain_text_id][sentence_id]["n_lefts"])
		dep_distances.extend(potec_comp[domain_text_id][sentence_id]["dep_distance"])
		n_words_in_sent = len(potec_basic[domain_text_id][sentence_id])
		for word_id in range(1, n_words_in_sent+1):
			word_id = str(word_id)
			all_ssts_tags.append(potec_basic[domain_text_id][sentence_id][word_id]["SST_pos"])
			#print("all_ssts_tags")
			#import pdb; pdb.set_trace()
	# remove the last space after the last sentence
	entire_text = entire_text[:-1]
	for model_id in models:
		tokenizer = AutoTokenizer.from_pretrained(model_id)
		lm_model = AutoModelWithLMHead.from_pretrained(model_id).to(device)
		domain_text_surprisals = surprisal_per_word_from_scratch(sentence=entire_text, correcting=True, device=device, model_id=model_id, tokenizer=tokenizer, model=lm_model)[0]   # surprisal
		#print("text", entire_text)
		#print("surprisals", domain_text_surprisals)
		#import pdb; pdb.set_trace()
		print(domain_text_id,"length of text, length of surprisal list", len(entire_text.split()), len(domain_text_surprisals))

		# get general GPT-2 surprisals again too
		general_text_surprisals = surprisal_per_word_from_scratch(sentence=entire_text, correcting=True, device=device,model_id=general_model_id, tokenizer=general_tok,model=general_lm)[0] 
		surprisals_all["gpt"] = general_text_surprisals
		#surprisals_all["domain-spec-gpt"] = domain_text_surprisals
		if "bio" in model_id:
			surprisals_all["bio-gpt"] = domain_text_surprisals
		if "phy" in model_id:
			surprisals_all["phys-gpt"] = domain_text_surprisals
		
	# write into csv
	#col_n = "PhysGPT"
	#if "bio" in model_id:
	#	col_n = "BioGPT"
	pd_domain = pd.DataFrame(list(zip(all_pos_tags, all_ssts_tags, all_dependencies, n_rights, n_lefts, dep_distances, surprisals_all["gpt"],surprisals_all["bio-gpt"], surprisals_all["phys-gpt"])), columns = ["UniPOS","SSTPOS", "DependencyType", "NdepRight", "NdepLeft", "DistanceToHead", "GPT2", "BioGPT", "PhysGPT"])
	final_folder_name = "variant_text" 
	final_file_name = domain_text_id + "_surprisals.csv"
	f_path = os.path.join(final_folder_name, final_file_name)
	pd_domain.to_csv(f_path, sep=",", index=False, encoding="utf-8")

