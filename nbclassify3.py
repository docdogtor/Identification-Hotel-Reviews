import sys
import glob
import os
import collections

model = "nbmodel.txt"
output = "nboutput.txt"
input_path = sys.argv[1]

all_files = glob.glob(os.path.join(input_path, '*/*/*/*.txt'))

# for file in all_files:
#     class1, class2, fold, file_name = file.split('/')[-4:]
#     if "positive" in class1:
#         class1 = "positive"
#     elif "negative" in class1:
#         class1 = "negative"
#     if "truthful" in class2:
#         class2 = "truthful"
#     elif "deceptive" in class2:
#         class2 = "deceptive"
#     test_data[class1].append(file)
#     test_data[class2].append(file)

def new_word_format(word):
    new_word = word.lower().strip()
    char_list = []
    for char in new_word:
        if char not in ". , /":
            char_list.append(char)
    return "".join(char_list)

def read_tokens_from_files(all_files):
    file_tokens_dict = {}
    for file_txt in all_files:
        tokens_list = []
        current_file = open(file_txt, "r")
        for line in current_file:
            word_list = line.split(" ")
            for word in word_list:
                word = new_word_format(word)
                tokens_list.append(word)
        current_file.close()
        file_tokens_dict[file_txt] = tokens_list
    return file_tokens_dict

# ...WordDict{[word]:this word's number}, num...Word int(), num...File int()
file_tokens_dict = read_tokens_from_files(all_files)

model_file = open(model, "r")
first_line = model_file.readline().split(",")

log_prior_dict = {}
log_prior_dict["positive"] = float(first_line[0])
log_prior_dict["negative"] = float(first_line[1])
log_prior_dict["truthful"] = float(first_line[2])
log_prior_dict["deceptive"] = float(first_line[3].strip())

likelihood_positive = {}
likelihood_negative = {}
likelihood_truthful = {}
likelihood_deceptive = {}

for line in model_file:
    class_list = line.split(",")
    if class_list[0] == "positive":
        likelihood_positive[class_list[1]] = float(class_list[2].strip())
    elif class_list[0] == "negative":
        likelihood_negative[class_list[1]] = float(class_list[2].strip())
    elif class_list[0] == "truthful":
        likelihood_truthful[class_list[1]] = float(class_list[2].strip())
    elif class_list[0] == "deceptive":
        likelihood_deceptive[class_list[1]] = float(class_list[2].strip())

model_file.close()

likelihood = {}
likelihood["positive"] = likelihood_positive
likelihood["negative"] = likelihood_negative
likelihood["truthful"] = likelihood_truthful
likelihood["deceptive"] = likelihood_deceptive

sentiment_list = ["positive", "negative", "truthful", "deceptive"]
file_class_dict = {}

for file in file_tokens_dict:
    tokens_list = file_tokens_dict[file]
    sum_class = {}

    for sentiment in sentiment_list:
        sum_class[sentiment] = log_prior_dict[sentiment]
        for token in tokens_list:
            if token in likelihood[sentiment]:
                sum_class[sentiment] += likelihood[sentiment][token]

    if sum_class["positive"] >= sum_class["negative"]:
        file_class_dict[file] = ["positive"]
    else:
        file_class_dict[file] = ["negative"]

    if sum_class["truthful"] >= sum_class["deceptive"]:
        file_class_dict[file].append("truthful")
    else:
        file_class_dict[file].append("deceptive")

# num_test_file = len(file_class_dict)
# num_correct = 0
#
# for file in file_class_dict:
#     predict_class1 = file_class_dict[file][0]
#     predict_class2 = file_class_dict[file][1]
#     if file in test_data[predict_class1] and file in test_data[predict_class2]:
#         num_correct += 1
#
# print(num_correct/num_test_file)

file_output = open(output, "w")
for file in file_class_dict:
    file_output.writelines(" ".join([file_class_dict[file][1], file_class_dict[file][0], file]) + "\n")
