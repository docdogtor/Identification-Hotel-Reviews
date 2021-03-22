import sys
import glob
import collections
import os
import math

input_path = sys.argv[1]

file_list = glob.glob(os.path.join(input_path,'*/*/*/*.txt'))
training_data = collections.defaultdict(list)

for file in file_list:
    class1, class2, fold, file_name = file.split("/")[-4:]
    training_data[class1 + class2].append(file)

def new_word_format(word):
    new_word = word.lower().strip()
    char_list = []
    for char in new_word:
        if char not in ". , /":
            char_list.append(char)
    return "".join(char_list)

def read_words_from_files(sentiment, file_data):
    word_dict = {}
    word_counter = 0
    num_file = 0
    for key in file_data:
        if sentiment in key:
            file_list = file_data[key]
            for file_text in file_list:
                num_file += 1
                current_file = open(file_text, "r")
                for line in current_file:
                    word_list = line.split(" ")
                    for word in word_list:
                        word = new_word_format(word)
                        word_counter += 1
                        word_dict[word] = word_dict.get(word, 0) + 1
                current_file.close()
    return word_dict, word_counter, num_file

# ...WordDict{[word]:this word's number}, num...Word int(), num...File int()
word_dict_total, num_word_toal, num_file_total = read_words_from_files("",training_data)
word_dict_positive, num_word_positive, num_file_positive = read_words_from_files("positive", training_data)
word_dict_negative, num_word_negative, num_file_negative = read_words_from_files("negative", training_data)
word_dict_truthful, num_word_truthful, num_file_truthful = read_words_from_files("truthful", training_data)
word_dict_deceptive, num_word_deceptive, num_file_deceptive = read_words_from_files("deceptive", training_data)
abs_V = len(word_dict_total)

# logprior c
log_prior_positive = math.log(num_file_positive/num_file_total)
log_prior_negative = math.log(num_file_negative/num_file_total)
log_prior_truthful = math.log(num_file_truthful/num_file_total)
log_prior_deceptive = math.log(num_file_deceptive/num_file_total)
#print(log_prior_positive, log_prior_negative, log_prior_truthful, log_prior_deceptive)

# calculate p(w/c) OR loglikelihood[w, c]
def likelihood(word_dict_total, word_dict_sentiment, num_word_sentiment, abs_V):
    likelihood_dict = {}
    for word in word_dict_total:
        likelihood_dict[word] = math.log((word_dict_sentiment.get(word, 0) + 1)/(num_word_sentiment+abs_V))
    return likelihood_dict

likelihood_positive = likelihood(word_dict_total, word_dict_positive, num_word_positive, abs_V)
likelihood_negative = likelihood(word_dict_total, word_dict_negative, num_word_negative, abs_V)
likelihood_truthful = likelihood(word_dict_total, word_dict_truthful, num_word_truthful, abs_V)
likelihood_deceptive = likelihood(word_dict_total, word_dict_deceptive, num_word_deceptive, abs_V)

model_file = "nbmodel.txt"

fileOut = open(model_file, "w")
line = ",".join([str(log_prior_positive),str(log_prior_negative), str(log_prior_truthful), str(log_prior_deceptive)])
fileOut.writelines(line + "\n")

for word in likelihood_positive:
    elementList = ["positive", word, str(likelihood_positive[word])]
    line = ",".join(elementList)
    fileOut.writelines(line + "\n")

for word in likelihood_negative:
    elementList = ["negative", word, str(likelihood_negative[word])]
    line = ",".join(elementList)
    fileOut.writelines(line + "\n")

for word in likelihood_truthful:
    elementList = ["truthful", word, str(likelihood_truthful[word])]
    line = ",".join(elementList)
    fileOut.writelines(line + "\n")

for word in likelihood_deceptive:
    elementList = ["deceptive", word, str(likelihood_deceptive[word])]
    line = ",".join(elementList)
    fileOut.writelines(line + "\n")

fileOut.close()