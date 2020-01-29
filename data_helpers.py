import numpy as np
import re
import keras.preprocessing.sequence as s
import pandas as pd


def clean_sentence(text):
    text = text.lower()
    # Clean the sentence
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    # text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

'''
def load_data(path):
    lines = [line.strip() for line in open(path)]  # Get all lines in the file
    all_sentences = []
    all_relations = []
    all_entity1 = []
    all_entity2 = []

    for idx in range(0, len(lines), 2):
        # example:
        # 1 "The system as described above has its greatest application in an arrayed configuration of antenna elements ."
        # Component-Whole(elements,configuration)

        sentence_id = lines[idx].split("\"")[0].strip()  # id of the sentence
        sentence = lines[idx].replace(sentence_id, sentence_id + "\t")
        sentence = sentence.split("\t")[1]
        sentence = clean_sentence(sentence)

        # example:
        # sentence:
        # the system as described above has its greatest application in an arrayed configuration of antenna elements
        # id: 1

        # dealing with relation and words
        relation = lines[idx + 1].split("(")[0].strip()  # relation in the sentence
        temp = lines[idx + 1].split("(")[1].strip(")")
        word1 = temp.split(",")[0].lower()
        word2 = temp.split(",")[1].lower()
        # example:
        # relation: Component-Whole
        # word1: elements
        # word2: configuration

        if word1 in word2 and word2 is not word1:
            sentence = sentence.replace(" " + word2 + " ", "  <e2>" + word2 + "</e2>  ", 1)
            sentence = sentence.replace(" " + word1 + " ", "  <e1>" + word1 + "</e1>  ", 1)
        else:
            sentence = sentence.replace(" " + word1 + " ", "  <e1>" + word1 + "</e1>  ", 1)
            sentence = sentence.replace(" " + word2 + " ", "  <e2>" + word2 + "</e2>  ", 1)
        sentence = sentence.replace("<e1>", "<e1> ")
        sentence = sentence.replace("</e1>", " <e1>")
        sentence = sentence.replace("<e2>", "<e1> ")
        sentence = sentence.replace("</e2>", " <e1>")
        # example:
        # sentence:
        # ' the system as described above has its greatest application in an arrayed  <e1> configuration <e1>  of antenna  <e1> elements <e1>  '

        all_sentences.append(sentence)
        all_relations.append(relation)
        all_entity1.append(word1)
        all_entity2.append(word2)

    # print(all_sentences[0], all_relations[0], all_entity1[0], all_entity2[0]
    return all_sentences, all_relations, all_entity1, all_entity2
'''

def load_data(path):
    sentences, relations = [], []
    to_replace = [("\"", ""), ("\n", ""), ("<", " <"), (">", "> ")]
    last_was_sentence = False
    for line in open(path):
        sl = line.split("\t")
        if last_was_sentence:
            relations.append(sl[0].split("(")[0].replace("\n", ""))
            last_was_sentence = False
        if sl[0].isdigit():
            sent = sl[1]
            for rp in to_replace:
                sent = sent.replace(rp[0], rp[1])
            sent = clean_sentence(sent)
            sentences.append(sent)
            last_was_sentence = True

    return sentences, relations


def tokenize_data(tokenizer, all_sentences):
    all_tokens = []  # all sentence tokens
    max_length = 128  # max sentence length

    # Use BERT to tokenize, first split the sentence into separate words
    for sentence in all_sentences:

        sentence = sentence.replace('/e1', 'e1')
        sentence = sentence.replace('e2', 'e1')
        sentence = sentence.replace('/e2', 'e1')
        sentence_trunks = sentence.split("<e1>")
        tokens = ["[CLS]"]
        for i, trunk in enumerate(sentence_trunks):
            part_token = tokenizer.tokenize(trunk)
            tokens.extend(part_token)
            if i == 0 or i == 1:
                tokens.append('[EN1]')
            elif i == 2 or i == 3:
                tokens.append('[EN2]')

        tokens.append("[SEP]")
        all_tokens.append(tokens)
        # print(tokens)

    # Align data
    token_ids = s.pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in all_tokens],
                              maxlen=max_length, dtype="long", truncating="post", padding="post")
    '''
    example:
    [101  1996  2291  2004  2649  2682  2038  2049  4602  4646  1999  2019
     9140  2098     1  9563     1  1997 13438     2  3787     2     2   102
     0     0     0     0     0     0     0     0     0     0     0     0
     ...
     0     0     0     0     0     0]
     '''

    # Use a matrix to illustrate position of entity
    entity1_positions, entity2_positions = find_entity_position(all_tokens, max_length)

    return all_tokens, token_ids, entity1_positions, entity2_positions


def find_entity_position(all_tokens, max_length):

    # Get all positions of four labels
    marks = [[i for i, x in enumerate(token) if x == '[EN1]' or x == '[EN2]'] for token in all_tokens]
    e1 = []
    e2 = []
    for token in all_tokens:
        temp1 = np.zeros((max_length,))
        temp2 = np.zeros((max_length,))
        flag1 = False
        flag2 = False
        # Entity can be more than one word
        for i, x in enumerate(token):
            if x == '[EN1]':
                if flag1:
                    flag1 = False
                else:
                    flag1 = True
            elif x == '[EN2]':
                if flag2:
                    flag2 = False
                else:
                    flag2 = True
            elif flag1:
                temp1[i] = 1
            elif flag2:
                temp2[i] = 1

        e1.append(temp1)
        e2.append(temp2)

    return e1, e2