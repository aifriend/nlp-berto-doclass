import re
import stanza

import pandas as pd
from texthero.preprocessing import remove_digits, remove_diacritics, remove_whitespace, remove_round_brackets, \
    remove_curly_brackets, remove_square_brackets, remove_angle_brackets, remove_brackets, remove_html_tags, \
    remove_urls, remove_punctuation

from warnings import simplefilter

from ClassFile import ClassFile

simplefilter(action='ignore', category=FutureWarning)

nlp = stanza.Pipeline(lang='es', processors='tokenize')


def remove_regex(data, reg):
    return list(map(lambda x: re.sub(reg, '', x), data))


def sentence_splitter(doc_list):
    doc_sentences = list()
    for idx, doc in enumerate(doc_list):
        print(f"{idx}* ", end='')
        nlp_doc = nlp(doc)
        for sentence in nlp_doc.sentences:
            doc_sentences.append(sentence.text)
    return doc_sentences


df = pd.read_csv(
    "dataset/first_section.csv",
    header=None, sep=',', names=['CONTENT', 'CLASS']
)

pd.set_option('display.max_colwidth', None)
# .pipe(remove_digits) \
# .pipe(remove_punctuation) \
df['CONTENT'] = df['CONTENT'] \
    .pipe(remove_diacritics) \
    .pipe(remove_html_tags) \
    .pipe(remove_urls) \
    .pipe(remove_square_brackets) \
    .pipe(remove_square_brackets) \
    .pipe(remove_square_brackets) \
    .pipe(remove_square_brackets) \
    .pipe(remove_round_brackets) \
    .pipe(remove_angle_brackets) \
    .pipe(remove_curly_brackets) \
    .pipe(remove_brackets) \
    .pipe(remove_whitespace)

print(df.values[0])
df.to_csv("first_section_clean.tsv", sep='\t')
exit(-1)

#content_list = remove_regex(content_list, '[-<>]')

#content_list = sentence_splitter(content_list)

# print("=====================================")
# for st in content_list:
#     print(f"* {st}", end="\n")
#     print()
#     if len(st) >= 90:
#         ClassFile.to_txtfile(f"{st}\n", "process_dataset.txt", "a")
