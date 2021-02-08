export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased

curl -L 'https://drive.google.com/uc?export=download&id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
curl -L 'https://drive.google.com/uc?export=download&id=1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
curl -L 'https://drive.google.com/uc?export=download&id=1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp

python preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
python preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt

cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
