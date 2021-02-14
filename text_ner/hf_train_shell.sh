export BERT_MODEL=bert-base-multilingual-cased

python run_ner.py \
--model_name_or_path $BERT_MODEL \
--train_file data/train.txt \
--validation_file data/dev.txt \
--output_dir test-ner \
--do_train \
--do_eval

# --model_name_or_path bert-base-multilingual-cased --train_file data/train.txt --validation_file data/dev.txt --output_dir test-ner --do_train --do_eval
