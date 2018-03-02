1. download trec.tgz, unzip it. download CuratedTrec-test-lstm.preds.txt, CuratedTrec-test.txt
(for squad dataset, download squad.tgz, SQuAD-v1.1-dev-lstm.preds.txt, SQuAD-v1.1-dev.txt accordingly.)

2. run prepare_data.py, e.g.
python prepare_data.py -p CuratedTrec-test-lstm.preds.txt -a CuratedTrec-test.txt -f trec -r records --regex

3. start train the model (use default train args)
python linear_model.py -r records

For squad, you can use dev-2k for training model and test on dev-1k

python scripts/stopping/prepare_data.py -p data/SQuAD-v1.1-dev-2k-multitask-lstm.preds.txt -a data/SQuAD-v1.1-dev-2k.txt -f data/squad -r data/dev2k_records

python scripts/stopping/linear_model.py -p data/SQuAD-v1.1-dev-2k-multitask-lstm.preds.txt -a data/SQuAD-v1.1-dev-2k.txt -f data/squad -r data/dev2k_records

python scripts/stopping/eval_model.py -p data/SQuAD-v1.1-dev-1k-multitask-lstm.preds.txt -a data/SQuAD-v1.1-dev-1k.txt -f data/squad -m linear_model_84.40610428916152.mdl


