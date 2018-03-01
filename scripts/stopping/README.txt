1. download trec.tgz, unzip it. download CuratedTrec-test-lstm.preds.txt, CuratedTrec-test.txt
(for squad dataset, download squad.tgz, SQuAD-v1.1-dev-lstm.preds.txt, SQuAD-v1.1-dev.txt accordingly.)

2. run prepare_data.py, e.g.
python prepare_data.py -p CuratedTrec-test-lstm.preds.txt -a CuratedTrec-test.txt -f trec -r records

3. start train the model (use default train args)
python linear_model.py -r records


