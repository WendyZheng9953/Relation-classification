import numpy as np

BERT_VOCAB = "bert_model/vocab.txt"
BERT_INIT_CHKPNT = "bert_model/bert_model.ckpt"
BERT_CONFIG = "bert_model/bert_config.json"

class2label = {'Other': 0,
               'Message-Topic': 1,
               'Product-Producer': 2,
               'Instrument-Agency': 3,
               'Entity-Destination': 4,
               'Cause-Effect': 5,
               'Component-Whole': 6,
               'Entity-Origin': 7,
               'Member-Collection': 8,
               'Content-Container': 9}

label2class = {0: 'Other',
               1: 'Message-Topic',
               2: 'Product-Producer',
               3: 'Instrument-Agency',
               4: 'Entity-Destination',
               5: 'Cause-Effect',
               6: 'Component-Whole',
               7: 'Entity-Origin',
               8: 'Member-Collection',
               9: 'Content-Container'}

