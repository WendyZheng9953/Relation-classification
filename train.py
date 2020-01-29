import tensorflow as tf
import data_helpers
import utils
import numpy as np

from bert import tokenization
from configure import FLAGS
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from Average_Connect_Layer import AverageConnectLayer
from BERT_Layer import BERTLayer
from sklearn.metrics import precision_recall_fscore_support

max_length = 128

def train():
    # Load train data
    all_sentences, all_relations = data_helpers.load_data(FLAGS.train_path)
    # Load test data
    all_test_sentences, all_test_relations = data_helpers.load_data(FLAGS.test_path)

    # Set Tokenizer for BERT
    tokenization.validate_case_matches_checkpoint(True, utils.BERT_INIT_CHKPNT)
    tokenizer = tokenization.FullTokenizer(vocab_file=utils.BERT_VOCAB, do_lower_case=True)

    # Tokenize training data
    all_tokens, token_ids, entity1_positions, entity2_positions = data_helpers.tokenize_data(tokenizer, all_sentences)
    all_test_tokens, test_token_ids, test_entity1_positions, test_entity2_positions = data_helpers.tokenize_data(tokenizer, all_test_sentences)

    '''
    all_sentences: 
         the system as described above has its greatest application in an arrayed  
         <e1> configuration <e1>  of antenna  <e1> elements <e1>  
    all_relation:
         Component-Whole
    all_entity1:
        elements
    all_entity2:
        configuration
    all_tokens:
        ['[CLS]', 'the', 'system', 'as', 'described', 'above', 'has', 'its', 'greatest', 'application', 'in', 'an', 
        'array', '##ed', '[EN1]', 'configuration', '[EN1]', 'of', 'antenna', '[EN2]', 'elements', '[EN2]', '[SEP]']
    token_ids:
        [  101  1996  2291  2004  2649  2682  2038  2049  4602  4646  1999  2019
          9140  2098     1  9563     1  1997 13438     2  3787     2   102     0 ...
             0     0     0     0     0]
    entity1_positions:
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. ...
         0. 0.]
    entity2_positions:
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ...
         0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    '''
    # Define Tensors
    token_id = Input(shape=(max_length,))
    token_mask = Input(shape=(max_length,))
    token_segment = Input(shape=(max_length,))

    # import a BERT Layer and get output
    BERT_input = [token_id, token_mask, token_segment]
    BERT_output = BERTLayer(n_fine_tune_layers=0)(BERT_input)

    # Define Tensors
    entity1_pos = Input(shape=(max_length,))
    entity2_pos = Input(shape=(max_length,))

    # import AverageAndConnect Layer and get output
    average_input = [BERT_output, entity1_pos, entity2_pos]
    average_output = AverageConnectLayer()(average_input)

    # Last Layer
    average_output = Dropout(0.1)(average_output)
    out = Dense(units=10, activation="softmax")(average_output)

    all_input = [token_id, token_mask, token_segment, entity1_pos, entity2_pos]
    model = Model(all_input, out)
    model.compile(optimizer=Adam(lr=2e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # The actual training process
    # Get entity positions
    encoder = LabelEncoder()
    encoder.fit(all_relations)
    train_relations_temp = encoder.transform(all_relations)
    train_relations = np.expand_dims(train_relations_temp, 2)
    test_relations_temp = encoder.transform(all_test_relations)
    test_relations = np.expand_dims(test_relations_temp, 2)

    entity_positions = [[float(i > 0) for i in ii] for ii in token_ids]
    test_entity_positions = [[float(i > 0) for i in ii] for ii in test_token_ids]

    # Train
    batch_size = 16

    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        history = model.fit(
            [token_ids, np.array(entity_positions), np.zeros_like(token_ids), np.array(entity1_positions), np.array(entity2_positions)],
            train_relations,
            batch_size=batch_size,
            validation_data=(
                [test_token_ids,
                 np.array(test_entity_positions),
                 np.zeros_like(test_token_ids),
                 np.array(test_entity1_positions),
                 np.array(test_entity2_positions)],
                 test_relations),
            epochs=5,
            verbose=1)

    predictions = model.predict([test_token_ids, np.array(test_entity_positions), np.zeros_like(test_token_ids), np.array(test_entity1_positions), np.array(test_entity2_positions)])
    precision, recall, fscore, support = precision_recall_fscore_support(all_test_relations, np.argmax(predictions, -1))
    score = [s for i, s in enumerate(fscore) if i != encoder.transform(['Other'])[0]]
    np.mean(score)


if __name__ == "__main__":
    train()