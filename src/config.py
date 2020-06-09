import os


class Config:
    target = 'category'
    assert target in ['category', 'subcategory']

    num_batches = 3000  # Number of batches to train
    num_batches_show_loss = 50  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 100
    batch_size = 64
    learning_rate = 0.001
    validation_proportion = 0.2
    num_workers = 4  # Number of workers for data loading
    # Whether try to load checkpoint
    load_checkpoint = os.environ['LOAD_CHECKPOINT'] == '1' if 'LOAD_CHECKPOINT' in os.environ else True
    num_words_a_sentence = 30
    num_sentences_an_abstract = 4
    num_words_a_news = 100  # used when not hierarchical
    word_freq_threshold = 3
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 31752
    num_categories = 17
    num_subcategories = 256
    word_embedding_dim = 300
    word_gru_hidden_size = 100
    word_query_vector_dim = 200
    sentence_gru_hidden_size = 100
    sentence_query_vector_dim = 200

    hierarchical = os.environ['HIERARCHICAL'] == '1' if 'HIERARCHICAL' in os.environ else True
    pack_gru = os.environ['PACK_GRU'] == '1' if 'PACK_GRU' in os.environ else True
    connect_hidden = os.environ['CONNECT_HIDDEN'] == '1' if 'CONNECT_HIDDEN' in os.environ else True
    aggregate_mode = os.environ['AGGREGATE_MODE'] if 'AGGREGATE_MODE' in os.environ else 'attention'
    assert aggregate_mode in ['attention', 'last_hidden', 'average', 'max']
