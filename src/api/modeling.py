import ast
import logging
import gensim
import pickle
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
gensim_logger = logging.getLogger('gensim')
smart_open_logger = logging.getLogger('smart_open')
smart_open_logger.setLevel(logging.WARNING)
gensim_logger.setLevel(logging.WARNING)


def preprocess_data_to_bow(dataSI):
    # Read MK data
    dataMK = pd.read_csv('./src/api/data/preprocessed_dataMK.csv')

    # Process MK data - convert string representation of list to actual list
    dataMK_tokens = []
    for x in dataMK['deskripsiMK']:
        try:
            # Try to evaluate the string as a literal list
            tokens = ast.literal_eval(str(x))
            if isinstance(tokens, list):
                dataMK_tokens.append(tokens)
            else:
                dataMK_tokens.append(str(x).split())
        except (ValueError, SyntaxError):
            # If evaluation fails, split the string into words
            dataMK_tokens.append(str(x).split())

    # Ensure dataSI is properly tokenized
    dataSI_tokens = dataSI if isinstance(dataSI[0], list) else [
        tokens.split() for tokens in dataSI]

    logger.debug(f"Data MK first document: {dataMK_tokens[0][:10]}...")
    logger.debug(f"Data SI first document: {dataSI_tokens[0][:10]}...")

    # Create Dictionary instances from tokenized documents
    dict_MK = Dictionary(dataMK_tokens)
    dict_SI = Dictionary(dataSI_tokens)

    logger.debug(f"Dictionary MK size: {len(dict_MK)}")
    logger.debug(f"Dictionary SI size: {len(dict_SI)}")

    # Convert the tokenized data to bag-of-words representation
    corpus_MK = [dict_MK.doc2bow(doc) for doc in dataMK_tokens]
    corpus_SI = [dict_SI.doc2bow(doc) for doc in dataSI_tokens]

    logger.debug(f"Corpus MK size: {len(corpus_MK)}")
    logger.debug(f"Corpus SI size: {len(corpus_SI)}")

    # Validate output types
    if not isinstance(corpus_MK, list):
        raise ValueError(f"Unexpected type for corpus_MK: {type(corpus_MK)}")
    if not isinstance(corpus_SI, list):
        raise ValueError(f"Unexpected type for corpus_SI: {type(corpus_SI)}")

    return dict_MK, dict_SI, corpus_MK, corpus_SI, dataMK_tokens, dataSI_tokens


def evaluate_LDA(doc_bow, dictionary, num_topics, text, alpha, beta, top_n_words=30):
    lda_model = models.LdaModel(
        corpus=[doc_bow],
        num_topics=num_topics,
        id2word=dictionary,
        passes=50,
        random_state=0,
        alpha=alpha,
        eta=beta
    )

    # Extract the phi (topic-word distribution)
    phi = lda_model.get_topics()

    # Extract the words for each topic
    topic_words = []
    word_list = [lda_model.id2word[i] for i in range(len(lda_model.id2word))]
    for topic_distribution in phi:
        top_word_indices = np.argsort(topic_distribution)[-top_n_words:][::-1]
        top_words = [word_list[i] for i in top_word_indices]
        topic_words.append(top_words)

    # Extract the theta (document-topic distribution)
    doc_topics = lda_model.get_document_topics(doc_bow)
    topic_distribution = [0.0] * num_topics
    for topic_id, prob in doc_topics:
        topic_distribution[topic_id] = prob

    return lda_model, topic_words, topic_distribution
