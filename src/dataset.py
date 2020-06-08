from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from config import Config
from itertools import chain


class HANDataset(Dataset):
    def __init__(self, news_path):
        super(HANDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            converters={
                'title': literal_eval,
                'abstract': literal_eval
            })

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        row = self.news_parsed.iloc[idx]

        if Config.hierarchical:
            def pad_sentence(sent):
                """
                Returns:
                    sent:
                    length: Num of words in the sentence
                """
                sent = sent[:Config.num_words_a_sentence]
                length = len(sent)
                if len(sent) < Config.num_words_a_sentence:
                    sent += [0] * (Config.num_words_a_sentence - len(sent))
                return sent, length

            def pad_document(doc):
                """
                Returns:
                    doc:
                    length: Num of sentences in the document
                    sentence_length: Num of words in each sentences of the document
                """
                doc = doc[:Config.num_sentences_an_abstract]
                length = len(doc)
                if len(doc) < Config.num_sentences_an_abstract:
                    doc += [[]] * (Config.num_sentences_an_abstract - len(doc))
                doc, sentence_length = zip(*[pad_sentence(x) for x in doc])
                return doc, length, sentence_length
            title, title_length = pad_sentence(row.title)
            abstract, abstract_length, abstract_sentence_length = pad_document(
                row.abstract)
            if not Config.pack_gru:
                title_length = Config.num_words_a_sentence
                abstract_length = Config.num_sentences_an_abstract
                abstract_sentence_length = [
                    Config.num_words_a_sentence] * Config.num_sentences_an_abstract
            item = {
                "id": row.id,
                "category": row.category,
                "subcategory": row.subcategory,
                "title": title,
                "title_length": title_length,
                "abstract": abstract,
                "abstract_length": abstract_length,
                "abstract_sentence_length": abstract_sentence_length
            }
        else:
            news = (row.title + list(chain.from_iterable(row.abstract))
                    )[:Config.num_words_a_news]
            news_length = len(news)
            if len(news) < Config.num_words_a_news:
                news += [0] * (Config.num_words_a_news - len(news))
            if not Config.pack_gru:
                news_length = Config.num_words_a_news
            item = {
                "id": row.id,
                "category": row.category,
                "subcategory": row.subcategory,
                "news": news,
                "news_length": news_length
            }

        return item
