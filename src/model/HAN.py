import torch
import torch.nn as nn
from model.aggregator import Aggregator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAN(torch.nn.Module):
    """
    HAN network.
    """

    def __init__(self, config, pretrained_word_embedding=None):
        super(HAN, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.word_aggregator = Aggregator(
            config.word_embedding_dim, config.word_gru_hidden_size, config.word_query_vector_dim, config.aggregate_mode)
        self.sentence_aggregator = Aggregator(
            config.word_gru_hidden_size * 2, config.sentence_gru_hidden_size, config.sentence_query_vector_dim, config.aggregate_mode)
        self.final_linear = nn.Linear(
            config.sentence_gru_hidden_size *
            2 if config.hierarchical else config.word_gru_hidden_size * 2,
            config.num_categories if config.target == 'category' else config.num_subcategories)

    def forward(self, minibatch):
        """
        Args:
            minibatch:
                hierarchical:
                    {
                        "title": Tensor(batch_size) * num_words_a_sentence,
                        "title_length": Tensor(batch_size)
                        "abstract": [Tensor(batch_size) * num_words_a_sentence] * num_sentences_an_abstract ,
                        "abstract_length": Tensor(batch_size)
                        "abstract_sentence_length": Tensor(batch_size) * num_sentences_an_abstract
                    }
                no hierarchical:
                    {
                        "news": Tensor(batch_size) * num_words_a_sentence,
                        "news_length": Tensor(batch_size)
                    }
        Returns:
            classification: batch_size, num_(sub)categories
        """
        if self.config.hierarchical:
            # batch_size, num_words_a_sentence, word_embedding_dim
            title_vector = self.word_embedding(
                torch.stack(minibatch["title"], dim=1).to(device))
            # batch_size
            title_length = minibatch["title_length"]
            # num_sentences_an_abstract, batch_size, num_words_a_sentence, word_embedding_dim
            abstract_vector = torch.stack([self.word_embedding(
                torch.stack(sent, dim=1).to(device)) for sent in minibatch["abstract"]], dim=0)
            # batch_size
            abstract_length = minibatch["abstract_length"]
            # num_sentences_an_abstract, batch_size
            abstract_sentence_length = torch.stack(
                minibatch["abstract_sentence_length"], dim=0)

            # 1 + num_sentences_an_abstract, batch_size, num_words_a_sentence, word_embedding_dim
            integrated_vector = torch.cat(
                (title_vector.unsqueeze(dim=0), abstract_vector), dim=0)
            # 1 + num_sentences_an_abstract, batch_size
            integrated_vector_sentence_length = torch.cat(
                (title_length.unsqueeze(dim=0), abstract_sentence_length), dim=0)
            # batch_size
            integrated_vector_length = abstract_length + 1

            real_batch_size = integrated_vector_length.size(0)
            # TODO
            integrated_vector_sentence_length[integrated_vector_sentence_length == 0] = 1
            temp = []
            hidden = torch.zeros(2, real_batch_size,
                                 self.config.word_gru_hidden_size).to(device)
            for x, y in zip(integrated_vector, integrated_vector_sentence_length):
                aggregated, hidden = self.word_aggregator(
                    x, y, hidden if self.config.connect_hidden else None)
                temp.append(aggregated)
            # batch_size, 1 + num_sentences_an_abstract, word_gru_hidden_size * 2
            word_aggregated_vector = torch.stack(temp, dim=1)
            # batch_size, sentence_gru_hidden_size * 2
            sentence_aggregated_vector, _ = self.sentence_aggregator(
                word_aggregated_vector, integrated_vector_length)
            # batch_size, num_(sub)categories
            classification = self.final_linear(
                sentence_aggregated_vector)  # TODO
            return classification
        else:
            # batch_size, num_words_a_news, word_embedding_dim
            news_vector = self.word_embedding(
                torch.stack(minibatch["news"], dim=1).to(device))
            # batch_size
            news_length = minibatch["news_length"]
            # batch_size, word_gru_hidden_size * 2
            word_aggregated_vector, _ = self.word_aggregator(
                news_vector, news_length)
            # batch_size, num_(sub)categories
            classification = self.final_linear(word_aggregated_vector)
            return classification
