import torch
import torch.nn as nn
import torch.nn.functional as F
from model.additive_attention import AdditiveAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Aggregator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, query_vector_dim, aggregate_mode):
        super(Aggregator, self).__init__()
        self.hidden_size = hidden_size
        self.bi_gru = nn.GRU(input_size, hidden_size,
                             batch_first=True,
                             bidirectional=True)
        self.additive_attention = AdditiveAttention(
            query_vector_dim, hidden_size * 2)
        self.aggregate_mode = aggregate_mode

    def forward(self, sequence, length, initial_hidden=None):
        """
        Args:
            sequence: batch_size, input_length, input_size
            length: batch_size
        Returns:
            (shape) batch_size, hidden_size * 2
        """
        total_length = sequence.size(1)
        real_batch_size = length.size(0)
        packed_sequence = pack_padded_sequence(
            sequence, length, batch_first=True, enforce_sorted=False)
        if initial_hidden is None:
            initial_hidden = torch.zeros(
                2, real_batch_size, self.hidden_size).to(device)
        # last_hidden: 2, batch_size, hidden_size
        output, last_hidden = self.bi_gru(packed_sequence, initial_hidden)
        # batch_size, input_length, 2 * hidden_size
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=total_length)
        # aggregated: batch_size, 2 * hidden_size
        if self.aggregate_mode == 'attention':
            aggregated = self.additive_attention(output)
        elif self.aggregate_mode == 'last_hidden':
            aggregated = last_hidden.transpose(
                0, 1).reshape(real_batch_size, -1)
        elif self.aggregate_mode == 'average':
            aggregated = torch.stack([x[:y].mean(dim=0)
                                      for x, y in zip(output, length)], dim=0)
        elif self.aggregate_mode == 'max':
            aggregated = F.max_pool1d(output.transpose(
                1, 2), kernel_size=output.size(1)).squeeze(dim=2)
        else:
            print('Wrong aggregate mode!')
            exit()
        return aggregated, last_hidden
