import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.embedding_size = embedding_size

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.embedding_size)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_sequence_length=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position_indices = torch.arange(max_sequence_length).unsqueeze(1)
        dimension_indices = torch.arange(0, embedding_size, 2)
        angle_rates = torch.exp(dimension_indices * (-math.log(10000.0) / embedding_size))

        positional_matrix = torch.zeros(max_sequence_length, embedding_size)
        positional_matrix[:, 0::2] = torch.sin(position_indices * angle_rates)
        positional_matrix[:, 1::2] = torch.cos(position_indices * angle_rates)

        self.register_buffer('positional_matrix', positional_matrix.unsqueeze(0))

    def forward(self, token_embeddings):
        sequence_length = token_embeddings.size(1)
        return self.dropout(token_embeddings + self.positional_matrix[:, :sequence_length])


class BidirectionalAttention(nn.Module):
    def __init__(self, embedding_size, number_of_heads, dropout=0.1):
        super().__init__()
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_size // number_of_heads
        self.scale = math.sqrt(self.head_dimension)

        self.query_projection = nn.Linear(embedding_size, embedding_size)
        self.key_projection = nn.Linear(embedding_size, embedding_size)
        self.value_projection = nn.Linear(embedding_size, embedding_size)
        self.output_projection = nn.Linear(embedding_size, embedding_size)

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, sequence_length, embedding_size = hidden_states.shape

        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        queries = queries.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)
        keys = keys.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)
        values = values.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)

        return self.output_projection(attended_values)


class FeedForwardBlock(nn.Module):
    def __init__(self, embedding_size, feedforward_size, dropout=0.1):
        super().__init__()
        self.expand = nn.Linear(embedding_size, feedforward_size)
        self.compress = nn.Linear(feedforward_size, embedding_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        return self.compress(self.dropout(self.activation(self.expand(hidden_states))))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, number_of_heads, feedforward_size, dropout=0.1):
        super().__init__()
        self.attention = BidirectionalAttention(embedding_size, number_of_heads, dropout)
        self.feedforward = FeedForwardBlock(embedding_size, feedforward_size, dropout)
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.feedforward_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(self.attention_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.dropout(attention_output)

        feedforward_output = self.feedforward(self.feedforward_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(feedforward_output)

        return hidden_states


class BidirectionalLM(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size=768,
        number_of_layers=12,
        number_of_heads=12,
        feedforward_size=3072,
        max_sequence_length=512,
        dropout=0.1,
        mask_token_id=103
    ):
        super().__init__()

        self.mask_token_id = mask_token_id
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.token_embedding = TokenEmbedding(vocabulary_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, max_sequence_length, dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_size, number_of_heads, feedforward_size, dropout)
            for _ in range(number_of_layers)
        ])

        self.output_norm = nn.LayerNorm(embedding_size)
        self.token_predictor = nn.Linear(embedding_size, vocabulary_size)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, attention_mask=None):
        embedded_tokens = self.token_embedding(token_ids)
        hidden_states = self.positional_encoding(embedded_tokens)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)

        hidden_states = self.output_norm(hidden_states)
        token_logits = self.token_predictor(hidden_states)

        return token_logits

    def get_confidence_scores(self, token_logits):
        token_probabilities = torch.softmax(token_logits, dim=-1)
        confidence_scores, predicted_tokens = token_probabilities.max(dim=-1)
        return confidence_scores, predicted_tokens

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
