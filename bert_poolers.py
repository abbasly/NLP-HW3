import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, BertModel,
    BertEmbeddings, BertEncoder, BertForSequenceClassification, BertPooler, BertAttention
)


class MeanMaxTokensBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.new_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.new_activation = nn.Tanh()

    def forward(self, hidden_states, *args, **kwargs):
        mean_tokens = torch.mean(hidden_states, 1) # (B, L, H) -> (B, H)
        max_tokens = torch.max(hidden_states, 1)[0] # (B, L, H) -> (B, H)
        pooled_output = torch.cat((mean_tokens, max_tokens), 1) # (B, H) -> (B, 2H)
        pooled_output = self.new_dense(pooled_output)
        pooled_output = self.new_activation(pooled_output)
        return pooled_output


class MyBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.new_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.new_activation = nn.Tanh()
        self.attention_pooler = BertAttention(config)

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = self.attention_pooler(hidden_states)[0]
        first_token = hidden_states[:, 0] # get CLS
        avg_of_word_tokens = torch.mean(hidden_states[:,1:]) # get avg of word representations
        new_result = torch.cat((first_token, avg_of_word_tokens), 1) # concat above tensors
        pooled_output = self.new_dense(new_result)
        pooled_output = self.new_activation(pooled_output)
        return pooled_output



class MyBertConfig(BertConfig):
    def __init__(self, pooling_layer_type="CLS", **kwargs):
        super().__init__(**kwargs)
        self.pooling_layer_type = pooling_layer_type


class MyBertModel(BertModel):

    def __init__(self, config: MyBertConfig):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        if config.pooling_layer_type == "CLS":
            # See src/transformers/models/bert/modeling_bert.py#L610
            # at huggingface/transformers (9f43a425fe89cfc0e9b9aa7abd7dd44bcaccd79a)
            self.pooler = BertPooler(config)
        elif config.pooling_layer_type == "MEAN_MAX":
            self.pooler = MeanMaxTokensBertPooler(config)
        elif config.pooling_layer_type == "MINE":
            self.pooler = MyBertPooler(config)
        else:
            raise ValueError(f"Wrong pooling_layer_type: {config.pooling_layer_type}")

        self.init_weights()

    @property
    def pooling_layer_type(self):
        return self.config.pooling_layer_type


class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.bert.pooling_layer_type in ["CLS", "MEAN_MAX", "MINE"]:
            return super().forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states, return_dict
            )
        else:
            raise Exception(f"Wrong pooling layer type: {self.bert.pooling_layer_type}")
