import torch
import torch.nn as nn
from transformers import BertModel,BertConfig,AutoTokenizer
import  os

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTBiLSTMClassifier, self).__init__()
        # # 获取当前文件的绝对路径
        # current_file_path = os.path.abspath(__file__)
        # # 获取当前文件所在的目录
        # current_directory = os.path.dirname(current_file_path)
        # # 构建配置文件的完整路径
        # config_file_path = os.path.join(current_directory, 'bert_config/config.json')
        # self.bert = BertModel.from_pretrained('microsoft/codebert-base', config = config_file_path)
        self.bert = BertModel.from_pretrained('microsoft/codebert-base')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.bilstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def getTokenizer(self):
        return self.tokenizer
    def forward(self, input_ids, attention_mask):
        batch_size, seq_count, max_seq_length = input_ids.size()

        # Reshape input_ids and attention_mask to combine batch dimension and sequence dimension
        input_ids = input_ids.view(-1, max_seq_length)
        attention_mask = attention_mask.view(-1, max_seq_length)

        # BERT encoding
        bert_encoded = self.bert(input_ids, attention_mask).last_hidden_state
        bert_encoded = bert_encoded.mean(dim=1)
        # Reshape bert_encoded back to original shape
        bert_encoded = bert_encoded.view(batch_size, -1, 768)

        # LSTM encoding
        lstm_output, _ = self.bilstm(bert_encoded)

        # Classification layer
        logits = self.fc(lstm_output)

        return logits
