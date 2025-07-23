import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMSentimentModel(nn.Module):
    """LSTM情感分析模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, 
                 num_classes: int = 3, dropout: float = 0.3):
        """
        初始化LSTM情感分析模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数量（正面/负面/中性）
            dropout: Dropout比率
        """
        super(LSTMSentimentModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            lengths: 序列长度 [batch_size]
            
        Returns:
            分类概率 [batch_size, num_classes]
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM前向传播
        if lengths is not None:
            # 使用pack_padded_sequence优化变长序列处理
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, _) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出（双向LSTM的拼接）
        # hidden: [num_layers*2, batch_size, hidden_dim]
        # 取最后一层的前向和后向隐藏状态
        forward_hidden = hidden[-2]  # 前向LSTM的最后一层
        backward_hidden = hidden[-1]  # 后向LSTM的最后一层
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Dropout
        final_hidden = self.dropout(final_hidden)
        
        # 分类
        output = self.fc(final_hidden)  # [batch_size, num_classes]
        
        return output
    
    def init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    # 全连接层权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif 'embedding' in name:
                    # 词嵌入权重使用正态分布初始化
                    nn.init.normal_(param, mean=0, std=0.1)
            elif 'bias' in name:
                # 偏置初始化为0
                nn.init.constant_(param, 0)


class TextCNN(nn.Module):
    """TextCNN情感分析模型（备选方案）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_filters: int = 100, filter_sizes: list = [3, 4, 5],
                 num_classes: int = 3, dropout: float = 0.3):
        """
        初始化TextCNN模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            num_filters: 每种卷积核的数量
            filter_sizes: 卷积核大小列表
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=size)
            for size in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # 卷积和池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, conv_seq_len]
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # Dropout和分类
        output = self.dropout(concatenated)
        output = self.fc(output)
        
        return output