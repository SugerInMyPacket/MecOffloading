import torch.nn as nn

class DQNTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, output_size):
        super(DQNTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.embedding(src)  # 应用位置编码前或在嵌入层之后添加位置信息
        out = self.transformer_encoder(src)
        q_values = self.fc_out(out[:, -1, :])  # 假设我们只关心序列最后一个元素的输出
        return q_values
