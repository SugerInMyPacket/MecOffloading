import torch.nn as nn

class DQNRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 假设x是[batch_size, sequence_length, input_size]
        _, h_n = self.rnn(x)  # h_n是最后一个时间步的隐藏状态 [1, batch_size, hidden_size]
        q_values = self.fc(h_n.squeeze(0))  # 将隐藏状态转换为Q值
        return q_values
