import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fq
import os
cuda = torch.cuda.is_available()

class Linear_QNet(nn.Module):
    # def __init__(self, input_size, hidden_size, hidden_size2 = None, hidden_size3 = None, output_size = None):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        print("CUDA is available:", cuda)
        self.device = torch.device("cuda:0" if cuda else "cpu")

        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size, device=self.device)
        self.linear2 = nn.Linear(hidden_size, hidden_size2, device=self.device)
        self.linear3 = nn.Linear(hidden_size2, output_size, device=self.device)

    def forward(self, x):
        x = Fq.relu(self.linear1(x))
        x = self.linear2(x)
        x = Fq.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print("Model saved successfully")

    def load(self, file_name="model.pth"):
        model_folder_path = "./model"
        file_name = os.path.join(model_folder_path, file_name)

        if not os.path.exists(file_name):
            return

        self.load_state_dict(torch.load(file_name))
        print("Model loaded successfully")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        stat = torch.tensor(state, dtype=torch.float, device=self.model.device)
        next_stat = torch.tensor(next_state, dtype=torch.float, device=self.model.device)
        action = torch.tensor(action, dtype=torch.long, device=self.model.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.model.device)

        if len(stat.shape) == 1:
            stat = torch.unsqueeze(stat, 0)
            next_stat = torch.unsqueeze(next_stat, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(stat)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_stat[idx]))

            # target[idx][torch.argmax(action).item()] = Q_new
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
