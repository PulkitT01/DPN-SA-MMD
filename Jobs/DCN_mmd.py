import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import Utils


class DCN(nn.Module):
    def __init__(self, training_flag, input_nodes):
        super(DCN, self).__init__()
        self.training = training_flag

        # Shared layers
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=200)
        nn.init.xavier_uniform_(self.shared1.weight)

        self.shared2 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.shared2.weight)

        # Potential outcome Y(1)
        self.hidden1_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y1.weight)

        self.hidden2_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y1.weight)

        self.out_Y1 = nn.Linear(in_features=200, out_features=2)
        nn.init.xavier_uniform_(self.out_Y1.weight)

        # Potential outcome Y(0)
        self.hidden1_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y0.weight)

        self.hidden2_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y0.weight)

        self.out_Y0 = nn.Linear(in_features=200, out_features=2)
        nn.init.xavier_uniform_(self.out_Y0.weight)

    def get_representation(self, x, ps_score):
        """
        Returns the shared representation (dimension=200) after passing through 
        the shared layers, before branching into Y(1)/Y(0) towers.
        """
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training:
            # In 'training' mode, we do specialized dropout logic:
            entropy = Utils.get_shanon_entropy(ps_score.item())
            dropout_prob = Utils.get_dropout_probability(entropy, gama=1)
            shared_mask = Utils.get_dropout_mask(dropout_prob, self.shared1(x))

            x = F.relu(shared_mask * self.shared1(x))
            x = F.relu(shared_mask * self.shared2(x))
        else:
            # In 'eval' mode, no custom dropout:
            x = F.relu(self.shared1(x))
            x = F.relu(self.shared2(x))

        return x

    def forward(self, x, ps_score):
        """
        Forward returns (logits_y1, logits_y0).
        We branch into either __train_net (with custom dropout) or __eval_net.
        """
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training:
            y1, y0 = self.__train_net(x, ps_score)
        else:
            y1, y0 = self.__eval_net(x)

        return y1, y0

    def __train_net(self, x, ps_score):
        # Specialized dropout logic for the "shared" layers
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # Shared layers
        shared_mask = Utils.get_dropout_mask(dropout_prob, self.shared1(x))
        x = F.relu(shared_mask * self.shared1(x))
        x = F.relu(shared_mask * self.shared2(x))

        # Potential outcome Y(1)
        y1_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # Potential outcome Y(0)
        y0_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0

    def __eval_net(self, x):
        # Standard forward pass (no special dropout)
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        # Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0
