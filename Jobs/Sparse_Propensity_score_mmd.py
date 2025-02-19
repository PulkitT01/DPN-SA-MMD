import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from Sparse_Propensity_net import Sparse_Propensity_net
from Sparse_Propensity_net_shallow import Sparse_Propensity_net_shallow
from Utils import Utils, MMD

class Sparse_Propensity_score_mmd:
    def __init__(self):
        self.sparse_classifier_e2e = None

    def train(self, train_parameters, device, phase):
        print(".. Training started with MMD regularization ..")
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        train_set = train_parameters["train_set"]
        input_nodes = train_parameters["input_nodes"]

        lambda_mmd = train_parameters.get("lambda_mmd", 0.1)

        data_loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                        shuffle=shuffle, num_workers=0)

        model = Sparse_Propensity_net(training_mode=phase, device=device,
                                      input_nodes=input_nodes).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch += 1
            model.train()
            total_loss = 0
            for batch in data_loader_train:
                covariates, _ = batch
                covariates = covariates.to(device)
                covariates = covariates[:, :-2]

                treatment_pred = model(covariates)
                mse_loss = criterion(treatment_pred, covariates)
                mmd_loss = lambda_mmd * MMD(treatment_pred, covariates)

                loss = mse_loss + mmd_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {total_loss:.4f}")

        print("Training completed with MMD.")
        return model

