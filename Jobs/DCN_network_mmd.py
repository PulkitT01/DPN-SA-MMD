from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from DCN_mmd import DCN  # <--- The DCN definition above
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel="rbf"):
    """
    Empirical Maximum Mean Discrepancy (MMD) between two samples x and y in [batch_size, d].
    We use an RBF kernel with multiple bandwidths.
    """
    xx = torch.mm(x, x.t())   # [n, n]
    yy = torch.mm(y, y.t())   # [m, m]
    xy = torch.mm(x, y.t())   # [n, m]

    rx = (x**2).sum(dim=1, keepdim=True)  # shape [n,1]
    ry = (y**2).sum(dim=1, keepdim=True)  # shape [m,1]

    # Pairwise squared distances
    dxx = rx + rx.t() - 2.*xx
    dyy = ry + ry.t() - 2.*yy
    dxy = rx + ry.t() - 2.*xy

    # Example RBF with multiple bandwidths
    bandwidth_range = [10, 15, 20, 50]
    XX = YY = XY = 0.
    for a in bandwidth_range:
        XX += torch.exp(-0.5*dxx/a)
        YY += torch.exp(-0.5*dyy/a)
        XY += torch.exp(-0.5*dxy/a)

    return XX.mean() + YY.mean() - 2.*XY.mean()


class DCN_network:

    def train(self, train_parameters, device):
        """
        Train the DCN model with CrossEntropy + MMD, 
        comparing predicted distribution vs. true distribution (one-hot) 
        for TREATED and CONTROL separately.
        """
        # Unpack parameters
        epochs = train_parameters["epochs"]
        treated_batch_size = train_parameters["treated_batch_size"]
        control_batch_size = train_parameters["control_batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"].format(epochs, lr)

        treated_set_train = train_parameters["treated_set_train"]
        control_set_train = train_parameters["control_set_train"]
        input_nodes = train_parameters["input_nodes"]

        # Lambda for MMD
        lambda_mmd = train_parameters.get("lambda_mmd", 0.1)

        print("Saved model path: {0}".format(model_save_path))

        # DataLoaders
        treated_data_loader = DataLoader(
            treated_set_train,
            batch_size=treated_batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        control_data_loader = DataLoader(
            control_set_train,
            batch_size=control_batch_size,
            shuffle=shuffle,
            num_workers=0
        )

        # Instantiate DCN
        network = DCN(training_flag=True, input_nodes=input_nodes).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        ce_loss = nn.CrossEntropyLoss()

        print(".. Training started ..")
        print(f"Device: {device}")

        for epoch in range(epochs):
            network.train()
            total_loss = 0.0

            # =========================================
            # (A) Train TREATED tower (Y1) on even epochs
            # =========================================
            if epoch % 2 == 0:
                # Enable Y1 tower
                network.hidden1_Y1.weight.requires_grad = True
                network.hidden1_Y1.bias.requires_grad = True
                network.hidden2_Y1.weight.requires_grad = True
                network.hidden2_Y1.bias.requires_grad = True
                network.out_Y1.weight.requires_grad = True
                network.out_Y1.bias.requires_grad = True

                # Disable Y0 tower
                network.hidden1_Y0.weight.requires_grad = False
                network.hidden1_Y0.bias.requires_grad = False
                network.hidden2_Y0.weight.requires_grad = False
                network.hidden2_Y0.bias.requires_grad = False
                network.out_Y0.weight.requires_grad = False
                network.out_Y0.bias.requires_grad = False

                for batch in treated_data_loader:
                    covariates_X, ps_score, y_f = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    y_f = y_f.to(device, dtype=torch.long)

                    # Forward -> (logits_y1, logits_y0), but we only optimize y1
                    logits_y1, _ = network(covariates_X, ps_score)

                    # (1) Cross-entropy
                    loss_ce = ce_loss(logits_y1, y_f)

                    # (2) MMD: compare softmax(logits_y1) vs. one-hot(y_f)
                    p1 = F.softmax(logits_y1, dim=1)  # shape [B, #classes=2]
                    y_f_onehot = F.one_hot(y_f, num_classes=p1.shape[1]).float()

                    loss_mmd_val = MMD(p1, y_f_onehot, kernel="rbf")

                    # Combine
                    loss = loss_ce + lambda_mmd * loss_mmd_val

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            # =========================================
            # (B) Train CONTROL tower (Y0) on odd epochs
            # =========================================
            else:
                # Enable Y0 tower
                network.hidden1_Y0.weight.requires_grad = True
                network.hidden1_Y0.bias.requires_grad = True
                network.hidden2_Y0.weight.requires_grad = True
                network.hidden2_Y0.bias.requires_grad = True
                network.out_Y0.weight.requires_grad = True
                network.out_Y0.bias.requires_grad = True

                # Disable Y1 tower
                network.hidden1_Y1.weight.requires_grad = False
                network.hidden1_Y1.bias.requires_grad = False
                network.hidden2_Y1.weight.requires_grad = False
                network.hidden2_Y1.bias.requires_grad = False
                network.out_Y1.weight.requires_grad = False
                network.out_Y1.bias.requires_grad = False

                for batch in control_data_loader:
                    covariates_X, ps_score, y_f = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    y_f = y_f.to(device, dtype=torch.long)

                    # Forward -> (logits_y1, logits_y0), but we only optimize y0
                    _, logits_y0 = network(covariates_X, ps_score)

                    # (1) Cross-entropy
                    loss_ce = ce_loss(logits_y0, y_f)

                    # (2) MMD: compare softmax(logits_y0) vs. one-hot(y_f)
                    p0 = F.softmax(logits_y0, dim=1)
                    y_f_onehot = F.one_hot(y_f, num_classes=p0.shape[1]).float()

                    loss_mmd_val = MMD(p0, y_f_onehot, kernel="rbf")

                    # Combine
                    loss = loss_ce + lambda_mmd * loss_mmd_val

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            # Print progress every 10 epochs
            if epoch % 10 == 9:
                print(f"Epoch {epoch+1}/{epochs}, total loss (CE+MMD): {total_loss:.4f}")

        # Save the final model
        torch.save(network.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")


    def eval(self, eval_parameters, device, input_nodes):
        """
        Evaluate the DCN on treated + control sets.
        We'll do forward passes and record the predicted outcomes (y1_hat, y0_hat).
        """
        print(".. Evaluation started ..")
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]
        model_path = eval_parameters["model_save_path"]

        # Load the model
        network = DCN(training_flag=False, input_nodes=input_nodes).to(device)
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()

        treated_data_loader = DataLoader(treated_set, shuffle=False, num_workers=0)
        control_data_loader = DataLoader(control_set, shuffle=False, num_workers=0)

        ITE_dict_list = []
        predicted_ITE_list = []
        y_f_list = []
        y1_hat_list = []
        y0_hat_list = []
        e_list = []
        T_list = []

        # ---- Evaluate on Treated set ----
        for batch in treated_data_loader:
            covariates_X, ps_score, y_f, t, e = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)

            logits_y1, logits_y0 = network(covariates_X, ps_score)
            _, y1_hat = torch.max(logits_y1, dim=1)
            _, y0_hat = torch.max(logits_y0, dim=1)

            predicted_ITE = y1_hat - y0_hat

            ITE_dict_list.append(self.create_ITE_Dict(
                covariates_X,
                ps_score.item(),
                y_f.item(),
                y1_hat.item(),
                y0_hat.item(),
                predicted_ITE.item()
            ))

            y_f_list.append(y_f.item())
            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())
            e_list.append(e.item())
            T_list.append(t)
            predicted_ITE_list.append(predicted_ITE.item())

        # ---- Evaluate on Control set ----
        for batch in control_data_loader:
            covariates_X, ps_score, y_f, t, e = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)

            logits_y1, logits_y0 = network(covariates_X, ps_score)
            _, y1_hat = torch.max(logits_y1, dim=1)
            _, y0_hat = torch.max(logits_y0, dim=1)

            predicted_ITE = y1_hat - y0_hat

            ITE_dict_list.append(self.create_ITE_Dict(
                covariates_X,
                ps_score.item(),
                y_f.item(),
                y1_hat.item(),
                y0_hat.item(),
                predicted_ITE.item()
            ))

            y_f_list.append(y_f.item())
            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())
            predicted_ITE_list.append(predicted_ITE.item())
            e_list.append(e.item())
            T_list.append(t)

        return {
            "predicted_ITE": predicted_ITE_list,
            "ITE_dict_list": ITE_dict_list,
            "y1_hat_list": y1_hat_list,
            "y0_hat_list": y0_hat_list,
            "e_list": e_list,
            "yf_list": y_f_list,
            "T_list": T_list
        }

    @staticmethod
    def create_ITE_Dict(covariates_X, ps_score, y_f, y1_hat, y0_hat, predicted_ITE):
        """
        Build a dictionary of data for each sample. 
        Adjust or remove as you see fit.
        """
        result_dict = OrderedDict()
        covariate_list = [element.item() for element in covariates_X.flatten()]
        for idx, val in enumerate(covariate_list, start=1):
            result_dict[f"X{idx}"] = val

        result_dict["ps_score"] = ps_score
        result_dict["factual"] = y_f
        result_dict["y1_hat"] = y1_hat
        result_dict["y0_hat"] = y0_hat
        result_dict["predicted_ITE"] = predicted_ITE
        return result_dict
