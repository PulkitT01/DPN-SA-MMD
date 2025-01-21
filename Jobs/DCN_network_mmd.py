from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from DCN_mmd import DCN 
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel="rbf"):
    """
    Empirical Maximum Mean Discrepancy (MMD) between two samples x and y.
    x: [n, d], y: [m, d]
    Returns a scalar MMD value.
    """
    # Compute dot products
    xx = torch.mm(x, x.t())  # [n, n]
    yy = torch.mm(y, y.t())  # [m, m]
    xy = torch.mm(x, y.t())  # [n, m]
    
    # Compute squared norms
    rx = (x ** 2).sum(dim=1, keepdim=True)  # shape [n, 1]
    ry = (y ** 2).sum(dim=1, keepdim=True)  # shape [m, 1]
    
    # Pairwise squared Euclidean
    dxx = rx + rx.t() - 2.0 * xx  # [n, n]
    dyy = ry + ry.t() - 2.0 * yy  # [m, m]
    dxy = rx + ry.t() - 2.0 * xy  # [n, m]
    
    # We'll use an RBF kernel with multiple bandwidths
    bandwidth_range = [10, 15, 20, 50]
    XX = YY = XY = 0.0
    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)
    
    return XX.mean() + YY.mean() - 2.0 * XY.mean()


class DCN_network:

    def train(self, train_parameters, device):
        """
        Train the DCN model, where we add an MMD term between the predicted
        distribution (softmax of logits) and the true (one-hot) distribution.
        
        We alternate:
          - Even epochs: Train the "treated" (Y1) tower
          - Odd epochs:  Train the "control" (Y0) tower
        """
        # === Unpack parameters ===
        epochs = train_parameters["epochs"]
        treated_batch_size = train_parameters["treated_batch_size"]
        control_batch_size = train_parameters["control_batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"].format(epochs, lr)

        treated_set_train = train_parameters["treated_set_train"]
        control_set_train = train_parameters["control_set_train"]
        input_nodes = train_parameters["input_nodes"]

        # MMD hyperparam
        lambda_mmd = train_parameters.get("lambda_mmd", 0.1)

        print("Saved model path: {0}".format(model_save_path))

        # === DataLoaders ===
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

        # === Initialize network + optimizer + CE loss
        network = DCN(training_flag=True, input_nodes=input_nodes).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        ce_loss = nn.CrossEntropyLoss()

        print(".. Training started ..")
        print(device)

        for epoch in range(epochs):
            network.train()
            total_loss = 0.0

            # ============================================
            # (A) Train TREATED tower on even epochs
            # ============================================
            if epoch % 2 == 0:
                # Enable grads for Y1 tower
                network.hidden1_Y1.weight.requires_grad = True
                network.hidden1_Y1.bias.requires_grad = True
                network.hidden2_Y1.weight.requires_grad = True
                network.hidden2_Y1.bias.requires_grad = True
                network.out_Y1.weight.requires_grad = True
                network.out_Y1.bias.requires_grad = True

                # Disable grads for Y0 tower
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

                    # Forward
                    logits_y1, _ = network(covariates_X, ps_score)  # Y1 tower

                    # (1) Cross-entropy
                    loss_ce = ce_loss(logits_y1, y_f)

                    # (2) MMD (predicted dist vs. one-hot of labels)
                    p1 = F.softmax(logits_y1, dim=1)
                    num_classes = logits_y1.shape[1]
                    y_f_onehot = F.one_hot(y_f, num_classes=num_classes).float()
                    loss_mmd_val = MMD(p1, y_f_onehot, kernel="rbf")

                    # Combine
                    loss = loss_ce + lambda_mmd * loss_mmd_val

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            # ============================================
            # (B) Train CONTROL tower on odd epochs
            # ============================================
            else:
                # Enable grads for Y0 tower
                network.hidden1_Y0.weight.requires_grad = True
                network.hidden1_Y0.bias.requires_grad = True
                network.hidden2_Y0.weight.requires_grad = True
                network.hidden2_Y0.bias.requires_grad = True
                network.out_Y0.weight.requires_grad = True
                network.out_Y0.bias.requires_grad = True

                # Disable grads for Y1 tower
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

                    # Forward
                    _, logits_y0 = network(covariates_X, ps_score)  # Y0 tower

                    # (1) Cross-entropy
                    loss_ce = ce_loss(logits_y0, y_f)

                    # (2) MMD
                    p0 = F.softmax(logits_y0, dim=1)
                    num_classes = logits_y0.shape[1]
                    y_f_onehot = F.one_hot(y_f, num_classes=num_classes).float()
                    loss_mmd_val = MMD(p0, y_f_onehot, kernel="rbf")

                    # Combine
                    loss = loss_ce + lambda_mmd * loss_mmd_val

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            # Print out every 10 epochs or as you prefer
            if epoch % 10 == 9:
                print(f"Epoch {epoch+1}/{epochs}, total loss (CE+MMD): {total_loss:.4f}")

        # Save model
        torch.save(network.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")


    def eval(self, eval_parameters, device, input_nodes):
        """
        Evaluate the DCN on treated + control sets.
        We'll do a forward pass and get Y1, Y0 predictions, then store them.
        """
        print(".. Evaluation started ..")
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]
        model_path = eval_parameters["model_save_path"]

        # Load trained network
        network = DCN(training_flag=False, input_nodes=input_nodes).to(device)
        # Remove "weights_only=True" if your PyTorch version doesn't support it
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()

        treated_data_loader = torch.utils.data.DataLoader(treated_set,
                                                          shuffle=False, num_workers=0)
        control_data_loader = torch.utils.data.DataLoader(control_set,
                                                          shuffle=False, num_workers=0)

        ITE_dict_list = []
        predicted_ITE_list = []

        y_f_list = []
        y1_hat_list = []
        y0_hat_list = []
        e_list = []
        T_list = []

        # --- Evaluate on treated set ---
        for batch in treated_data_loader:
            covariates_X, ps_score, y_f, t, e = batch

            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)

            # Forward
            logits_y1, logits_y0 = network(covariates_X, ps_score)
            _, y1_hat = torch.max(logits_y1, dim=1)
            _, y0_hat = torch.max(logits_y0, dim=1)

            predicted_ITE = y1_hat - y0_hat

            # Build a record
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

        # --- Evaluate on control set ---
        for batch in control_data_loader:
            covariates_X, ps_score, y_f, t, e = batch

            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)

            # Forward
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
        Helper to package up results in a dictionary. Adjust as you wish.
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
