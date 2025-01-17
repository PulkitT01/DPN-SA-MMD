from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from DCN_mmd import DCN
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel="rbf"):
    """
    Empirical Maximum Mean Discrepancy (MMD) between two samples x and y.
    Works for x of shape [n, d] and y of shape [m, d].
    """
    # Compute dot products
    xx = torch.mm(x, x.t())  # [n, n]
    yy = torch.mm(y, y.t())  # [m, m]
    zz = torch.mm(x, y.t())  # [n, m]
    
    # Compute squared norms for each row in x and y
    rx = (x ** 2).sum(dim=1, keepdim=True)  # shape [n, 1]
    ry = (y ** 2).sum(dim=1, keepdim=True)  # shape [m, 1]
    
    # Compute pairwise squared Euclidean distances:
    dxx = rx + rx.t() - 2.0 * xx  # [n, n]
    dyy = ry + ry.t() - 2.0 * yy  # [m, m]
    dxy = rx + ry.t() - 2.0 * zz  # [n, m]
    
    # Initialize kernel sums
    XX = torch.zeros_like(dxx).to(device)
    YY = torch.zeros_like(dyy).to(device)
    XY = torch.zeros_like(dxy).to(device)
    
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
    elif kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    
    return XX.mean() + YY.mean() - 2.0 * XY.mean()

class DCN_network:

    def train(self, train_parameters, device):
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

        # MMD hyperparameters (you can define these in train_parameters or hardcode)
        lambda_mmd = train_parameters.get("lambda_mmd", 1.0)
        split_ratio = train_parameters.get("split_ratio", 0.8)  # e.g., 80-20 split

        print("Saved model path: {0}".format(model_save_path))

        # === 1) Internal split of treated_set_train into train/val ===
        treated_len = len(treated_set_train)
        treated_train_len = int(split_ratio * treated_len)
        treated_val_len = treated_len - treated_train_len
        treated_train_ds, treated_val_ds = random_split(
            treated_set_train, [treated_train_len, treated_val_len]
        )

        # === 2) Internal split of control_set_train into train/val ===
        control_len = len(control_set_train)
        control_train_len = int(split_ratio * control_len)
        control_val_len = control_len - control_train_len
        control_train_ds, control_val_ds = random_split(
            control_set_train, [control_train_len, control_val_len]
        )

        # === 3) DataLoaders: main train-split + val-split
        treated_data_loader_train = DataLoader(
            treated_train_ds,
            batch_size=treated_batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        treated_data_loader_val = DataLoader(
            treated_val_ds,
            batch_size=treated_batch_size,
            shuffle=False,
            num_workers=0
        )

        control_data_loader_train = DataLoader(
            control_train_ds,
            batch_size=control_batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        control_data_loader_val = DataLoader(
            control_val_ds,
            batch_size=control_batch_size,
            shuffle=False,
            num_workers=0
        )

        # === 4) Instantiate DCN + optimizer + loss
        network = DCN(training_flag=True, input_nodes=input_nodes).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        ce_loss = nn.CrossEntropyLoss()  # same as your F.cross_entropy usage

        min_loss = 1e9
        dataset_loss = 0.0

        print(".. Training started ..")
        print(device)

        # === 5) Training loop (with your original Y1/Y0 alternation) ===
        for epoch in range(epochs):
            network.train()  # set mode=training
            total_loss = 0.0

            # =========================
            # (A) Train TREATED tower
            # =========================
            if epoch % 2 == 0:
                dataset_loss = 0

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

                # ---- (1) Cross-entropy pass on TREATED train-split ----
                for batch in treated_data_loader_train:
                    covariates_X, ps_score, y_f = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    y_f = y_f.to(device, dtype=torch.long)

                    # Forward
                    y1_hat = network(covariates_X, ps_score)[0]
                    
                    # Cross-entropy
                    loss_ce = ce_loss(y1_hat, y_f)

                    # Backprop
                    optimizer.zero_grad()
                    loss_ce.backward()
                    optimizer.step()

                    total_loss += loss_ce.item()

                # ---- (2) MMD pass on TREATED (train-split vs. val-split) ----
                # We'll do a separate pass to gather embeddings
                if treated_val_len > 0:  # only if val-split is non-empty
                    # Gather embeddings from train-split
                    train_emb_list = []
                    network.train()  # ensure dropout logic is correct for "train" data
                    for batch in treated_data_loader_train:
                        covariates_X, ps_score, _ = batch
                        covariates_X = covariates_X.to(device)
                        ps_score = ps_score.squeeze().to(device)

                        # We want to compute MMD and backprop, so no .detach()
                        rep_train = network.get_representation(covariates_X, ps_score)
                        train_emb_list.append(rep_train)
                    
                    if len(train_emb_list) > 0:
                        train_emb_tensor = torch.cat(train_emb_list, dim=0)
                    else:
                        train_emb_tensor = None

                    # Gather embeddings from val-split
                    val_emb_list = []
                    # Typically we do "eval()" for val data, but if you want 
                    # consistent dropout usage, you might keep 'train()' mode. 
                    # That depends on whether you want them to be "true eval" 
                    # embeddings or "train-mode" embeddings. We'll do eval() 
                    # so no dropout for val embeddings:
                    network.eval()
                    with torch.no_grad():
                        for batch in treated_data_loader_val:
                            covariates_X, ps_score, _ = batch
                            covariates_X = covariates_X.to(device)
                            ps_score = ps_score.squeeze().to(device)
                            rep_val = network.get_representation(covariates_X, ps_score)
                            val_emb_list.append(rep_val)

                    if len(val_emb_list) > 0:
                        val_emb_tensor = torch.cat(val_emb_list, dim=0)
                    else:
                        val_emb_tensor = None

                    # Compute MMD if both sides are non-empty
                    if train_emb_tensor is not None and val_emb_tensor is not None:
                        mmd_value = MMD(train_emb_tensor, val_emb_tensor, kernel="rbf")
                        
                        # Weighted MMD
                        loss_mmd = lambda_mmd * mmd_value
                        
                        # Because the val embeddings were computed with no_grad(),
                        # we only have a graph for the train embeddings. 
                        # So let's do one more pass with train embeddings 
                        # in a differentiable manner:
                        optimizer.zero_grad()
                        loss_mmd.backward()
                        optimizer.step()

                        total_loss += loss_mmd.item()

                dataset_loss = total_loss

            # =========================
            # (B) Train CONTROL tower
            # =========================
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

                # ---- (1) Cross-entropy pass on CONTROL train-split ----
                for batch in control_data_loader_train:
                    covariates_X, ps_score, y_f = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    y_f = y_f.to(device, dtype=torch.long)

                    y0_hat = network(covariates_X, ps_score)[1]
                    loss_ce = F.cross_entropy(y0_hat, y_f)

                    optimizer.zero_grad()
                    loss_ce.backward()
                    optimizer.step()

                    total_loss += loss_ce.item()

                # ---- (2) MMD pass on CONTROL (train-split vs. val-split) ----
                if control_val_len > 0:
                    # Gather embeddings from train-split
                    train_emb_list = []
                    network.train()  # train-mode for the control train data
                    for batch in control_data_loader_train:
                        covariates_X, ps_score, _ = batch
                        covariates_X = covariates_X.to(device)
                        ps_score = ps_score.squeeze().to(device)

                        rep_train = network.get_representation(covariates_X, ps_score)
                        train_emb_list.append(rep_train)
                    
                    if len(train_emb_list) > 0:
                        train_emb_tensor = torch.cat(train_emb_list, dim=0)
                    else:
                        train_emb_tensor = None

                    # Gather embeddings from val-split
                    val_emb_list = []
                    network.eval()  # eval-mode for val data
                    with torch.no_grad():
                        for batch in control_data_loader_val:
                            covariates_X, ps_score, _ = batch
                            covariates_X = covariates_X.to(device)
                            ps_score = ps_score.squeeze().to(device)
                            rep_val = network.get_representation(covariates_X, ps_score)
                            val_emb_list.append(rep_val)

                    if len(val_emb_list) > 0:
                        val_emb_tensor = torch.cat(val_emb_list, dim=0)
                    else:
                        val_emb_tensor = None

                    if train_emb_tensor is not None and val_emb_tensor is not None:
                        mmd_value = MMD(train_emb_tensor, val_emb_tensor, kernel="rbf")
                        loss_mmd = lambda_mmd * mmd_value
                        optimizer.zero_grad()
                        loss_mmd.backward()
                        optimizer.step()

                        total_loss += loss_mmd.item()

                dataset_loss += total_loss

            # === Print every 10 epochs (or whatever schedule you like) ===
            if epoch % 10 == 9:
                print("epoch: {0}, Combined loss (CE + MMD): {1:.4f}".format(epoch, dataset_loss))

        # === Save the model parameters ===
        torch.save(network.state_dict(), model_save_path)

    def eval(self, eval_parameters, device, input_nodes):
        """
        The eval function remains exactly the same as in your original code.
        """
        print(".. Evaluation started ..")
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]
        model_path = eval_parameters["model_save_path"]

        network = DCN(training_flag=False, input_nodes=input_nodes).to(device)
        # If desired, you can remove 'weights_only=True' if it causes issues:
        network.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        network.eval()

        treated_data_loader = torch.utils.data.DataLoader(treated_set, shuffle=False, num_workers=0)
        control_data_loader = torch.utils.data.DataLoader(control_set, shuffle=False, num_workers=0)

        err_treated_list = []
        err_control_list = []
        true_ITE_list = []
        predicted_ITE_list = []

        ITE_dict_list = []

        y_f_list = []
        y1_hat_list = []
        y0_hat_list = []
        e_list = []
        T_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score, y_f, t, e = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            treatment_pred = network(covariates_X, ps_score)

            pred_y1_hat = treatment_pred[0]
            pred_y0_hat = treatment_pred[1]

            _, y1_hat = torch.max(pred_y1_hat.data, 1)
            _, y0_hat = torch.max(pred_y0_hat.data, 1)

            predicted_ITE = y1_hat - y0_hat
            ITE_dict_list.append(self.create_ITE_Dict(
                covariates_X,
                ps_score.item(), y_f.item(),
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

        for batch in control_data_loader:
            covariates_X, ps_score, y_f, t, e = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)

            treatment_pred = network(covariates_X, ps_score)

            pred_y1_hat = treatment_pred[0]
            pred_y0_hat = treatment_pred[1]

            _, y1_hat = torch.max(pred_y1_hat.data, 1)
            _, y0_hat = torch.max(pred_y0_hat.data, 1)

            predicted_ITE = y1_hat - y0_hat
            ITE_dict_list.append(self.create_ITE_Dict(
                covariates_X,
                ps_score.item(), y_f.item(),
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
        result_dict = OrderedDict()
        covariate_list = [element.item() for element in covariates_X.flatten()]
        idx = 0
        for item in covariate_list:
            idx += 1
            result_dict["X" + str(idx)] = item

        result_dict["ps_score"] = ps_score
        result_dict["factual"] = y_f
        result_dict["y1_hat"] = y1_hat
        result_dict["y0_hat"] = y0_hat
        result_dict["predicted_ITE"] = predicted_ITE

        return result_dict
