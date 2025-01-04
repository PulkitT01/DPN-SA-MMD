Below is a comprehensive, high-level view of the project structure, tracing how the scripts, classes, and functions interact and call one another. The outline follows a top-down flow, starting from the main execution script(s), then detailing the underlying classes and utilities in the order they are invoked. You can think of this as a directed flow or dependency map of the modules and classes.

---
## 1. **Main Entry Point**

### `main_propensity_dropout.py`
- **Purpose**: This script serves as the primary entry point.  
- **Key Steps**:
  1. **Imports**:
     - `Experiments` (from `Experiments.py`)
     - `Graphs` (from `Graphs.py`) - commented out usage in the code
     - `Model_25_1_25` (from `Model25_10_25.py`) - usage commented out
  2. **Main Section** (`if __name__ == '__main__':`)
     - Prints "Using original data"
     - Calls `Experiments().run_all_experiments(iterations=10, running_mode="original_data")`
       - (Commented lines show optional calls to `Model_25_1_25().run_all_expeiments()` and `Graphs().draw_scatter_plots()`)

**Flow**:
```
main_propensity_dropout.py
    --> Experiments.run_all_experiments(...)
    --> (Optionally) Model_25_1_25().run_all_expeiments(...)
    --> (Optionally) Graphs.draw_scatter_plots(...)
```

---

## 2. **Experiment Management**

### `Experiments.py`
- **Purpose**: Orchestrates multiple runs of training and testing experiments for causal inference models.
- **Key Class**:
  - `class Experiments`
    - **`run_all_experiments(self, iterations, running_mode)`**  
      - Configures device via `Utils.get_device()`
      - Determines run parameters via `__get_run_parameters(running_mode)`
      - For each `iter_id` in the specified number of iterations:
        1. Instantiates `DataLoader()`
        2. Loads data using one of:
           - `dL.preprocess_data_from_csv(...)`
           - `dL.preprocess_data_from_csv_augmented(...)` (depending on `running_mode`)
        3. Creates an instance of `DPN_SA_Deep`
        4. Calls `dp_sa.train_eval_DCN(...)` -> trains DCN with various propensity estimators
        5. Retrieves trained classifier references, e.g., `sparse_classifier`, `LR_model`, etc.
        6. Calls `dp_sa.test_DCN(...)` -> tests DCN with the previously trained models
        7. Collects experiment results (ATE, ATT, policy risk, etc.) into lists
        8. Writes final aggregated results to CSV and a summary file

    - **Private**:
      - `__get_run_parameters(running_mode)`: Sets file paths & hyperparams depending on real vs. synthetic data.
      - `load_data(running_mode, dL, train_path, test_path, iter_id)`: Helper that delegates to `dL.preprocess_data_from_csv...`.

**Flow**:
```
Experiments.run_all_experiments(...)
    --> DataLoader(...)                # loads data
    --> DPN_SA_Deep.train_eval_DCN(...) 
    --> DPN_SA_Deep.test_DCN(...)
    --> Aggregation and saving of results
```

---

## 3. **Main Training & Evaluation Logic**

### `DPN_SA_Deep.py`
- **Purpose**: Trains and tests the DCN (Deep Causal Network) with various propensity-estimation approaches (NN, SAE, LR, Lasso, etc.).
- **Key Class**:
  - `class DPN_SA_Deep`
    - **`train_eval_DCN(...)`**  
      1. Takes training data (`np_covariates_X_train`, `np_covariates_Y_train`) and a `DataLoader` instance.
      2. Trains multiple propensity-score models:
         - Neural network (`__train_propensity_net_NN`)
         - Sparse AutoEncoder (SAE) variants (`__train_propensity_net_SAE`)
         - Logistic Regression (`__train_propensity_net_LR`)
         - Logistic Regression Lasso (`__train_propensity_net_LR_Lasso`)
      3. For each trained propensity model, it calls `__train_DCN(...)` to train the DCN on the derived propensity scores.

      Returns references (classifiers/models) to be used later during testing.

    - **`test_DCN(...)`**  
      1. Takes test data (`np_covariates_X_test`, `np_covariates_Y_test`).
      2. Evaluates DCN using the already trained propensity estimators:
         - `__test_DCN_NN(...)`
         - `__test_DCN_SAE(...)`
         - `__test_DCN_LR(...)`
         - `__test_DCN_LR_Lasso(...)`
      3. Each of these calls eventually leads to `__do_test_DCN(...)`, which loads the respective DCN model, runs inference, and computes metrics (ATE, ATT, policy risk, etc.).

    - **Sub-Training Methods**:
      - `__train_propensity_net_NN(...)` -> uses `Propensity_socre_network` and then calls `__train_DCN(...)`.
      - `__train_propensity_net_SAE(...)` -> uses `Sparse_Propensity_score` (multiple variants), then calls `__train_DCN(...)`.
      - `__train_propensity_net_LR(...)` and `__train_propensity_net_LR_Lasso(...)` -> uses `Propensity_socre_LR`.
      - `__train_DCN(...)` -> delegates to `DCN_network.train(...)`.

    - **Sub-Testing Methods**:
      - `__test_DCN_NN(...)`, `__test_DCN_SAE(...)`, `__test_DCN_LR(...)`, `__test_DCN_LR_Lasso(...)`
        - Evaluate the propensity model, pass the results to `__do_test_DCN(...)`.
      - `__do_test_DCN(...)` -> uses `DCN_network().eval(...)`.

**Flow**:
```
DPN_SA_Deep.train_eval_DCN(...)
    --> __train_propensity_net_NN(...) 
         --> Propensity_socre_network.train(...) 
         --> __train_DCN(...) -> DCN_network.train(...)
    --> __train_propensity_net_SAE(...)
         --> Sparse_Propensity_score.train(...) [Multiple SAE variants]
         --> __train_DCN(...) -> DCN_network.train(...)
    --> __train_propensity_net_LR(...) 
         --> Propensity_socre_LR.train(...)
         --> __train_DCN(...) -> DCN_network.train(...)
    --> __train_propensity_net_LR_Lasso(...)
         --> Propensity_socre_LR.train(...) with regularized=True
         --> __train_DCN(...) -> DCN_network.train(...)
    --> returns references to the trained classifiers

DPN_SA_Deep.test_DCN(...)
    --> __test_DCN_NN(...)  -> __do_test_DCN(...) -> DCN_network.eval(...)
    --> __test_DCN_SAE(...) -> __do_test_DCN(...) -> DCN_network.eval(...)
    --> __test_DCN_LR(...)  -> __do_test_DCN(...) -> DCN_network.eval(...)
    --> __test_DCN_LR_Lasso(...) -> __do_test_DCN(...) -> DCN_network.eval(...)
    --> aggregates final metrics (ATE, ATT, policy risk, etc.)
```

---

## 4. **DCN Model Architecture & Training**

### `DCN.py`
- **Purpose**: Defines the core **Deep Causal Network** architecture for potential outcomes \(Y(0)\) and \(Y(1)\).
- **Key Class**:
  - `class DCN(nn.Module)`
    - **Constructor**: Defines shared layers and separate heads for `Y(1)` and `Y(0)`.
    - **`forward(self, x, ps_score)`**:
      - If `training` flag is `True`, uses `__train_net(...)`; else `__eval_net(...)`.
      - Returns two potential outcomes (`y1`, `y0`).
    - **`__train_net(...)`**:
      - Dynamically calculates dropout probability based on Shannon entropy of the propensity score.
      - Applies dropout masks to shared layers and separate outcome heads.
    - **`__eval_net(...)`**:
      - Standard forward pass without dropout.

---

### `DCN_network.py`
- **Purpose**: Provides the training (`train`) and evaluation (`eval`) methods that wrap around the `DCN` model.
- **Key Class**:
  - `class DCN_network`
    - **`train(self, train_parameters, device)`**:
      1. Builds data loaders for treated and control sets.
      2. Initializes `DCN` with `training_flag=True`.
      3. Runs gradient updates in alternating fashion (even epochs for treated data, odd epochs for control data).
      4. Saves the trained model to disk.

    - **`eval(self, eval_parameters, device, input_nodes)`**:
      1. Loads the saved `DCN` model in eval mode.
      2. Runs predictions for treated and control test sets.
      3. Collects predicted ITE, potential outcomes, etc.

    - **Other methods**:
      - `create_ITE_Dict(...)` -> organizes ITE predictions in a structured dictionary.

**Flow**:
```
DCN_network.train(...)
    --> Instantiates DCN(training_flag=True)
    --> Alternates training on treated vs. control sets
    --> Saves final DCN model

DCN_network.eval(...)
    --> Loads DCN(training_flag=False)
    --> Predicts on test sets
    --> Returns ITE predictions, potential outcomes, etc.
```

---

## 5. **Additional Model Variation**

### `Model25_10_25.py` (Class: `Model_25_1_25`)
- **Purpose**: Demonstrates a variant workflow for training/evaluating a DCN with a certain AE architecture (25-1-25 or “SAE E2E”).
- **Flow**:
  - `run_all_expeiments()`:
    1. Loops over iterations.
    2. Prepares data via `DataLoader` (similar to `Experiments`).
    3. Uses `__train_eval_DCN(...)` to train a shallow-like AE + DCN pipeline.
    4. Tests with `__test_DCN(...)`.
    5. Logs and saves results.

- Internally uses:
  - `shallow_train.py`–like approach to train a “shallow” AE,  
  - `DCN_network.py` to train/eval DCN,  
  - `Utils.py` for conversions and I/O.

---

## 6. **Propensity Estimation Methods**

### `Propensity_socre_network.py`  
- **Purpose**: Trains a neural network to estimate propensity scores (treatment probabilities).
- **Key Class**:
  - `class Propensity_socre_network`
    - **`train(train_parameters, device, phase)`**:
      1. Creates a `Propensity_net_NN` instance.
      2. Runs training loops (cross-entropy vs. treatment label).
      3. Saves the trained model.

    - **`eval(eval_parameters, device, phase)`**:
      - Loads the saved `Propensity_net_NN`.
      - Performs inference and extracts treatment probability \( p(T=1|X) \).
      - Returns `prop_score_list`.

    - **`eval_return_complete_list(...)`**: Similar to `eval` but returns a richer data structure.

### `Propensity_net_NN.py`
- **Key Class**:
  - `class Propensity_net_NN(nn.Module)`
    - Standard feed-forward network with two hidden layers + final 2-class output.
    - Depending on `phase == "eval"`, returns `F.softmax(...)` or raw logits.

**Flow**:
```
Propensity_socre_network.train(...) 
    --> Instantiates Propensity_net_NN
    --> Trains with cross-entropy
    --> Saves model
Propensity_socre_network.eval(...)
    --> Loads Propensity_net_NN
    --> Returns predicted probabilities
```

---

### `Propensity_score_LR.py`
- **Purpose**: Trains or tests a Logistic Regression (optionally L1-regularized) for propensity estimation.
- **Class**:
  - `class Propensity_socre_LR`
    - **`train(...)`**: Fits `LogisticRegression` on `cpu().numpy()` data. Returns predicted probabilities + LR model.
    - **`test(...)`**: Given test data and a trained LR model, returns predicted probabilities.

---

### `Sparse_Propensity_score.py`
- **Purpose**: Trains a **Sparse AutoEncoder** that ultimately classifies treatment vs. control.
- **Key Class**:
  - `class Sparse_Propensity_score`
    - **`train(...)`**:
      1. End-to-end SAE training with `__end_to_end_train_SAE(...)`.
      2. Layer-wise training with `__layer_wise_train_SAE(...)` (greedy stacking).
      3. Returns multiple classifier variants: (e2e, stacked_all_layer_active, stacked_cur_layer_active).

    - **`eval(...)`**:
      - Uses the final stacked network with a classification head to predict propensities.

    - Internally uses:
      - `Sparse_Propensity_net` or `Sparse_Propensity_net_shallow`
      - `Utils.KL_divergence(...)` for the sparsity constraint
      - A final classification layer (`nn.Linear(...)`) appended to the AE.

### `Sparse_Propensity_net.py` / `Sparse_Propensity_net_shallow.py`
- **Purpose**: Provide different AE architectures:
  - `Sparse_Propensity_net(...)` with deeper (20→10) layers in the encoder + symmetrical decoder.
  - `Sparse_Propensity_net_shallow(...)` with single-layer (20) encoder + optional appended layers.

---

## 7. **Shallow AutoEncoder Variation**

### `shallow_net.py` & `shallow_train.py`
- **Purpose**: Provide a simpler or “shallow” AE approach (10 hidden units) and training routine for the same usage as above.
- **`shallow_net`**:  
  - Minimal autoencoder with an encoder (17→10) and decoder (10→17).  
- **`shallow_train`**:  
  - Similar approach to `Sparse_Propensity_score` but only the shallow net is used.  
  - After AE is trained, a 2-output classification head is appended.

---

## 8. **Data Loading & Utilities**

### `dataloader.py`
- **Class**: `DataLoader`
  - Responsible for reading `.npz` or `.csv` data, merging columns, and preparing the final feature sets for training DCN or AE networks.
  - Key methods:
    - `preprocess_data_from_csv(...)`
    - `preprocess_data_from_csv_augmented(...)`
    - `prepare_tensor_for_DCN(...)`: Splits dataset into treated vs. control for DCN.

**Flow**:
```
DataLoader
    --> .preprocess_data_from_csv(...) 
    --> .prepare_tensor_for_DCN(...) -> returns treated_data, control_data
```

### `Utils.py`
- Provides a suite of utility functions:
  - Tensor conversions (`convert_to_tensor`, `convert_to_tensor_DCN`, etc.)
  - Shannon entropy / KL divergence for dropout & AE sparseness.
  - I/O helpers to write CSV or combine arrays.
  - Basic scoring (get_num_correct, cal_policy_val, etc.)

---

## 9. **Graphs & Testing**

### `Graphs.py` (Referenced but not deeply used in the shared code)
- Potentially draws scatter plots, correlations, or other analysis. It’s commented out.

### `t_test.py`
- Contains statistical significance testing using `scipy.stats.ttest_ind` to compare arrays (e.g., MSE results) from two methods (SAE vs. LR-lasso).

---

## 10. **Overview of the Directed Flow**

1. **Start**: `main_propensity_dropout.py`
   - Invokes **`Experiments.run_all_experiments(...)`**.

2. **`Experiments.run_all_experiments(...)`**:
   - For each iteration:
     - Loads data (train/test) via **`DataLoader`**.
     - Calls **`DPN_SA_Deep.train_eval_DCN(...)`** to train all models.

3. **Within `DPN_SA_Deep.train_eval_DCN(...)`**:
   - Trains multiple propensity scorers:
     1. **NN**: `Propensity_socre_network.train(...)`
     2. **SAE**: `Sparse_Propensity_score.train(...)`
     3. **LR / LR Lasso**: `Propensity_socre_LR.train(...)`
   - After each propensity model is trained, it calls:
     - **`__train_DCN(...)`** -> which uses `DCN_network().train(...)`.

4. **At test time**: `DPN_SA_Deep.test_DCN(...)`:
   - Calls the relevant `_test_DCN_*` function for each propensity approach:
     - **`__test_DCN_NN`** -> loads DCN model and does `DCN_network.eval(...)`
     - **`__test_DCN_SAE`**
     - **`__test_DCN_LR`**, etc.
   - Collects & returns final metrics (ATE, bias_att, policy_risk, etc.).

5. **`DCN_network`**:
   - Manages the creation, training, and evaluation of the `DCN(nn.Module)`.
   - Training:
     - Alternates between treated and control subsets.
   - Evaluation:
     - Loads the model, runs forward, and returns potential outcomes & ITE.

6. **Results**:
   - The final results are stored (CSV or displayed) by `Experiments`.

---

### **Simplified Diagram**

```
main_propensity_dropout.py
     -> Experiments.run_all_experiments(iterations, mode)
          -> DataLoader(...) -> loads train/test data
          -> For each iter_id:
               -> DPN_SA_Deep.train_eval_DCN(...)
                    -> train_propensity_net_* (NN, SAE, LR, Lasso)
                    -> each calls DCN_network.train(...)
               -> DPN_SA_Deep.test_DCN(...)
                    -> test_DCN_* (NN, SAE, LR, Lasso)
                    -> each calls DCN_network.eval(...)
               -> Accumulate & save results
```

---

## Conclusion

- **`main_propensity_dropout.py`** is the primary driver, invoking **`Experiments`**.  
- **`Experiments`** orchestrates data loading (`DataLoader`), model training, and testing (`DPN_SA_Deep`).  
- **`DPN_SA_Deep`** uses various **propensity-estimation** classes (`Propensity_socre_network`, `Sparse_Propensity_score`, `Propensity_socre_LR`) and trains the **DCN** (via `DCN_network`).  
- **`Utils`, `dataloader`, and AE modules** (`Sparse_Propensity_net`, `shallow_net`, etc.) provide the underlying building blocks.  
- **Final ITE** predictions and metrics are saved, with optional significance testing (`t_test.py`) or graphs (`Graphs.py`).

This structure collectively offers a pipeline for **causal effect** estimation using different propensity-score methods and a Deep Causal Network for potential outcome modeling.
