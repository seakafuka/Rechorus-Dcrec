# Dcrec: Debiased Contrastive Learning for Sequential Recommendation

This repository contains the implementation of **Dcrec (Debiased Contrastive Learning for Sequential Recommendation)**, a model designed to improve recommendation performance by addressing the issues of **popularity bias** and **sparse data**. The model has been successfully reproduced within the **Rechorus** framework, which provides a flexible environment for evaluating various recommendation models.

## Our Modifications

In our implementation of the **Dcrec** model, we made several important modifications to the original codebase:

1. We **added** the `DcrecRunner` and `DcrecReader` files to better handle data loading and model training.

2. The core implementation of the **Dcrec** model is located in the `model/Dcrec.py` file, where the primary model architecture and logic are defined.

3. We moved additional utility functions to the `utils/dcrec_util.py` file to better organize the code.

These changes were made to better align with the original framework, maintaining the overall structure and completeness of the **Dcrec** model within the **Rechorus** framework.



## Getting Started

To get started with **Dcrec** in the **Rechorus** framework, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/seakafuka/Rechorus-Dcrec.git
   ```
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Prepare the datasets

4. Train and evaluate the model:
  ```bash
    cd Rechorus-Dcrec
    cd src
    python main.py --model_name Dcrec --path your_path_of_data_dir
```
## Evaluation Results

The **Dcrec** model has been evaluated on the **Grocery and Gourmet Food** and **MovieLens 1M** datasets. Below is the key results from the experiments(epoch=20):

### Grocery and Gourmet Food Dataset

| Metrics      | HR@5  | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|--------------|-------|-------|-------|-------|--------|---------|---------|---------|
| Caser        | 0.3084 | 0.4170 | 0.5414 | 0.7711 | 0.2085 | 0.2435  | 0.2749  | 0.3203  |
| ComiRec      | 0.3676 | 0.4701 | 0.5833 | 0.7877 | 0.2593 | 0.2926  | 0.3211  | 0.3615  |
| SASRec       | 0.3667 | 0.4595 | 0.5681 | 0.7738 | 0.2722 | 0.3022  | 0.3296  | 0.3703  |
| GRU4Rec      | 0.3341 | 0.4419 | 0.5664 | 0.7869 | 0.2338 | 0.2687  | 0.3000  | 0.3436  |
| NARM         | 0.3552 | 0.4665 | 0.5844 | 0.7894 | 0.2538 | 0.2898  | 0.3211  | 0.3633  |
| **Dcrec**    | **0.3970** | **0.4848** | **0.5876** | **0.7927** | **0.3086** | **0.3369** | **0.3627** | **0.4032** |
