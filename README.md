# DP-TabTransformer
## Preparation

Clone the repo:

```
git clone https://github.com/IBM/DP-TabTransformer.git
cd DP-TabTransformer
```

environment configuration:

```
conda create -n DP-TabTransformer python=3.9
conda activate DP-TabTransformer
pip install -r requirements.txt
```

You are all set! 🎉

## Train from Scratch

We provide a simple training scripts `run_train_from_scratch.py` ,to train `TabTransformer` from scratch, simply run

```
python run_train_from_scratch.py
```

we use `ACSIncome_IN` dataset by default, to use your own data, just put it on `data` fold, and replace the data path in `train.py` accordingly.

## Parameter-Efficient Fine-Tuning

We offer a straightforward script, `run_dp_sgd.py`, which performs pre-training and fine-tuning of the `TabTransformer` using `DP-SGD`. This is done using the `ACSIncome_CA` dataset for pre-training and the `ACSIncome_IN` dataset for fine-tuning. To experiment with it, simply execute the following command:

```
python run_dp_sgd.py
```

By default, this script employs `Shallow Tuning`. To utilize other methods, you can enable them as follows:

- For `LoRA`, set `use_lora=True` in `run_dp_sgd.py`.
- To use `Adapter`, set `use_adapter=True`.
- For `Deep Tuning`, activate `deep_tuning=True`.
- To enable full tuning, set `full_tuning=True` accordingly.

