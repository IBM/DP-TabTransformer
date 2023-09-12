import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.layer_utils import count_params

from tabtransformertf.models.tabtransformer import TabTransformer, TabTransformerRTD, LoraLayer, AdapterLayer
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep, df_to_pretrain_dataset
import tensorflow_privacy

def evaluate(tabtransformer):
    total_data = pd.read_csv('data/2018/1-Year/ACSIncome_IN.csv')

    # Column information
    NUMERIC_FEATURES = total_data.select_dtypes(include=np.number).columns
    CATEGORICAL_FEATURES = total_data.select_dtypes(exclude=np.number).columns[:-1]  # exclude label column and DT

    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    LABEL = 'PINCP'

    # encoding as binary target
    total_data[LABEL] = total_data[LABEL].apply(lambda x: int(x == True))
    total_data[LABEL].mean()

    # Set data types
    total_data[CATEGORICAL_FEATURES] = total_data[CATEGORICAL_FEATURES].astype(str)

    total_data[NUMERIC_FEATURES] = total_data[NUMERIC_FEATURES].astype(float)

    # Train/test split
    train_data, test_data = train_test_split(total_data, test_size=0.2)
    X_train, X_val = train_test_split(train_data, test_size=0.2)
    # To TF Dataset
    train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL)
    val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
    # test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], shuffle=False)  # No target, no shuffle
    test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], LABEL, shuffle=False)  # No target, no shuffle

    test_preds = tabtransformer.predict(test_dataset)
    acc = np.round(accuracy_score(test_data[LABEL], test_preds.ravel() > 0.5), 4)
    return acc

def finetune(tabtransformer,
             noise,
             batch_size,
             epochs,
             lr,
             full_tuning = False,
             deep_tuning = False,
             use_lora = False,
             use_adapter = False):


    total_data = pd.read_csv('data/2018/1-Year/ACSIncome_IN.csv')

    # Column information
    NUMERIC_FEATURES = total_data.select_dtypes(include=np.number).columns
    CATEGORICAL_FEATURES = total_data.select_dtypes(exclude=np.number).columns[:-1]  # exclude label column and DT

    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    LABEL = 'PINCP'

    # encoding as binary target
    total_data[LABEL] = total_data[LABEL].apply(lambda x: int(x == True))
    total_data[LABEL].mean()

    # Set data types
    total_data[CATEGORICAL_FEATURES] = total_data[CATEGORICAL_FEATURES].astype(str)

    total_data[NUMERIC_FEATURES] = total_data[NUMERIC_FEATURES].astype(float)

    # Train/test split
    train_data, test_data = train_test_split(total_data, test_size=0.2)
    X_train, X_val = train_test_split(train_data, test_size=0.2)
    # To TF Dataset
    train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL)
    val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
    # test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], shuffle=False)  # No target, no shuffle
    test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], LABEL, shuffle=False)  # No target, no shuffle


    if not full_tuning:
        # freeze the backbone
        tabtransformer.encoder.trainable = False
        for layer in tabtransformer.mlp_final:
            layer.trainable = False
        tabtransformer.output_layer.trainable = False

    # activate the prompt if needed
    for i in range(len(tabtransformer.prompts)):
        if full_tuning or use_lora or use_adapter:
            tabtransformer.prompts[i].trainable = False
        else:
            tabtransformer.prompts[i].trainable = (i < 1 or deep_tuning)

    # add lora
    if use_lora:
        for i in range(len(tabtransformer.encoder.transformers)):
            tabtransformer.encoder.transformers[i].ffn = LoraLayer(
                tabtransformer.encoder.transformers[i].ffn
            )

    # add adapter
    if use_adapter:
        for i in range(len(tabtransformer.encoder.transformers)):
            tabtransformer.encoder.transformers[i].ffn = AdapterLayer(
                tabtransformer.encoder.transformers[i].ffn
            )

    _ = tabtransformer.predict(val_dataset)

    if use_lora:
        tabtransformer.encoder.trainable = True
        for each_layer in tabtransformer.encoder.layers:
            if "transformer_block" not in each_layer.name:
                each_layer.trainable = False

        for each_block in tabtransformer.encoder.transformers:
            each_block.att.trainable = False
            each_block.layernorm1.trainable = False
            each_block.layernorm2.trainable = True
            each_block.ffn.original_layer.trainable = False
            each_block.ffn.lora_down.trainable = True
            each_block.ffn.lora_up.trainable = True


    if use_adapter:
        tabtransformer.encoder.trainable = True
        for each_layer in tabtransformer.encoder.layers:
            if "transformer_block" not in each_layer.name:
                each_layer.trainable = False

        for each_block in tabtransformer.encoder.transformers:
            each_block.att.trainable = False
            each_block.layernorm1.trainable = False
            each_block.layernorm2.trainable = True
            each_block.ffn.pretrained_ffn.trainable = False
            each_block.ffn.down_project.trainable = True
            each_block.ffn.up_project.trainable = True


    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        learning_rate=lr,
        l2_norm_clip=2,
        noise_multiplier=noise,
        num_microbatches=1,
    )


    tabtransformer.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="PR AUC", curve='PR')],
    )

    print(count_params(tabtransformer.trainable_weights))
    history = tabtransformer.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        batch_size=batch_size,
    )

    test_preds = tabtransformer.predict(test_dataset)
    acc = np.round(accuracy_score(test_data[LABEL], test_preds.ravel() > 0.5), 4)
    return acc

def pretrain(with_dp = False, noise = 0, epochs = 1, use_adapter = False, use_lora = False):

    train_data = pd.read_csv("data/2018/1-Year/ACSIncome_CA.csv")

    # Column information
    NUMERIC_FEATURES = train_data.select_dtypes(include=np.number).columns
    CATEGORICAL_FEATURES = train_data.select_dtypes(exclude=np.number).columns[:-1]  # exclude label column and DT

    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    LABEL = 'PINCP'

    # encoding as binary target
    train_data[LABEL] = train_data[LABEL].apply(lambda x: int(x == True))
    train_data[LABEL].mean()


    # Set data types
    train_data[CATEGORICAL_FEATURES] = train_data[CATEGORICAL_FEATURES].astype(str)

    train_data[NUMERIC_FEATURES] = train_data[NUMERIC_FEATURES].astype(float)

    # Train/test split
    X_train, X_val = train_test_split(train_data, test_size=0.2)
    category_prep_layers = build_categorical_prep(X_train, CATEGORICAL_FEATURES)
    # To TF Dataset
    train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL)
    val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
    tabtransformer = TabTransformer(
        numerical_features = NUMERIC_FEATURES,
        categorical_features = CATEGORICAL_FEATURES,
        categorical_lookup=category_prep_layers,
        embedding_dim=32,
        out_dim=1,
        out_activation='sigmoid',
        depth=4,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.2,
        mlp_hidden_factors= [4]*5,
        use_column_embedding=True,
    )


    if with_dp:
        optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            learning_rate=0.0005,
            l2_norm_clip=2,
            noise_multiplier=noise,
            num_microbatches=1,
        ) # 0.0005 for CA, 0,001 for IN
    else:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=0.0001, weight_decay=0.0001
        )


    _ = tabtransformer.predict(val_dataset)



    tabtransformer.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="PR AUC", curve='PR')],
    )


    history = tabtransformer.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        verbose=1,
    )


    return tabtransformer


def train_from_scratch(noise, batch_size, epochs, lr):
    total_data = pd.read_csv('data/2018/1-Year/ACSIncome_CA.csv')

    # Column information
    NUMERIC_FEATURES = total_data.select_dtypes(include=np.number).columns
    CATEGORICAL_FEATURES = total_data.select_dtypes(exclude=np.number).columns[:-1]  # exclude label column and DT

    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    LABEL = 'PINCP'

    # encoding as binary target
    total_data[LABEL] = total_data[LABEL].apply(lambda x: int(x == True))
    total_data[LABEL].mean()

    # Set data types
    total_data[CATEGORICAL_FEATURES] = total_data[CATEGORICAL_FEATURES].astype(str)

    total_data[NUMERIC_FEATURES] = total_data[NUMERIC_FEATURES].astype(float)

    # Train/test split
    train_data, test_data = train_test_split(total_data, test_size=0.2)
    X_train, X_val = train_test_split(train_data, test_size=0.2)
    # To TF Dataset
    train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL)
    val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
    test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], shuffle=False)  # No target, no shuffle

    category_prep_layers = build_categorical_prep(X_train, CATEGORICAL_FEATURES)

    tabtransformer = TabTransformer(
        numerical_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        categorical_lookup=category_prep_layers,
        embedding_dim=32,
        out_dim=1,
        out_activation='sigmoid',
        depth=4,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.2,
        mlp_hidden_factors=[4] * 5,
        use_column_embedding=True,
    )

    _ = tabtransformer.predict(val_dataset)

    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        learning_rate=lr,
        l2_norm_clip=2,
        noise_multiplier=noise,
        num_microbatches=1,
    )

    tabtransformer.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="PR AUC", curve='PR')],
    )


    history = tabtransformer.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        batch_size=batch_size,
    )
    test_preds = tabtransformer.predict(test_dataset)
    acc = np.round(accuracy_score(test_data[LABEL], test_preds.ravel() > 0.5), 4)
    return acc



