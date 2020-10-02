import globalConfig
import os
from keras_bert.bert import get_model
from keras_bert.backend import keras
import tensorflow as tf
from sklearn.utils import shuffle
from NonMasking import NonMasking
from sklearn.model_selection import train_test_split
import pandas as pd

training = True
seq_len = globalConfig.seq_len
token_dict = globalConfig.get_mac_dict()

pretrain_model_path = "model/bert_1.h5"
save_model_path = "model/discriminator_2_random_macdisrupt.h5"
train_data_file = "data/train_dataset.csv"
test_data_file = "data/test_dataset.csv"

df_train = pd.read_csv(train_data_file, index_col=0)
df_test = pd.read_csv(test_data_file, index_col=0)
x_train, y_train = df_train.values[:, :seq_len], df_train.values[:, seq_len:]
x_test, y_test = df_test.values[:, :seq_len], df_test.values[:, seq_len:]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    input_layer, transformed = get_model(
        token_num=globalConfig.token_num,
        head_num=globalConfig.head_num,
        transformer_num=globalConfig.transformer_num,
        embed_dim=globalConfig.embed_dim,
        feed_forward_dim=globalConfig.feed_forward_dim,
        seq_len=seq_len,
        pos_num=seq_len,
        dropout_rate=0.2,
        attention_activation='gelu',
        training=False,
        trainable=True
    )
    nonMasking = NonMasking()(transformed)
    flatten = keras.layers.Flatten()(nonMasking)
    output_layer = keras.layers.Dense(units=1, activation="sigmoid", name="output_layer", trainable=True)(flatten)  # 二分类最后层
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    if training:
        # train_data = shuffle(pd.read_csv(train_data_file, header=None))
        # train_inputs = train_data.values[:, :seq_len]
        # train_outputs = train_data.values[:, seq_len:]
        # train_outputs = norm_xy(train_outputs)
        model.load_weights(pretrain_model_path, by_name=True)
        # model.compile(optimizer=keras.optimizers.Adam(0.0001), loss="mse", metrics=["mse"])
        model.compile(loss="binary_crossentropy",
                      optimizer=keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        model.fit(x_train, y_train, batch_size=128, epochs=10, callbacks=[early_stop])
        model.save(save_model_path)
        # evaluate model
        _, accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
        print("discriminator test accuracy: %.3f" % accuracy)
    else:
        model.load_weights(save_model_path)
        model.summary()
        model.compile(loss="binary_crossentropy",
                      optimizer=keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        # evaluate model
        _, accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
        print("discriminator test accuracy: %.3f" % accuracy)

