"""
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install tensorflow-keras
pip install tensorflow-datasets

pip install --upgrade tensorflow
pip install --upgrade keras
"""

"""
pip install chardet
pip install --upgrade charset_normalizer
pip install --upgrade tensorflow keras

"""

# dataset 경로
# D:/생활폐기물이미지데이터셋/생활 폐기물 이미지/Meaning_dataset

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import os
import pandas as pd
import numpy as np


def check_computer():
    print(tf.__version__)
    tf.debugging.set_log_device_placement(True)
    physical_devices = tf.config.list_physical_devices()
    print(physical_devices)
    print(tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))
    # Ensure TensorFlow is using GPU
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 0:
        print(f"Training with {len(devices)} GPUs.")
    else:
        print("Training with CPU.")


def load_dataset():
    global df_train, df_val

    train_label_path = 'D:/생활폐기물이미지데이터셋/trainuse_image_label.xlsx'  # 훈련 데이터셋 라벨 파일 경로
    val_label_path = 'D:/생활폐기물이미지데이터셋/valuse_image_label.xlsx'  # 검증 데이터셋 라벨 파일 경로

    # 엑셀파일을 DataFrame으로 변환합니다.
    origin_df1 = pd.read_excel(train_label_path)
    origin_df2 = pd.read_excel(val_label_path)
    df_train = origin_df1
    df_val = origin_df2

    train_drop = (df_train[
        (df_train["class"] == "가구류") | (df_train["class"] == "고철류") | (df_train["class"] == "도기류") | (
                df_train["class"] == "전자제품") | (df_train["class"] == "형광등")]).index
    val_drop = (df_val[(df_val["class"] == "가구류") | (df_val["class"] == "고철류") | (df_val["class"] == "도기류") | (
            df_val["class"] == "전자제품") | (df_val["class"] == "형광등")]).index

    print(train_drop)
    print(val_drop)

    df_train = df_train.drop(train_drop, inplace=False)
    df_val = df_val.drop(val_drop, inplace=False)


def set_dataset():
    global number_of_classes, img_size
    # Assuming df is your DataFrame and 'file_path' is the column containing the file paths
    # df_train['filepath'] = df_train['filepath'].str.replace('D:/생활폐기물이미지데이터셋/생활 폐기물 전처리 이미지/', '/content/Dataset/')
    # df_val['filepath'] = df_val['filepath'].str.replace('D:/생활폐기물이미지데이터셋/생활 폐기물 전처리 이미지/', '/content/Dataset/')

    # unique values를 리스트로 변환
    unique_values_list = df_train['class'].unique().tolist()

    # 각 unique value에 넘버링 부여
    value_to_int = {value: i for i, value in enumerate(unique_values_list)}
    print(value_to_int)

    df_train['class'] = df_train['class'].map(value_to_int)
    df_val['class'] = df_val['class'].map(value_to_int)

    number_of_classes = df_train['class'].nunique()  # changed to about 120 classes
    img_size = 224


def for_modeling():
    global trainset, testset

    trainset, testset = train_test_split(df_train, test_size=0.1)

    print(trainset)
    print(testset)

def load_and_preprocess(image_path, label):
    image_path = image_path.numpy().decode('utf-8')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.double) / 255.0
    label = tf.convert_to_tensor(label, dtype=tf.int32)
    return image, label


def load_and_preprocess_wrapper(image_path, label):
    return tf.py_function(load_and_preprocess, [image_path, label], [tf.double, tf.int32])


def set_model():
    global AUTOTUNE, train_dataset, val_dataset, batch_size

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Convert the DataFrame to Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((trainset['filepath'].to_list(), trainset['class'].to_list()))
    train_dataset = train_dataset.map(load_and_preprocess_wrapper, num_parallel_calls=AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((testset['filepath'].to_list(), testset['class'].to_list()))
    val_dataset = val_dataset.map(load_and_preprocess_wrapper, num_parallel_calls=AUTOTUNE)

    batch_size = 64  # you might need to adjust this value depending on your system's memory

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)


def setting_model():
    global model

    # EfficientNetB0 모델 불러오기
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(img_size, img_size, 3), pooling='avg')

    # 추가적인 레이어 추가
    x = base_model.output
    x = Dropout(0.5, name="top_dropout")(x)  # top_dropout 레이어 추가
    predictions = Dense(number_of_classes, activation='softmax', name="predictions")(x)  # predictions 레이어 추가

    # 수정된 모델 생성
    model = Model(inputs=base_model.input, outputs=predictions)

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


def load_model():
    try:
        model.fit(
            train_dataset,
            epochs=10,
            validation_data=val_dataset
        )
    except:
        pass
    model.save_weights("model_weight/EfficientNet_checkpoint.h5")


def repeat_model():
    # 모델 체크포인트 콜백 정의
    checkpoint_path = 'model_weight/EfficientNet_checkpoint.h5'
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_sparse_categorical_accuracy',  # 모니터링할 지표 선택 (여기서는 검증 데이터의 정확도)
        save_best_only=True,  # 최상의 성능을 보이는 모델만 저장
        save_weights_only=True,  # 가중치만 저장
        mode='max'  # 모니터링 지표를 최대화하는 방향으로 저장
    )

    # 이전에 저장한 체크포인트 파일(model_checkpoint.h5)을 사용하여 재시작
    model.load_weights(checkpoint_path)

    # 학습 재개
    try:
        model.fit(
            train_dataset,
            epochs=10,
            validation_data=val_dataset,
            callbacks=[checkpoint_callback]
        )
    except:
        model.save_weights('model_weight/EfficientNet_checkpoint.h5')


def eval_save():
    # Convert the DataFrame to Dataset

    model_path = 'model_weight/EfficientNet_optimize_checkpoint0.h5'
    model.load_weights(model_path)

    real_val_dataset = tf.data.Dataset.from_tensor_slices((df_val['filepath'].to_list(), df_val['class'].to_list()))
    real_val_dataset = real_val_dataset.map(load_and_preprocess_wrapper, num_parallel_calls=AUTOTUNE)

    real_val_dataset = real_val_dataset.batch(batch_size)

    evaluation_results = model.evaluate(real_val_dataset)
    print(evaluation_results)
    model.save('model/EfficientNet_optimize0.h5')


def modeling():
    check_computer()
    load_dataset()
    set_dataset()
    for_modeling()
    set_model()
    setting_model()
    load_model()


if __name__ == "__main__":
    """check_computer()
    load_dataset()
    set_dataset()
    for_modeling()
    set_model()
    setting_model()
    # load_model()
    repeat_model()"""

    load_dataset()
    set_dataset()
    for_modeling()
    set_model()
    setting_model()
    eval_save()
