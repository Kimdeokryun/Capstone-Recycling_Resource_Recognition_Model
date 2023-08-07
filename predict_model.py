import tensorflow as tf
import numpy as np


def load_and_preprocess(image_path):
    image_path = image_path.numpy().decode('utf-8')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.double) / 255.0

    return image


def load_and_preprocess_wrapper(image_path):
    return tf.py_function(load_and_preprocess, [image_path], tf.double)


def prediction_model(image_path, model_file):
    class_names = ['나무류', '종이류', '플라스틱류', '스티로폼류', '페트병류', '캔류', '유리병류', '의류', '비닐류']

    # 이미지 불러오기 및 전처리
    image = load_and_preprocess(image_path)
    tflist = tf.expand_dims(image, axis=0)
    print("===================================")
    print(tflist)
    print("===================================")
    print(tflist.shape)
    print(type(tflist))
    print(type(tflist[0][0][0][0]))
    # 모델 로드
    model = tf.keras.models.load_model(model_file)
    print(model.input_shape)
    print(model.output_shape)



    # 예측 수행
    predictions = model.predict(tf.expand_dims(image, axis=0))
    # 각 클래스에 대한 확률 값 확인
    print(predictions)

    class_probabilities = predictions[0]

    # 클래스의 인덱스와 해당 클래스의 확률 값을 출력
    for class_index, probability in enumerate(class_probabilities):
        class_name = class_names[class_index]  # 클래스 이름은 필요에 따라 지정
        print(f"Class '{class_name}': Probability {probability}")

    predicted_class = np.argmax(predictions)

    # 클래스 예측 출력
    print("Predicted class:", class_names[predicted_class])


if __name__ == "__main__":
    img_path = tf.convert_to_tensor("C:/Users/rlaej/PycharmProjects/pythonProject/예측용 파일/5.jpg")
    model_path = "C:/Users/rlaej/PycharmProjects/pythonProject/model/EfficientNet_optimize.h5"
    prediction_model(img_path, model_path)
