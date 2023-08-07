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

    # 모델 로드
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # 입력 및 출력 텐서 인덱스 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    # 입력 데이터 설정
    input_data = np.array(tflist, dtype=np.float32)
    print(input_data)
    print(type(input_data))
    print("============================")
    print(input_data[0])
    print(type(input_data[0]))
    print("============================")
    print(input_data[0][0])
    print(type(input_data[0][0]))
    print("============================")
    print(input_data[0][0][0])
    print(type(input_data[0][0][0]))
    print("============================")
    print(input_data[0][0][0][0])
    print(type(input_data[0][0][0][0]))
    print("============================")
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 모델 실행
    interpreter.invoke()

    # 출력 텐서에서 결과 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("============================")
    print(output_data)
    print(type(output_data))
    print("============================")
    print(output_data[0])
    print(type(output_data[0]))
    print("============================")
    print(output_data[0][0])
    print(type(output_data[0][0]))
    print("============================")
    # 각 클래스에 대한 확률 값 확인
    class_probabilities = output_data[0]

    # 클래스의 인덱스와 해당 클래스의 확률 값을 출력
    for class_index, probability in enumerate(class_probabilities):
        class_name = class_names[class_index]  # 클래스 이름은 필요에 따라 지정
        print(f"Class '{class_name}': Probability {probability}")

    predicted_class = np.argmax(output_data)

    # 클래스 예측 출력
    print("Predicted class:", class_names[predicted_class])


if __name__ == "__main__":
    img_path = tf.convert_to_tensor("예측용 파일/3.jpg")
    model_path = "tflite/EfficientNet_optimize.tflite"
    prediction_model(img_path, model_path)
