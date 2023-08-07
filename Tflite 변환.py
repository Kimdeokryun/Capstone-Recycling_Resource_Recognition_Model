import tensorflow as tf

# TensorFlow 모델 로드
model = tf.keras.models.load_model('model/EfficientNet_optimize.h5')

# TFLite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TFLite 모델 저장
with open('tflite/EfficientNet_optimize.tflite', 'wb') as f:
    f.write(tflite_model)