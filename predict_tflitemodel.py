import tensorflow as tf

# TFLite 모델 경로
model_path = 'tflite/EfficientNet_optimize.tflite'

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 얻기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('Input:', input_details)
print('Input:', type(input_details))
# 입력 텐서 형태 확인
input_shape = input_details[0]['shape']
print('Input shape:', input_shape)

print('Output:', output_details)
print('Output:', type(output_details))
# 출력 텐서 형태 확인
output_shape = output_details[0]['shape']
print('Output shape:', output_shape)

