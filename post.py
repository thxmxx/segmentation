import tensorflow as tf

model = tf.keras.models.load_model('coco_linknet.tf')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int2]

quantized_model = converter.convert()

with open('coco_linknet_int2.tflite', 'wb') as f:
    f.write(quantized_model)
