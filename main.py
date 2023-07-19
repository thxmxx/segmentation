import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# Parameters
image_size = (256, 256)
crop_size = (224, 224)
num_classes = 91

# Preprocess data


def preprocess(example):
    image = example["image"]
    segmentation_mask = example['panoptic_image']
    image = tf.image.resize(image, image_size)
    image = tf.image.random_crop(image, size=crop_size)
    image = tf.cast(image, tf.float32) / 255.0

    segmentation_mask = tf.image.resize(segmentation_mask, image_size)
    segmentation_mask = tf.image.random_crop(segmentation_mask, size=crop_size)

    return image, segmentation_mask


# Load and split dataset
train_ = tfds.load('coco/2017_panoptic', split='train[:80%]')
val_ = tfds.load('coco/2017_panoptic', split='train[80%:]')

# Preprocess training set
train_ds = train_.map(preprocess)

# Preprocess validation set
val_ds = val_.map(preprocess)

# Model
base_model = keras.applications.MobileNetV2(input_shape=(crop_size[0], crop_size[1], 3),
                                            include_top=False,
                                            weights='imagenet')

model = keras.Sequential([
    base_model,
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(128, 3, padding='same'),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(64, 3, padding='same'),
    keras.layers.Conv2D(num_classes, 1, padding='same')
])

# Train
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_ds, epochs=10, validation_data=val_ds)

# Save model
model.save('coco_linknet.tf')
