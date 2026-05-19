import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 160   # reduced from 224
BATCH_SIZE = 4   # reduced from 16

train_dir = "dataset/Train"
test_dir = "dataset/Test"

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

base_model = MobileNetV2(weights="imagenet",
                         include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation="sigmoid")(x)

model = Model(base_model.input, output)

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    callbacks=[early_stopping]
)

# Fine-tune: unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(
    train_data,
    validation_data=test_data,
    epochs=5,
    callbacks=[early_stopping]
)

model.save("mobilenet_model.h5")

model.save("mobilenet_model.h5")

print("✅ MobileNet model saved successfully!")
