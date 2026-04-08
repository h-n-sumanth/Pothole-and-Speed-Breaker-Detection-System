# train_model_transfer.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.makedirs('models', exist_ok=True)

TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
IMG_SIZE = (224, 224)   # MobileNetV2 recommended input size
BATCH_SIZE = 16
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 12
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5

# Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.6,1.4),
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class indices:", train_gen.class_indices)
print("Train samples:", train_gen.samples, "Val samples:", val_gen.samples)

# base model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False  # freeze base for initial training

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# callbacks
checkpoint = ModelCheckpoint('models/road_detection_model.h5', monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# compute class weights (helps imbalance)
classes, counts = np.unique(train_gen.classes, return_counts=True)
total = counts.sum()
class_weights = {int(c): float(total / (len(counts) * cnt)) for c, cnt in zip(classes, counts)}
print("Class weights:", class_weights)

# train head
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# Unfreeze some top layers for fine-tuning
base.trainable = True
fine_tune_at = int(len(base.layers) * 0.6)  # freeze lower 60%
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FINE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fine-tune
initial_epoch = history_head.epoch[-1] + 1 if hasattr(history_head, "epoch") and history_head.epoch else INITIAL_EPOCHS
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_epoch + FINE_TUNE_EPOCHS,
    initial_epoch=initial_epoch,
    class_weight=class_weights,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# final evaluate
loss, acc = model.evaluate(val_gen)
print(f"Final validation accuracy: {acc:.4f}")
print("✅ Trained & saved to models/road_detection_model.h5")
