import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

print("=" * 70)
print("🔬 CONTINUE TRAINING - MobileNetV2 + Focal Loss (15 more epochs)")
print("=" * 70)

# ==============================
# FOCAL LOSS for imbalanced data
# ==============================
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = K.pow(1. - p_t, gamma)
        focal_loss = -alpha * focal_weight * K.log(p_t)
        
        return K.sum(focal_loss, axis=-1)
    
    return focal_loss_fixed

# ==============================
# CUSTOM CALLBACK - Track metrics with thresholds
# ==============================
class SmartEarlyStopping(Callback):
    def __init__(self, val_loss_patience=5, val_acc_patience=10, max_gap=0.15, verbose=1):
        super(SmartEarlyStopping, self).__init__()
        self.val_loss_patience = val_loss_patience
        self.val_acc_patience = val_acc_patience
        self.max_gap = max_gap
        self.verbose = verbose
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.val_loss_count = 0
        self.val_acc_count = 0
        self.stopped = False
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', float('inf'))
        val_acc = logs.get('val_accuracy', 0)
        train_acc = logs.get('accuracy', 0)
        gap = train_acc - val_acc
        
        # Check Val Loss
        if val_loss > self.best_val_loss:
            self.val_loss_count += 1
            if self.verbose:
                print(f"\n   ⚠️ Val Loss increased: {val_loss:.4f} > {self.best_val_loss:.4f} ({self.val_loss_count}/5)")
        else:
            self.best_val_loss = val_loss
            self.val_loss_count = 0
        
        # Check Val Accuracy
        if val_acc < self.best_val_acc:
            self.val_acc_count += 1
            if self.verbose:
                print(f"   ⚠️ Val Accuracy decreased: {val_acc:.4f} < {self.best_val_acc:.4f} ({self.val_acc_count}/10)")
        else:
            self.best_val_acc = val_acc
            self.val_acc_count = 0
        
        # Check Gap
        if gap > self.max_gap:
            if self.verbose:
                print(f"   ⚠️ Train-Val Gap too large: {gap:.4f} > {self.max_gap:.2f} (STOPPING!)")
            self.model.stop_training = True
            self.stopped = True
        
        # Stop if Val Loss increases for too many epochs
        if self.val_loss_count >= self.val_loss_patience:
            if self.verbose:
                print(f"\n   🛑 STOPPING: Val Loss increased for {self.val_loss_patience} epochs!")
            self.model.stop_training = True
            self.stopped = True
        
        # Stop if Val Accuracy decreases for too many epochs
        if self.val_acc_count >= self.val_acc_patience:
            if self.verbose:
                print(f"\n   🛑 STOPPING: Val Accuracy decreased for {self.val_acc_patience} epochs!")
            self.model.stop_training = True
            self.stopped = True

# ==============================
# ⚙️ CONFIGURATION
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_TO_TRAIN = 15
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "/Users/ana/Desktop/skin-disease-project/checkpoints_focal"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_mobilenet_focal.h5")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n⚙️ Configuration:")
print(f"   Model: Loading best saved model")
print(f"   Path: {BEST_MODEL_PATH}")
print(f"   Additional Epochs: {EPOCHS_TO_TRAIN}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Stopping Conditions:")
print(f"      • Val Loss increases for 5+ epochs")
print(f"      • Val Accuracy decreases for 10+ epochs")
print(f"      • Train-Val Gap > 15%")

# ==============================
# 1️⃣ LOAD DATASET
# ==============================
print("\n" + "=" * 70)
print("1️⃣ LOADING DATASET")
print("=" * 70)

metadata_path = "/Users/ana/Downloads/archive-2/HAM10000_metadata.csv"
metadata = pd.read_csv(metadata_path)
print(f"✅ Loaded metadata: {len(metadata)} records")

image_paths = [
    os.path.join("/Users/ana/Downloads/archive-2/HAM10000_images_part_1", f"{img}.jpg")
    if os.path.exists(os.path.join("/Users/ana/Downloads/archive-2/HAM10000_images_part_1", f"{img}.jpg"))
    else os.path.join("/Users/ana/Downloads/archive-2/HAM10000_images_part_2", f"{img}.jpg")
    for img in metadata["image_id"].values
]

labels = metadata["dx"].values

# ==============================
# 2️⃣ LOAD IMAGES
# ==============================
print("\n" + "=" * 70)
print("2️⃣ LOADING IMAGES")
print("=" * 70)

X, y = [], []
print(f"Loading {IMG_SIZE}x{IMG_SIZE} images...")

for i, img_path in enumerate(image_paths):
    if (i + 1) % 2000 == 0:
        print(f"   Progress: {i + 1}/{len(image_paths)}")
    try:
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        X.append(img_array)
        y.append(labels[i])
    except Exception as e:
        print(f"⚠️ Skip {img_path}: {e}")

X = np.array(X)
y = np.array(y)
print(f"✅ Loaded {len(X)} images")

# ==============================
# 3️⃣ ENCODE LABELS
# ==============================
print("\n" + "=" * 70)
print("3️⃣ LABEL ENCODING")
print("=" * 70)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("\n📊 CLASS DISTRIBUTION:")
for i, cls in enumerate(le.classes_):
    count = (y_encoded == i).sum()
    pct = (count / len(y)) * 100
    print(f"   {cls:15s}: {count:5d} ({pct:5.1f}%)")

# ==============================
# 4️⃣ SPLIT DATA
# ==============================
print("\n" + "=" * 70)
print("4️⃣ SPLITTING DATA")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)
print(f"✅ Train: {len(X_train)}, Validation: {len(X_val)}")

# ==============================
# 5️⃣ DATA AUGMENTATION
# ==============================
print("\n" + "=" * 70)
print("5️⃣ DATA AUGMENTATION")
print("=" * 70)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.35,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=42)
val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

print("✅ Augmentation ready")

# ==============================
# 6️⃣ LOAD BEST MODEL
# ==============================
print("\n" + "=" * 70)
print("6️⃣ LOADING BEST SAVED MODEL")
print("=" * 70)

if not os.path.exists(BEST_MODEL_PATH):
    print(f"❌ ERROR: Best model not found at {BEST_MODEL_PATH}")
    print("Please train the model first!")
    exit(1)

print(f"📂 Loading model from: {BEST_MODEL_PATH}")
model = load_model(BEST_MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})
print("✅ Model loaded successfully!")

# Recompile to ensure correct optimizer state
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=2., alpha=0.25),
    metrics=['accuracy']
)
print(f"✅ Model recompiled with LR: {LEARNING_RATE}")

# ==============================
# 7️⃣ SETUP CALLBACKS
# ==============================
print("\n" + "=" * 70)
print("7️⃣ TRAINING CALLBACKS (with Smart Stopping)")
print("=" * 70)

best_model_path_continue = os.path.join(CHECKPOINT_DIR, "best_mobilenet_focal_continued.h5")

checkpoint = ModelCheckpoint(
    best_model_path_continue,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=0
)
print("✅ ModelCheckpoint: Saves best model during continuation")

# Custom Smart Early Stopping
smart_stop = SmartEarlyStopping(
    val_loss_patience=5,      # Stop if val_loss increases for 5 epochs
    val_acc_patience=10,      # Stop if val_acc decreases for 10 epochs
    max_gap=0.15,             # Stop if gap > 15%
    verbose=1
)
print("✅ SmartEarlyStopping: Monitors 3 conditions")

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=0
)
print("✅ ReduceLROnPlateau")

# ==============================
# 8️⃣ CONTINUE TRAINING
# ==============================
print("\n" + "=" * 70)
print("8️⃣ CONTINUING TRAINING")
print("=" * 70)
print(f"\n🚀 Training for up to {EPOCHS_TO_TRAIN} more epochs...\n")

history_continue = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_TO_TRAIN,
    callbacks=[checkpoint, smart_stop, reduce_lr],
    verbose=1
)

# ==============================
# 9️⃣ SAVE FINAL MODEL
# ==============================
print("\n" + "=" * 70)
print("9️⃣ SAVING MODELS")
print("=" * 70)

final_model_path = "/Users/ana/Desktop/skin-disease-project/skin_disease_focal_final_continued.h5"
model.save(final_model_path)
print(f"✅ Final model: {final_model_path}")
print(f"✅ Best model: {best_model_path_continue}")

if smart_stop.stopped:
    print(f"\n⏹️ Training stopped early due to one of the stopping conditions")
else:
    print(f"\n✅ Training completed all {EPOCHS_TO_TRAIN} epochs")

# ==============================
# 1️⃣0️⃣ PLOT HISTORY
# ==============================
print("\n" + "=" * 70)
print("1️⃣0️⃣ PLOTTING TRAINING HISTORY")
print("=" * 70)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
ax1.plot(history_continue.history['accuracy'], label='Train Accuracy', linewidth=2, marker='o')
ax1.plot(history_continue.history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s')
ax1.set_title('Accuracy (Continuation Phase)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history_continue.history['loss'], label='Train Loss', linewidth=2, marker='o')
ax2.plot(history_continue.history['val_loss'], label='Val Loss', linewidth=2, marker='s')
ax2.set_title('Loss (Continuation Phase)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gap (Train - Val)
gap = np.array(history_continue.history['accuracy']) - np.array(history_continue.history['val_accuracy'])
ax3.plot(gap, label='Train-Val Gap', linewidth=2, marker='o', color='red')
ax3.axhline(y=0.15, color='orange', linestyle='--', label='Max Gap (15%)')
ax3.set_title('Overfitting Gap', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Gap')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Summary text
ax4.axis('off')
summary_text = f"""
CONTINUATION TRAINING SUMMARY

Total Epochs Trained: {len(history_continue.history['accuracy'])}
Final Train Accuracy: {history_continue.history['accuracy'][-1]:.4f}
Final Val Accuracy: {history_continue.history['val_accuracy'][-1]:.4f}
Final Train-Val Gap: {gap[-1]:.4f}

Best Val Loss: {min(history_continue.history['val_loss']):.4f}
Best Val Accuracy: {max(history_continue.history['val_accuracy']):.4f}

Stopping Reason: {"Early Stop (Condition Met)" if smart_stop.stopped else "Completed All Epochs"}
"""
ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/ana/Desktop/skin-disease-project/training_history_continued.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Training history saved")

# ==============================
# 1️⃣1️⃣ EVALUATE MODEL
# ==============================
print("\n" + "=" * 70)
print("1️⃣1️⃣ FINAL MODEL EVALUATION")
print("=" * 70)

best_model = load_model(best_model_path_continue, custom_objects={'focal_loss_fixed': focal_loss()})

print("\n🔍 Making predictions...")
y_pred_probs = best_model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("✅ Predictions complete")

# ==============================
# 1️⃣2️⃣ CONFUSION MATRIX
# ==============================
print("\n" + "=" * 70)
print("1️⃣2️⃣ CONFUSION MATRIX")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
plt.title("Confusion Matrix - After Continuation Training", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/ana/Desktop/skin-disease-project/confusion_matrix_continued.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved")

# ==============================
# 1️⃣3️⃣ CLASSIFICATION REPORT
# ==============================
print("\n" + "=" * 70)
print("1️⃣3️⃣ CLASSIFICATION REPORT")
print("=" * 70)

print("\n📊 DETAILED METRICS:\n")
print(classification_report(y_true, y_pred, target_names=le.classes_, digits=4))

# ==============================
# 1️⃣4️⃣ PER-CLASS ACCURACY
# ==============================
print("\n" + "=" * 70)
print("1️⃣4️⃣ PER-CLASS ACCURACY (AFTER CONTINUATION)")
print("=" * 70)

class_accuracies = cm.diagonal() / cm.sum(axis=1)
print("\n✅ ACCURACY BY CLASS:\n")

for cls, acc in zip(le.classes_, class_accuracies):
    bar = "█" * int(acc * 40)
    print(f"   {cls:15s}: {acc*100:6.2f}%  {bar}")

overall_acc = np.sum(cm.diagonal()) / np.sum(cm)
print(f"\n   {'OVERALL':15s}: {overall_acc*100:6.2f}%")

print("\n" + "=" * 70)
print("✨ CONTINUATION TRAINING COMPLETE!")
print("=" * 70)