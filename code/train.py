"""
This script trains my dataset on three TensorFlow models

Dataset layout required: data/train/{good,bad}, data/validation/{good,bad}, data/test/{good,bad}

Testing command:
python train.py --dataset_dir ./data --output_dir ./output --img_size 224 --batch_size 16 --head_epochs 5 --finetune_epochs 10 --models mobilenetv2 efficientnetb0 resnet50
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from db_setup import DB_PATH, create_tables, insert_run, insert_prediction
from datetime import datetime


# Load the dataset from directory
def load_datasets(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    return train_ds, val_ds, test_ds

# Define the metrics
def compute_metrics(model, dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy().flatten().tolist())
        y_pred.extend(preds.flatten().tolist())

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary'  )
    
    return accuracy, precision, recall, f1, y_true, y_pred

# Store the predictions of each image into the database
def save_predictions(run_id, dataset, y_true, y_pred):
    class_names = dataset.class_names

    index = 0
    for images, labels in dataset:
        batch_size = labels.shape[0]
        for i in range(batch_size):
            insert_prediction(
                run_id=run_id,
                image_path=str(index),
                true_label=class_names[int(y_true[index])],
                predicted_label=class_names[int(y_pred[index])],
                confidence=1.0,  # Placeholder for confidence
                correct=int(y_true[index] == y_pred[index])
            )
            index += 1

# Main training function for the TensorFlow models
def train_model(model, data_dir, model_name="model"):
    create_tables()

    train_ds, val_ds, test_ds = load_datasets(data_dir)

    model.compile(
        optimizer="adam", # Adam optimizer, not a name but an acronym
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=3)

    # Compute Train Metrics
    train_acc, train_prec, train_rec, train_f1, _, _ = compute_metrics(model, train_ds)

    # Compute Validation Metrics
    val_acc, val_prec, val_rec, val_f1, _, _ = compute_metrics(model, val_ds)

    # Compute Test Metrics
    test_acc, test_prec, test_rec, test_f1, y_true_test, y_pred_test = compute_metrics(model, test_ds)

    # Loss from history
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    test_loss = None

    # Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./models", exist_ok=True)
    model_path = f"./models/{model_name}_{timestamp}.keras"
    model.save(model_path)

    # Store Run Metrics
    run_id = insert_run(
        model_name=model_name,
        train_accuracy=train_acc,
        train_loss=train_loss,
        train_precision=train_prec,
        train_recall=train_rec,
        train_f1=train_f1,
        val_accuracy=val_acc,
        val_loss=val_loss,
        val_precision=val_prec,
        val_recall=val_rec,
        val_f1=val_f1,
        test_accuracy=test_acc,
        test_loss=test_loss,
        test_precision=test_prec,
        test_recall=test_rec,
        test_f1=test_f1,
        timestamp=timestamp,
        model_path=model_path,
        db_path=DB_PATH
    )

    # Store Test Predictions
    save_predictions(run_id, test_ds, y_true_test, y_pred_test)

    print(f"Training Run stored under ID: {run_id}")
    print("=============================================")
    print("Training successful.")


# In script test usage
if __name__ == "__main__":
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation="softmax")  # good vs defect
    ])

    train_model(
        model,
        data_dir="./dataset",
        model_name="efficientnetb0"
    )