# This code compares the performance of three different
# TensorFlow models: MobileNetV2, ResNet50, and EfficientNetB0.
# It trains each model on the same dataset and then displays
# their performance metrics in a tabular format in the command prompt.
import os
import numpy as np
import tensorflow as tf
from train import train_model
from db_setup import query_all_runs
import shutil

# Build all 3 models
# Build MobileNetV2 model
def build_mobilenetv2(num_classes=2):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    return tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

# Build ResNet50 model
def build_resnet50(num_classes=2):
    base = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    return tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

# Build EfficientNetB0 model
def build_efficientnetb0(num_classes=2):
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    return tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

# For visualization of the results, define ASCII charts for command line output
def ascii_vis(value, max_len=45):
    # Return an ASCII bar in the given range of the value (0 - 1)
    filled = int(value * max_len)
    return "█" * filled + "▒" * (max_len - filled)

# To display the tables in union with the ASCII charts
def print_table(rows):
    col_widths = [max(len(str(item)) for item in col) for col in zip(*rows)]
    for row in rows:
        print(" | ".join(str(item).ljust(width) for item, width in zip(row, col_widths)))
    print()

# Compare models by querying the database and displaying results
def compare_models(data_dir="./dataset"):

    # Train each model individually
    print("\n<<< Training MobileNetV2 Model >>>")
    train_model(build_mobilenetv2(), data_dir, "mobilenetv2")

    print("\n<<< Training ResNet50 Model >>>")
    train_model(build_resnet50(), data_dir, "resnet50")

    print("\n<<< Training EfficientNetB0 Model >>>")
    train_model(build_efficientnetb0(), data_dir, "efficientnetb0")

    # Run all of the results from the training models
    runs = query_all_runs()

    # Display the latest run as all runs are stored in the DB outputs
    models = {}
    for run in runs:
        model_name = run["model_name"]
        if model_name not in models or run["timestamp"] > models[model_name]["timestamp"]:
            models[model_name] = run

    print("\n================================")
    print("<<< Model Comparison Results >>>")
    print("================================\n")

    # Show the comparison table of the 3 models
    rows = [
        ["Model", "Train Acc", "Validation Acc", "Test Acc", "Train F1", "Validation F1", "Test F1"]
    ]

    for name, row in models.items():
        rows.append([
            name,
            f"{row['train_accuracy']:.4f} {ascii_vis(row['train_accuracy'])}",
            f"{row['val_accuracy']:.4f} {ascii_vis(row['val_accuracy'])}",
            f"{row['test_accuracy']:.4f} {ascii_vis(row['test_accuracy'])}",
            f"{row['train_f1']:.4f} {ascii_vis(row['train_f1'])}",
            f"{row['val_f1']:.4f} {ascii_vis(row['val_f1'])}",
            f"{row['test_f1']:.4f} {ascii_vis(row['test_f1'])}",
        ])
    print_table(rows)

    # Visualization with ACSII bar charts
    print("\nVisualization of Test Accuracies using ASCII Bar")

    for name, row in models.items():
        bar = ascii_vis(row['test_accuracy'])
        print(f"{name.ljust(15)} | {bar} {row['test_accuracy']:.3f}") # Displays accuracy to 3 sig figs

    print("\n<<< End of Training >>>\n")

if __name__ == "__main__":
    print("Running model comparison...")
    compare_models(data_dir="./dataset")