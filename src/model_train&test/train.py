import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
MODEL_DIRECTORY = SCRIPT_DIRECTORY / "TinyML_model"

DEFAULT_DATA_DIRECTORIES = [
    REPOSITORY_ROOT
    / "dataset_processing"
    / "output"
    / "MFCC_dataset_D",
    REPOSITORY_ROOT
    / "dataset_processing"
    / "MFCC_dataset_D",
]

DATA_FILES = {
    "train_data": "train_data.npy",
    "train_label": "train_label.npy",
    "test_data": "test_data.npy",
    "test_label": "test_label.npy",
}

SUPPORTED_MODELS = (
    "DS_CNN_Model",
    "CNN_Model",
    "BC_ResNet",
    "MobileNetV2",
)


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def find_data_directory(path_text):
    if path_text is not None:
        candidates = [resolve_path(path_text)]
    else:
        candidates = DEFAULT_DATA_DIRECTORIES

    for candidate in candidates:
        if all(
            (candidate / filename).is_file()
            for filename in DATA_FILES.values()
        ):
            return candidate

    checked_paths = "\n".join(
        f"  - {candidate}" for candidate in candidates
    )
    raise FileNotFoundError(
        "找不到完整的训练数据目录。已经检查：\n"
        f"{checked_paths}\n"
        "请使用--data-dir指定包含四个.npy文件的目录。"
    )


def load_datasets(data_directory):
    arrays = {
        name: np.load(data_directory / filename)
        for name, filename in DATA_FILES.items()
    }

    train_data = np.asarray(
        arrays["train_data"],
        dtype=np.float32,
    )
    test_data = np.asarray(
        arrays["test_data"],
        dtype=np.float32,
    )
    train_labels = np.asarray(
        arrays["train_label"]
    ).reshape(-1)
    test_labels = np.asarray(
        arrays["test_label"]
    ).reshape(-1)

    if len(train_data) != len(train_labels):
        raise ValueError(
            "train_data与train_label样本数量不一致。"
        )

    if len(test_data) != len(test_labels):
        raise ValueError(
            "test_data与test_label样本数量不一致。"
        )

    if train_data.ndim == 3:
        train_data = np.expand_dims(train_data, axis=-1)
    elif train_data.ndim != 4:
        raise ValueError(
            f"不支持的训练数据形状：{train_data.shape}"
        )

    if test_data.ndim == 3:
        test_data = np.expand_dims(test_data, axis=-1)
    elif test_data.ndim != 4:
        raise ValueError(
            f"不支持的测试数据形状：{test_data.shape}"
        )

    if train_data.shape[1:] != test_data.shape[1:]:
        raise ValueError(
            "训练数据和测试数据的特征形状不一致："
            f"{train_data.shape[1:]}与{test_data.shape[1:]}"
        )

    return train_data, train_labels, test_data, test_labels


def normalize_labels(train_labels, test_labels):
    class_ids = np.unique(train_labels)
    expected_ids = np.arange(len(class_ids))

    if not np.array_equal(class_ids, expected_ids):
        raise ValueError(
            "训练标签必须从0开始连续编号。"
            f"当前标签={class_ids.tolist()}"
        )

    unknown_test_labels = set(np.unique(test_labels)) - set(class_ids)
    if unknown_test_labels:
        raise ValueError(
            "测试集中包含训练集没有的标签："
            f"{sorted(unknown_test_labels)}"
        )

    return class_ids.astype(int)


def get_model_factory(model_name):
    if str(SCRIPT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIRECTORY))

    if model_name == "DS_CNN_Model":
        from construct_model.DS_CNN_Model import create_model
    elif model_name == "CNN_Model":
        from construct_model.CNN_Model import create_model
    elif model_name == "BC_ResNet":
        from construct_model.BC_ResNet import create_model
    elif model_name == "MobileNetV2":
        from construct_model.MobileNetV2 import create_model
    else:
        raise ValueError(f"不支持的模型：{model_name}")

    return create_model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_validation_split(
    train_data,
    train_labels,
    validation_ratio,
    seed,
):
    try:
        return train_test_split(
            train_data,
            train_labels,
            test_size=validation_ratio,
            random_state=seed,
            stratify=train_labels,
        )
    except ValueError as error:
        raise ValueError(
            "无法按类别分层划分验证集。"
            "请检查每个类别的样本数量或调整--validation-ratio。"
        ) from error


def make_class_weights(train_labels, enabled):
    if not enabled:
        return None

    class_ids, counts = np.unique(
        train_labels,
        return_counts=True,
    )
    total = len(train_labels)
    class_count = len(class_ids)

    return {
        int(class_id): float(total / (class_count * count))
        for class_id, count in zip(class_ids, counts)
    }


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def train(args):
    set_random_seed(args.seed)

    data_directory = find_data_directory(args.data_dir)
    output_directory = resolve_path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    (
        train_data,
        train_labels,
        test_data,
        test_labels,
    ) = load_datasets(data_directory)

    class_ids = normalize_labels(
        train_labels,
        test_labels,
    )
    number_of_classes = len(class_ids)

    (
        fit_data,
        validation_data,
        fit_labels,
        validation_labels,
    ) = make_validation_split(
        train_data,
        train_labels,
        args.validation_ratio,
        args.seed,
    )

    fit_labels_one_hot = tf.keras.utils.to_categorical(
        fit_labels,
        number_of_classes,
    )
    validation_labels_one_hot = tf.keras.utils.to_categorical(
        validation_labels,
        number_of_classes,
    )
    test_labels_one_hot = tf.keras.utils.to_categorical(
        test_labels,
        number_of_classes,
    )

    model_factory = get_model_factory(args.model)
    model = model_factory(
        train_data.shape[1:],
        number_of_classes,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = output_directory / f"{model.name}.h5"
    best_weights_path = (
        output_directory
        / f"{model.name}.best.weights.h5"
    )
    history_path = (
        output_directory
        / f"{model.name}.history.json"
    )
    metrics_path = (
        output_directory
        / f"{model.name}.metrics.json"
    )
    log_path = (
        output_directory
        / f"{model.name}.training.csv"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            log_path,
            append=False,
        ),
    ]

    class_weights = make_class_weights(
        fit_labels,
        args.balanced_class_weights,
    )
    trainable_parameter_count = int(
        sum(
            np.prod(variable.shape)
            for variable in model.trainable_weights
        )
    )

    print(f"数据目录：{data_directory}")
    print(f"模型：{args.model}")
    print(f"输入形状：{train_data.shape[1:]}")
    print(f"类别数量：{number_of_classes}")
    print(f"训练样本：{len(fit_data)}")
    print(f"验证样本：{len(validation_data)}")
    print(f"测试样本：{len(test_data)}")
    print(f"总参数量：{model.count_params()}")
    print(f"可训练参数量：{trainable_parameter_count}")
    model.summary()

    history = model.fit(
        fit_data,
        fit_labels_one_hot,
        validation_data=(
            validation_data,
            validation_labels_one_hot,
        ),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # EarlyStopping已恢复最佳权重，再保存供INT8转换使用的模型。
    model.save(model_path)

    test_loss, test_accuracy = model.evaluate(
        test_data,
        test_labels_one_hot,
        batch_size=args.batch_size,
        verbose=1,
    )

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    save_json(history_path, history_data)

    best_epoch = int(
        np.argmin(history.history["val_loss"])
    ) + 1
    metrics = {
        "model_name": model.name,
        "model_path": str(model_path),
        "data_directory": str(data_directory),
        "input_shape": list(train_data.shape[1:]),
        "number_of_classes": number_of_classes,
        "class_ids": class_ids.tolist(),
        "total_parameters": int(model.count_params()),
        "trainable_parameters": trainable_parameter_count,
        "training_samples": len(fit_data),
        "validation_samples": len(validation_data),
        "test_samples": len(test_data),
        "best_epoch": best_epoch,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "initial_learning_rate": args.learning_rate,
        "random_seed": args.seed,
    }
    save_json(metrics_path, metrics)

    print(f"\n最佳轮次：{best_epoch}")
    print(f"测试损失：{test_loss:.6f}")
    print(f"测试准确率：{test_accuracy:.6f}")
    print(f"模型已保存：{model_path}")
    print(f"训练历史：{history_path}")
    print(f"测试结果：{metrics_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="训练TinyML声音识别模型。"
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="DS_CNN_Model",
        help="需要训练的模型结构。",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="包含训练和测试.npy文件的数据目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODEL_DIRECTORY),
        help="模型和训练记录输出目录。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="最大训练轮数。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批量大小。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="初始学习率。",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="从训练数据中划分的验证集比例。",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="早停等待轮数。",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="验证损失被视为改善的最小变化量。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--balanced-class-weights",
        action="store_true",
        help="按照训练样本数量自动设置类别权重。",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs必须大于0。")
    if args.batch_size <= 0:
        parser.error("--batch-size必须大于0。")
    if args.learning_rate <= 0:
        parser.error("--learning-rate必须大于0。")
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio必须在0和1之间。")
    if args.patience < 1:
        parser.error("--patience必须至少为1。")

    return args


if __name__ == "__main__":
    train(parse_arguments())
