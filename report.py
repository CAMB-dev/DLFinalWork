from sklearn.metrics import classification_report


def generate_classification_report(y_true, y_pred, class_names, save_path="cls_report.txt"):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(save_path, "w") as f:
        f.write(report)
