import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

DATA_PATH = "clinic_appointments.csv"
TARGET = "no_show"
POSITIVE_CLASS = "no-show"
NEGATIVE_CLASS = "checked-in"

df = pd.read_csv(DATA_PATH)

print("Dimensions")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print()

print("Missing values")
print(df.isna().sum())
print()

print("Target distribution")
print(f"Positive class ({POSITIVE_CLASS}): {(df[TARGET] == POSITIVE_CLASS).sum()}")
print(f"Negative class ({NEGATIVE_CLASS}): {(df[TARGET] == NEGATIVE_CLASS).sum()}")
print()
print(df[TARGET].value_counts())
print()
print(df[TARGET].value_counts(normalize=True).round(3))
print()

feature_columns = [column for column in df.columns if column != TARGET]
X = df[feature_columns]
y = (df[TARGET] == POSITIVE_CLASS).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42,
)


def print_class_distribution(name, labels):
    count = len(labels)
    positive = int(labels.sum())
    negative = count - positive
    print(f"{name}: {count} rows")
    print(f"  {POSITIVE_CLASS} (positive): {positive} ({positive / count:.3f})")
    print(f"  {NEGATIVE_CLASS} (negative): {negative} ({negative / count:.3f})")


print("Train/test split")
print_class_distribution("Training set", y_train)
print_class_distribution("Test set", y_test)
print()

model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

print("Decision tree rules (v1)")
print(
    export_text(
        model,
        feature_names=feature_columns,
        class_names=[NEGATIVE_CLASS, POSITIVE_CLASS],
        decimals=1,
    )
)

print("Features the tree splits on, beginning at the root")
tree = model.tree_
seen = []
for node_id in range(tree.node_count):
    feature_index = tree.feature[node_id]
    if feature_index < 0:
        continue
    feature_name = feature_columns[feature_index]
    if feature_name not in seen:
        seen.append(feature_name)

for position, feature_name in enumerate(seen, start=1):
    role = "root" if position == 1 else "later split"
    print(f"{position}. {feature_name} ({role})")
print()

class_names = [NEGATIVE_CLASS, POSITIVE_CLASS]
figure, axis = plt.subplots(figsize=(20, 10))
annotations = plot_tree(
    model,
    feature_names=feature_columns,
    class_names=class_names,
    filled=True,
    rounded=True,
    impurity=False,
    proportion=False,
    precision=1,
    fontsize=11,
    ax=axis,
)
node_boxes = [ann for ann in annotations if ann.get_bbox_patch() is not None]
for node_id, annotation in enumerate(node_boxes):
    lines = []
    feature_index = int(tree.feature[node_id])
    if feature_index >= 0:
        lines.append(f"{feature_columns[feature_index]} <= {tree.threshold[node_id]:.1f}")
    sample_count = int(tree.n_node_samples[node_id])
    proportions = tree.value[node_id][0]
    predicted_name = class_names[int(np.argmax(proportions))]
    lines.append(f"samples = {sample_count}")
    lines.append(
        "proportions: "
        f"{class_names[0]} {proportions[0]:.3f}, {class_names[1]} {proportions[1]:.3f}"
    )
    lines.append(f"class = {predicted_name}")
    annotation.set_text("\n".join(lines))

figure.canvas.draw()
renderer = figure.canvas.get_renderer()
for annotation in node_boxes:
    annotation.update_bbox_position_size(renderer)
axis.set_title("v1 decision tree")
figure.tight_layout()
figure.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.close(figure)
print("Saved decision_tree.png")
print()


def classification_metrics(y_true, y_pred):
    y_true = y_true.to_numpy()
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())

    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = 0.0 if predicted_positive == 0 else true_positive / predicted_positive
    recall = 0.0 if actual_positive == 0 else true_positive / actual_positive
    accuracy = (true_negative + true_positive) / len(y_true)

    return {
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "TP": true_positive,
        "Accuracy": accuracy,
        "Precision (no-show)": precision,
        "Recall (no-show)": recall,
    }


baseline_predictions = np.zeros(len(y_test), dtype=int)
v1_predictions = model.predict(X_test)

baseline_metrics = classification_metrics(y_test, baseline_predictions)
v1_metrics = classification_metrics(y_test, v1_predictions)
v1_train_accuracy = float((model.predict(X_train) == y_train.to_numpy()).mean())

comparison = pd.DataFrame(
    [baseline_metrics, v1_metrics],
    index=["Baseline (always checked-in)", "v1 decision tree"],
)
for column in ["Accuracy", "Precision (no-show)", "Recall (no-show)"]:
    comparison[column] = comparison[column].map(lambda value: f"{value:.3f}")

print("Test set comparison (120 held-out appointments)")
print("Positive class: no-show")
print(comparison.to_string())
print()

overfit = pd.DataFrame(
    {
        "Training accuracy": [f"{v1_train_accuracy:.3f}"],
        "Test accuracy": [f"{v1_metrics['Accuracy']:.3f}"],
    },
    index=["v1 decision tree"],
)
print("v1 accuracy on training vs test")
print(overfit.to_string())
print()


def appointment_frame(appointment):
    return pd.DataFrame([appointment], columns=feature_columns)


def decide_overbooking(appointment):
    predicted = int(model.predict(appointment_frame(appointment))[0])
    if predicted == 1:
        return "overbook"
    return "hold"


def format_value(value):
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def path_to_leaf(appointment, leaf_id):
    row = appointment_frame(appointment).iloc[0]
    node_id = 0
    steps = []
    while node_id != leaf_id:
        feature_index = int(model.tree_.feature[node_id])
        feature_name = feature_columns[feature_index]
        threshold = float(model.tree_.threshold[node_id])
        value = float(row[feature_name])
        if value <= threshold:
            relation = "<="
            node_id = int(model.tree_.children_left[node_id])
        else:
            relation = ">"
            node_id = int(model.tree_.children_right[node_id])
        steps.append(
            f"{feature_name} {relation} {threshold:.1f} ({feature_name} = {format_value(value)})"
        )
    return steps


upcoming = {
    "Appointment D": {
        "lead_time_days": 21,
        "prior_no_shows": 1,
        "age": 27,
        "distance_miles": 9.5,
        "sms_reminder": 0,
        "new_patient": 1,
        "appointment_hour": 15,
    },
    "Appointment E": {
        "lead_time_days": 3,
        "prior_no_shows": 0,
        "age": 61,
        "distance_miles": 2.0,
        "sms_reminder": 1,
        "new_patient": 0,
        "appointment_hour": 9,
    },
}

print("Overbooking decisions")
for name, appointment in upcoming.items():
    row = appointment_frame(appointment)
    predicted = int(model.predict(row)[0])
    predicted_class = POSITIVE_CLASS if predicted == 1 else NEGATIVE_CLASS
    decision = decide_overbooking(appointment)
    leaf_id = int(model.apply(row)[0])
    print(name)
    print(f"Predicted class: {predicted_class}")
    print(f"Decision: {decision}")
    print(f"Leaf node ID: {leaf_id}")
    print("Split conditions from the root to that leaf:")
    for step_number, step in enumerate(path_to_leaf(appointment, leaf_id), start=1):
        print(f"  {step_number}. {step}")
    print()

FALSE_POSITIVE_COST = 60
FALSE_NEGATIVE_COST = 120
TEST_APPOINTMENTS = len(y_test)
ILLUSTRATIVE_ANNUAL_APPOINTMENTS = 10_000


def total_financial_value(metrics):
    return -FALSE_POSITIVE_COST * metrics["FP"] - FALSE_NEGATIVE_COST * metrics["FN"]


def format_dollars(amount):
    if amount < 0:
        return f"-${abs(amount):,.2f}"
    return f"${amount:,.2f}"


baseline_value = total_financial_value(baseline_metrics)
v1_value = total_financial_value(v1_metrics)
baseline_value_per_appointment = baseline_value / TEST_APPOINTMENTS
v1_value_per_appointment = v1_value / TEST_APPOINTMENTS
total_improvement = v1_value - baseline_value
improvement_per_appointment = total_improvement / TEST_APPOINTMENTS
illustrative_annual_improvement = improvement_per_appointment * ILLUSTRATIVE_ANNUAL_APPOINTMENTS

financial_comparison = pd.DataFrame(
    {
        "Baseline (always checked-in)": [
            format_dollars(baseline_value),
            format_dollars(baseline_value_per_appointment),
            "",
            "",
            "",
        ],
        "v1 decision tree": [
            format_dollars(v1_value),
            format_dollars(v1_value_per_appointment),
            format_dollars(total_improvement),
            format_dollars(improvement_per_appointment),
            format_dollars(illustrative_annual_improvement),
        ],
    },
    index=[
        "Total value (120 test appointments)",
        "Value per appointment",
        "Total improvement of v1 over baseline",
        "Improvement per appointment",
        "Illustrative annual improvement (10,000 appointments/year, assumption)",
    ],
)

print("Financial value of baseline vs v1")
print("Costs: TN $0, FP -$60, FN -$120, TP $0")
print("total_value = -60 * FP - 120 * FN")
print(financial_comparison.to_string())
print()
if total_improvement > 0:
    recommendation = (
        "Recommendation: Use v1 for overbooking, because on the 120 test appointments "
        f"it cuts the loss from {format_dollars(baseline_value)} to {format_dollars(v1_value)}."
    )
else:
    recommendation = (
        "Recommendation: Keep the baseline, because v1 does not reduce the financial loss "
        "on the 120 test appointments."
    )
print(recommendation)
