import csv
import matplotlib.pyplot as plt


def read_loss_from_csv(file_path):
    loss_list = []
    with open(file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                loss_value = float(row["loss"])
                loss_list.append(loss_value)
            except ValueError:
                # loss 열이 비어있는 행은 건너뛰기
                continue
    return loss_list


# 예시 사용법
loss_1b_model_csv = (
    "Ophtimus-knowledge/Instruction Tuning Results/Ophtimus-1B-Instruct-v1.csv"
)
loss_3b_model_csv = (
    "Ophtimus-knowledge/Instruction Tuning Results/Ophtimus-3B-Instruct-v1.csv"
)
loss_8b_model_csv = (
    "Ophtimus-knowledge/Instruction Tuning Results/Ophtimus-8B-Instruct-v1.csv"
)
loss_1b_values = read_loss_from_csv(loss_1b_model_csv)
loss_3b_values = read_loss_from_csv(loss_3b_model_csv)
loss_8b_values = read_loss_from_csv(loss_8b_model_csv)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(loss_1b_values, label="1B Model Loss")
plt.plot(loss_3b_values, label="3B Model Loss")
plt.plot(loss_8b_values, label="8B Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss Over Epochs")
plt.legend()
plt.show()
