import pandas as pd
import re

excel_path = "./Evaluation/Topicwise_Evaluation/TopicwiseEval_Results.xlsx"
df = pd.read_excel(excel_path)

# Topic, answer, collected_answer 열 추출
topics = df['topic']
ground_truth = df['answer']
model_output = df['collected_answer']

# Topic별로 집계할 딕셔너리
topic_stats = {}

for topic, output, gt in zip(topics, model_output, ground_truth):
    match = re.search(r"Answer:\s*([A-Z])\)", str(output))
    if match:
        pred = match.group(1)
        correct = (pred.lower() == str(gt).lower())
    else:
        correct = False  # 답변 형식이 맞지 않으면 오답 처리

    if topic not in topic_stats:
        topic_stats[topic] = {"correct": 0, "total": 0}
    topic_stats[topic]["total"] += 1
    if correct:
        topic_stats[topic]["correct"] += 1

# Topic별 정확도 출력
for topic, stats in topic_stats.items():
    accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"{topic}: {accuracy:.2%} ({stats['correct']}/{stats['total']})") 