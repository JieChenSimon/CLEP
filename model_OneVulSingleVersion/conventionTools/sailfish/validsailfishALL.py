def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split(',') for line in lines[1:]]  # 跳过标题行
    return data

def calculate_metrics_ignore_na(predictions, labels):
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, label in zip(predictions, labels):
        if pred == "N/A":
            continue  # 直接忽略 N/A
        if pred == label:
            if pred == '1':
                tp += 1
            else:
                tn += 1
        else:
            if pred == '1':
                fp += 1
            else:
                fn += 1
        
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1_score

def calculate_metrics_include_na(predictions, labels):
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, label in zip(predictions, labels):
        if pred == "N/A":
            if label == '0':
                fp += 1  # N/A 视为假正例（如果真实标签不是正类），即label为0，pred为N/A，
            else:
                fn += 1  # N/A 视为假负例（如果真实标签是正类），即label为1，pred为N/A，
            continue
        if pred == label:
            if pred == '1':
                tp += 1
            else:
                tn += 1
        else:
            if pred == '1': #两个label，不同，预测是1，但实际是0，则认为是假正例
                fp += 1
            else: #两个label，不同，预测是0，但实际是1，则认为是假负例
                fn += 1
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1_score

# 路径可能需要根据您的文件系统进行调整
label_info = read_file("/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulSingleVersion/conventionTools/labelInfo_new.txt")
reentrancy_info = read_file("/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulSingleVersion/conventionTools/reentrancy_information.csv")

# 将reentrancy_info转换为字典以便快速查找
reentrancy_dict = {row[0]: row for row in reentrancy_info}

# 合并所有版本的预测和标签
all_predictions, all_labels = [], []
for row in label_info:
    address, broad_version, _, ground_truth = row
    reentrancy_row = reentrancy_dict.get(address)
    if reentrancy_row:
        predicted_label = reentrancy_row[7]  # 第四列是securify1标签
        all_predictions.append(predicted_label)
        all_labels.append(ground_truth)

# 计算所有版本的指标
precision_ignore, recall_ignore, f1_ignore = calculate_metrics_ignore_na(all_predictions, all_labels)
precision_include, recall_include, f1_include = calculate_metrics_include_na(all_predictions, all_labels)

# 输出结果
print(f"All Versions (Ignoring N/A): Precision: {precision_ignore}, Recall: {recall_ignore}, F1-Score: {f1_ignore}")
print(f"All Versions (Including N/A): Precision: {precision_include}, Recall: {recall_include}, F1-Score: {f1_include}")