#读取/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulSingleVersion/conventionTools/labelInfo.txt，从其中分组分别读取Broad Version为0.5，0.6，0.7，0.8的合约地址Address，和每个地址对应的ground-truth label
# 按照版本号，分组去用地址到/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulSingleVersion/conventionTools/reentrancy_information.csv中匹配mythril的标签，这里的标签是predicted_label，如果没有匹配到，如果predicted_label和ground-truth label一致，则认为是正确的，否则认为是错误的。然后以此计算准确率和召回率，还有F1-score。如果predicted_label可能是N/A，即将所有N/A预测视为假正例（如果它们对应的真实标签不是正类）或假负例（如果它们对应的真实标签是正类）。再计算另一组准确率和召回率，F1-score，只统计不为N/A的预测结果。

# 此外，还有predicted_label可能是N/A，表示mythril 是failed to analyze，统计下失败和成功的数量。用失败的梳理比上总的个数表示，不用计算出比例。



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

results_ignore_na = {}
results_include_na = {}

for version in ['0.5', '0.6', '0.7', '0.8']:
    predictions, labels = [], []
    for row in label_info:
        address, broad_version, _, ground_truth = row
        if broad_version == version:
            reentrancy_row = reentrancy_dict.get(address)
            if reentrancy_row:
                predicted_label = reentrancy_row[6]  # 第四列是securify1标签
                predictions.append(predicted_label)
                labels.append(ground_truth)

    precision_ignore, recall_ignore, f1_ignore = calculate_metrics_ignore_na(predictions, labels)
    precision_include, recall_include, f1_include = calculate_metrics_include_na(predictions, labels)

    results_ignore_na[version] = {
        'precision': precision_ignore,
        'recall': recall_ignore,
        'f1_score': f1_ignore
    }
    results_include_na[version] = {
        'precision': precision_include,
        'recall': recall_include,
        'f1_score': f1_include
    }

# 输出结果
for version in ['0.5', '0.6', '0.7', '0.8']:
    print(f"Version {version} (Ignoring N/A): Precision: {results_ignore_na[version]['precision']}, Recall: {results_ignore_na[version]['recall']}, F1-Score: {results_ignore_na[version]['f1_score']}")
    print(f"Version {version} (Including N/A): Precision: {results_include_na[version]['precision']}, Recall: {results_include_na[version]['recall']}, F1-Score: {results_include_na[version]['f1_score']}")
