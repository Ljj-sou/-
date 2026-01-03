import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import time

print("=" * 70)
print("员工离职预测 - 纯Python实现（带详细查准率/查全率分析）")
print("=" * 70)


# ==================== 基础工具函数 ====================
def load_csv(filename: str) -> Tuple[List[str], List[List[str]]]:
    """读取CSV文件"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append(row)
    return header, data


def calculate_mean(values: List[float]) -> float:
    """计算平均值"""
    return sum(values) / len(values) if values else 0


def calculate_std(values: List[float]) -> float:
    """计算标准差"""
    if len(values) < 2:
        return 0
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def normalize_features(features: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """特征标准化（Z-score标准化）并返回标准化参数"""
    if not features:
        return features, [], []

    n_features = len(features[0])
    normalized = []

    # 计算每个特征的均值和标准差
    means = []
    stds = []

    for i in range(n_features):
        column = [row[i] for row in features]
        means.append(calculate_mean(column))
        stds.append(calculate_std(column))

    # 标准化每个特征
    for row in features:
        normalized_row = []
        for i, value in enumerate(row):
            if stds[i] != 0:
                normalized_row.append((value - means[i]) / stds[i])
            else:
                normalized_row.append(0)
        normalized.append(normalized_row)

    return normalized, means, stds


def normalize_features_with_params(features: List[List[float]], means: List[float], stds: List[float]) -> List[
    List[float]]:
    """使用已有参数标准化特征"""
    normalized = []
    for row in features:
        normalized_row = []
        for i, value in enumerate(row):
            if stds[i] != 0:
                normalized_row.append((value - means[i]) / stds[i])
            else:
                normalized_row.append(0)
        normalized.append(normalized_row)
    return normalized


def split_train_test(features: List[List[float]], labels: List[int],
                     test_size: float = 0.2, random_seed: int = 42) -> Tuple:
    """划分训练集和测试集"""
    n_samples = len(features)
    n_test = int(n_samples * test_size)

    random.seed(random_seed)
    indices = list(range(n_samples))
    random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = [features[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


# ==================== 数据预处理 ====================
class LabelEncoder:
    """简单的标签编码器"""

    def __init__(self):
        self.classes_ = {}
        self.reverse_map = {}

    def fit(self, values: List[str]):
        unique_values = sorted(set(values))
        self.classes_ = {val: idx for idx, val in enumerate(unique_values)}
        self.reverse_map = {idx: val for val, idx in self.classes_.items()}

    def transform(self, values: List[str]) -> List[int]:
        return [self.classes_[val] for val in values]

    def fit_transform(self, values: List[str]) -> List[int]:
        self.fit(values)
        return self.transform(values)


def preprocess_data(header: List[str], data: List[List[str]]) -> Tuple:
    """数据预处理"""
    col_idx = {col: i for i, col in enumerate(header)}

    encoders = {
        'Education': LabelEncoder(),
        'Gender': LabelEncoder(),
        'EverBenched': LabelEncoder()
    }

    for feature, encoder in encoders.items():
        idx = col_idx[feature]
        values = [row[idx] for row in data]
        encoder.fit(values)

    processed_features = []
    labels = []

    for row in data:
        features = []

        joining_year = float(row[col_idx['JoiningYear']])
        payment_tier = float(row[col_idx['PaymentTier']])
        age = float(row[col_idx['Age']])
        experience = float(row[col_idx['ExperienceInCurrentDomain']])

        features.extend([joining_year, payment_tier, age, experience])

        for feature, encoder in encoders.items():
            idx = col_idx[feature]
            encoded = encoder.transform([row[idx]])[0]
            features.append(float(encoded))

        # 特征工程
        years_of_service = 2024 - joining_year
        features.append(years_of_service)

        if age < 25:
            age_group = 1
        elif age < 30:
            age_group = 2
        elif age < 35:
            age_group = 3
        elif age < 40:
            age_group = 4
        else:
            age_group = 5
        features.append(age_group)

        if experience <= 1:
            exp_level = 0
        elif experience <= 3:
            exp_level = 1
        elif experience <= 5:
            exp_level = 2
        else:
            exp_level = 3
        features.append(exp_level)

        processed_features.append(features)
        labels.append(int(row[col_idx['LeaveOrNot']]))

    return processed_features, labels, encoders


# ==================== 机器学习算法 ====================
class SimpleDecisionTree:
    """简单的决策树分类器"""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: List[List[float]], y: List[int]):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: List[List[float]], y: List[int], depth: int) -> Dict:
        n_samples = len(X)
        n_classes = len(set(y))

        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            leaf_value = self._most_common_label(y)
            return {'type': 'leaf', 'value': leaf_value}

        best_split = self._find_best_split(X, y)
        if not best_split:
            leaf_value = self._most_common_label(y)
            return {'type': 'leaf', 'value': leaf_value}

        feature_idx, threshold = best_split

        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(n_samples):
            if X[i][feature_idx] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        return {
            'type': 'node',
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X: List[List[float]], y: List[int]) -> Tuple:
        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        if n_samples <= 1:
            return None

        parent_gini = self._gini_impurity(y)
        best_gini_gain = 0
        best_split = None

        for feature_idx in range(n_features):
            values = [row[feature_idx] for row in X]
            unique_values = sorted(set(values))

            for threshold in unique_values:
                left_y, right_y = [], []
                for i in range(n_samples):
                    if X[i][feature_idx] <= threshold:
                        left_y.append(y[i])
                    else:
                        right_y.append(y[i])

                if not left_y or not right_y:
                    continue

                gini_gain = self._gini_gain(y, left_y, right_y)

                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    best_split = (feature_idx, threshold)

        return best_split

    def _gini_impurity(self, y: List[int]) -> float:
        if not y:
            return 0

        n_samples = len(y)
        counts = Counter(y)
        impurity = 1

        for count in counts.values():
            prob = count / n_samples
            impurity -= prob ** 2

        return impurity

    def _gini_gain(self, parent_y: List[int], left_y: List[int], right_y: List[int]) -> float:
        parent_gini = self._gini_impurity(parent_y)

        n_total = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)

        child_gini = (n_left / n_total) * self._gini_impurity(left_y) + \
                     (n_right / n_total) * self._gini_impurity(right_y)

        return parent_gini - child_gini

    def _most_common_label(self, y: List[int]) -> int:
        if not y:
            return 0
        return Counter(y).most_common(1)[0][0]

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._predict_one(x, self.tree) for x in X]

    def _predict_one(self, x: List[float], node: Dict) -> int:
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """预测概率（简单实现，基于叶节点中各类的比例）"""
        probabilities = []
        for x in X:
            prob = self._predict_proba_one(x, self.tree)
            probabilities.append(prob)
        return probabilities

    def _predict_proba_one(self, x: List[float], node: Dict) -> List[float]:
        if node['type'] == 'leaf':
            # 简单返回 [不离职概率, 离职概率]
            if node['value'] == 1:
                return [0.0, 1.0]
            else:
                return [1.0, 0.0]

        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_proba_one(x, node['left'])
        else:
            return self._predict_proba_one(x, node['right'])


class SimpleLogisticRegression:
    """简单的逻辑回归分类器"""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = []
        self.bias = 0

    def _sigmoid(self, z: float) -> float:
        if z > 100:
            return 1.0
        elif z < -100:
            return 0.0
        return 1 / (1 + math.exp(-z))

    def fit(self, X: List[List[float]], y: List[int]):
        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for iteration in range(self.n_iterations):
            predictions = []
            for i in range(n_samples):
                z = self.bias
                for j in range(n_features):
                    z += self.weights[j] * X[i][j]
                predictions.append(self._sigmoid(z))

            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                error = predictions[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += error * X[i][j]

            self.bias -= self.learning_rate * db / n_samples
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples

    def predict(self, X: List[List[float]]) -> List[int]:
        predictions = []
        for x in X:
            z = self.bias
            for j in range(len(self.weights)):
                z += self.weights[j] * x[j]
            prob = self._sigmoid(z)
            predictions.append(1 if prob > 0.5 else 0)
        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        probabilities = []
        for x in X:
            z = self.bias
            for j in range(len(self.weights)):
                z += self.weights[j] * x[j]
            prob = self._sigmoid(z)
            probabilities.append(prob)
        return probabilities


# ==================== 详细评估函数 ====================
def calculate_precision_recall_metrics(y_true: List[int], y_pred: List[int],
                                       class_label: int = 1) -> Dict[str, float]:
    """详细计算查准率、查全率和相关指标"""
    tp = fp = tn = fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == class_label and pred == class_label:
            tp += 1
        elif true != class_label and pred == class_label:
            fp += 1
        elif true != class_label and pred != class_label:
            tn += 1
        elif true == class_label and pred != class_label:
            fn += 1

    n_samples = len(y_true)
    accuracy = (tp + tn) / n_samples if n_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 额外指标
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假正率
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0  # 假负率

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'support': tp + fn  # 该类别的真实样本数
    }


def evaluate_with_threshold(y_true: List[int], y_proba: List[float],
                            threshold: float = 0.5) -> Dict[str, float]:
    """使用不同阈值评估模型"""
    y_pred = [1 if prob >= threshold else 0 for prob in y_proba]
    return calculate_precision_recall_metrics(y_true, y_pred)


def find_optimal_threshold(y_true: List[int], y_proba: List[float],
                           metric: str = 'f1_score') -> Tuple[float, Dict[str, float]]:
    """寻找最佳阈值"""
    best_metric = 0
    best_threshold = 0.5
    best_metrics = {}

    thresholds = [i / 100 for i in range(0, 101)]  # 0.00 到 1.00

    for threshold in thresholds:
        metrics = evaluate_with_threshold(y_true, y_proba, threshold)

        if metrics[metric] > best_metric:
            best_metric = metrics[metric]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def calculate_pr_curve(y_true: List[int], y_proba: List[float],
                       num_points: int = 20) -> List[Dict[str, float]]:
    """计算PR曲线数据点"""
    pr_points = []

    for i in range(num_points + 1):
        threshold = i / num_points
        metrics = evaluate_with_threshold(y_true, y_proba, threshold)
        pr_points.append({
            'threshold': threshold,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })

    return pr_points


def print_confusion_matrix_analysis(metrics: Dict[str, float], model_name: str = ""):
    """打印混淆矩阵和详细分析"""
    tp = metrics['tp']
    fp = metrics['fp']
    tn = metrics['tn']
    fn = metrics['fn']

    print(f"\n{'=' * 60}")
    print(f"混淆矩阵详细分析 - {model_name}")
    print(f"{'=' * 60}")

    print(f"\n混淆矩阵:")
    print(f"          预测为正    预测为负    总计")
    print(f"实际为正    {tp:6d}        {fn:6d}    {tp + fn:6d}")
    print(f"实际为负    {fp:6d}        {tn:6d}    {fp + tn:6d}")
    print(f"总计        {tp + fp:6d}      {tn + fn:6d}    {tp + fp + tn + fn:6d}")

    print(f"\n核心指标:")
    print(f"准确率 (Accuracy):    {metrics['accuracy']:.4f}")
    print(f"查准率 (Precision):   {metrics['precision']:.4f}  - 预测为正的样本中，实际为正的比例")
    print(f"查全率 (Recall):      {metrics['recall']:.4f}    - 实际为正的样本中，被正确预测的比例")
    print(f"F1分数:               {metrics['f1_score']:.4f}  - 查准率和查全率的调和平均")
    print(f"特异度 (Specificity): {metrics['specificity']:.4f} - 实际为负的样本中，被正确预测的比例")

    print(f"\n错误指标:")
    print(f"假正率 (FPR):         {metrics['false_positive_rate']:.4f}")
    print(f"假负率 (FNR):         {metrics['false_negative_rate']:.4f}")

    print(f"\n业务解释:")
    print(f"1. 查准率 {metrics['precision']:.1%} 表示：")
    print(f"   当模型预测员工会离职时，有 {metrics['precision']:.1%} 的概率预测正确")

    print(f"\n2. 查全率 {metrics['recall']:.1%} 表示：")
    print(f"   在所有实际会离职的员工中，模型能识别出 {metrics['recall']:.1%}")

    print(f"\n3. 假正率 {metrics['false_positive_rate']:.1%} 表示：")
    print(f"   模型错误地将 {metrics['false_positive_rate']:.1%} 的不离职员工预测为离职")

    print(f"\n4. 假负率 {metrics['false_negative_rate']:.1%} 表示：")
    print(f"   模型漏掉了 {metrics['false_negative_rate']:.1%} 的实际会离职的员工")


def print_pr_curve_analysis(pr_points: List[Dict[str, float]], model_name: str = ""):
    """打印PR曲线分析"""
    print(f"\n{'=' * 60}")
    print(f"PR曲线分析 - {model_name}")
    print(f"{'=' * 60}")

    print(f"\n阈值对查准率和查全率的影响:")
    print(f"{'阈值':<8} {'查准率':<10} {'查全率':<10} {'F1分数':<10}")
    print(f"{'-' * 40}")

    for point in pr_points[::5]:  # 每5个点显示一次
        print(f"{point['threshold']:<8.2f} {point['precision']:<10.4f} "
              f"{point['recall']:<10.4f} {point['f1_score']:<10.4f}")

    # 找到最佳F1分数的点
    best_point = max(pr_points, key=lambda x: x['f1_score'])
    print(f"\n最佳F1分数点:")
    print(f"阈值: {best_point['threshold']:.2f}")
    print(f"查准率: {best_point['precision']:.4f}")
    print(f"查全率: {best_point['recall']:.4f}")
    print(f"F1分数: {best_point['f1_score']:.4f}")


def compare_precision_recall_tradeoff(results: Dict[str, Dict[str, Any]]):
    """比较不同模型的查准率-查全率权衡"""
    print(f"\n{'=' * 60}")
    print("不同模型的查准率-查全率比较")
    print(f"{'=' * 60}")

    print(f"\n{'模型名称':<15} {'查准率':<10} {'查全率':<10} {'F1分数':<10} {'最佳阈值':<10}")
    print(f"{'-' * 55}")

    for model_name, result in results.items():
        metrics = result['metrics']
        best_threshold = result.get('best_threshold', 0.5)

        print(f"{model_name:<15} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
              f"{best_threshold:<10.2f}")

    # 分析权衡关系
    print(f"\n查准率-查全率权衡分析:")
    print("1. 高查准率模型: 预测离职时很准确，但可能漏掉一些实际会离职的员工")
    print("2. 高查全率模型: 能识别出大部分离职员工，但可能有较多误报")
    print("3. 平衡模型 (高F1分数): 在查准率和查全率之间取得较好平衡")


# ==================== 主程序 ====================
def main():
    print("1. 加载数据...")
    header, data = load_csv('task3_dataset.csv')

    print(f"数据集大小: {len(data)} 条记录")

    print("\n2. 数据预处理...")
    features, labels, encoders = preprocess_data(header, data)

    label_dist = Counter(labels)
    print(f"标签分布: 离职={label_dist[1]}, 不离职={label_dist[0]}")
    print(f"离职比例: {label_dist[1] / len(labels):.2%}")

    print("\n3. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = split_train_test(features, labels, test_size=0.2)

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    print("\n4. 特征标准化...")
    X_train_normalized, train_means, train_stds = normalize_features(X_train)
    X_test_normalized = normalize_features_with_params(X_test, train_means, train_stds)

    print("\n5. 训练和评估模型...")

    models = {
        '决策树': SimpleDecisionTree(max_depth=5),
        '逻辑回归': SimpleLogisticRegression(learning_rate=0.1, n_iterations=500),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'*' * 60}")
        print(f"训练 {name}...")
        print(f"{'*' * 60}")

        start_time = time.time()
        model.fit(X_train_normalized, y_train)
        training_time = time.time() - start_time

        # 预测
        y_pred = model.predict(X_test_normalized)

        # 计算详细指标
        metrics = calculate_precision_recall_metrics(y_test, y_pred)

        # 保存结果
        results[name] = {
            'model': model,
            'metrics': metrics,
            'training_time': training_time,
            'y_pred': y_pred,
            'y_test': y_test
        }

        # 打印详细分析
        print_confusion_matrix_analysis(metrics, name)

        # 如果是概率模型，进行阈值分析
        if hasattr(model, 'predict_proba'):
            print(f"\n进行阈值分析...")
            y_proba = model.predict_proba(X_test_normalized)
            if isinstance(y_proba[0], list):  # 处理二维概率数组
                y_proba = [prob[1] for prob in y_proba]

            # 寻找最佳阈值
            best_threshold, best_metrics = find_optimal_threshold(y_test, y_proba, 'f1_score')
            results[name]['best_threshold'] = best_threshold
            results[name]['best_metrics'] = best_metrics

            print(f"\n阈值分析结果:")
            print(f"默认阈值 (0.5):")
            print(f"  查准率: {metrics['precision']:.4f}, 查全率: {metrics['recall']:.4f}")
            print(f"最佳阈值 ({best_threshold:.2f}):")
            print(f"  查准率: {best_metrics['precision']:.4f}, 查全率: {best_metrics['recall']:.4f}")
            print(f"  F1分数提升: {best_metrics['f1_score'] - metrics['f1_score']:.4f}")

            # 计算PR曲线
            pr_points = calculate_pr_curve(y_test, y_proba)
            print_pr_curve_analysis(pr_points, name)

            results[name]['pr_points'] = pr_points

        print(f"\n训练时间: {training_time:.2f}秒")

    # 比较不同模型
    print("\n" + "=" * 70)
    print("模型性能综合比较")
    print("=" * 70)

    compare_precision_recall_tradeoff(results)

    # 找到最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
    best_result = results[best_model_name]

    print(f"\n{'*' * 60}")
    print(f"最佳模型: {best_model_name}")
    print(f"F1分数: {best_result['metrics']['f1_score']:.4f}")
    print(f"查准率: {best_result['metrics']['precision']:.4f}")
    print(f"查全率: {best_result['metrics']['recall']:.4f}")
    print(f"{'*' * 60}")

    # 业务建议
    print("\n" + "=" * 70)
    print("基于查准率和查全率的业务建议")
    print("=" * 70)

    precision = best_result['metrics']['precision']
    recall = best_result['metrics']['recall']
    f1_score = best_result['metrics']['f1_score']

    if precision > 0.7 and recall > 0.7:
        print("✅ 模型性能优秀：查准率和查全率都较高")
        print("建议：可以直接用于员工离职风险预警系统")

    elif precision > 0.7 and recall < 0.5:
        print("⚠️ 高查准率、低查全率模型")
        print("特点：预测很准确，但会漏掉很多实际会离职的员工")
        print("适用场景：")
        print("  1. 干预成本高的情况（如高额留人奖金）")
        print("  2. 只关注高风险员工")

    elif precision < 0.5 and recall > 0.7:
        print("⚠️ 低查准率、高查全率模型")
        print("特点：能识别大部分离职员工，但有很多误报")
        print("适用场景：")
        print("  1. 误报成本低的情况")
        print("  2. 需要全面筛查的情况")

    elif f1_score > 0.6:
        print("✅ 模型性能良好：在查准率和查全率之间取得较好平衡")
        print("建议：可以用于日常员工管理决策支持")
    else:
        print("⚠️ 模型性能有待提升")
        print("建议：")
        print("  1. 收集更多特征数据")
        print("  2. 尝试更复杂的模型")
        print("  3. 优化特征工程")

    print(f"\n具体建议：")
    print(f"1. 模型查准率 {precision:.1%}：")
    print(f"   当模型预测员工会离职时，有 {precision:.1%} 的概率是正确的")
    print(f"   建议针对这些高风险员工制定个性化留人方案")

    print(f"\n2. 模型查全率 {recall:.1%}：")
    print(f"   模型能识别出 {recall:.1%} 的实际会离职员工")
    print(f"   还有 {1 - recall:.1%} 的离职员工未被识别，需加强人工观察")

    print(f"\n3. 误报分析：")
    fpr = best_result['metrics']['false_positive_rate']
    print(f"   有 {fpr:.1%} 的不离职员工被误判为会离职")
    print(f"   避免对这些员工进行不必要的干预")

    # 阈值调整建议
    if 'best_threshold' in best_result:
        default_precision = best_result['metrics']['precision']
        default_recall = best_result['metrics']['recall']
        best_precision = best_result['best_metrics']['precision']
        best_recall = best_result['best_metrics']['recall']

        if best_result['best_threshold'] > 0.5:
            print(f"\n4. 阈值调整建议：")
            print(f"   最佳阈值({best_result['best_threshold']:.2f}) > 默认阈值(0.5)")
            print(f"   提高阈值可以提升查准率({default_precision:.3f} → {best_precision:.3f})")
            print(f"   但会降低查全率({default_recall:.3f} → {best_recall:.3f})")
            print(f"   适用于：误报成本高的情况")
        elif best_result['best_threshold'] < 0.5:
            print(f"\n4. 阈值调整建议：")
            print(f"   最佳阈值({best_result['best_threshold']:.2f}) < 默认阈值(0.5)")
            print(f"   降低阈值可以提升查全率({default_recall:.3f} → {best_recall:.3f})")
            print(f"   但会降低查准率({default_precision:.3f} → {best_precision:.3f})")
            print(f"   适用于：漏报成本高的情况")

    # 预测函数
    print("\n" + "=" * 70)
    print("预测函数示例（带阈值调整）")
    print("=" * 70)

    print('''
def predict_with_threshold(employee_data, model, encoders, 
                          threshold=0.5, return_prob=False):
    """
    预测员工是否离职（可调整阈值）

    参数:
    employee_data: 员工信息字典
    model: 训练好的模型
    encoders: 编码器
    threshold: 决策阈值 (默认0.5)
    return_prob: 是否返回概率

    返回:
    预测结果字典
    """
    # 预处理特征
    features = []

    joining_year = float(employee_data['JoiningYear'])
    payment_tier = float(employee_data['PaymentTier'])
    age = float(employee_data['Age'])
    experience = float(employee_data['ExperienceInCurrentDomain'])

    features.extend([joining_year, payment_tier, age, experience])

    for feature, encoder in encoders.items():
        encoded = encoder.transform([employee_data[feature]])[0]
        features.append(float(encoded))

    years_of_service = 2024 - joining_year
    features.append(years_of_service)

    if age < 25:
        age_group = 1
    elif age < 30:
        age_group = 2
    elif age < 35:
        age_group = 3
    elif age < 40:
        age_group = 4
    else:
        age_group = 5
    features.append(age_group)

    if experience <= 1:
        exp_level = 0
    elif experience <= 3:
        exp_level = 1
    elif experience <= 5:
        exp_level = 2
    else:
        exp_level = 3
    features.append(exp_level)

    # 标准化
    normalized_features = normalize_features_with_params(
        [features], train_means, train_stds
    )[0]

    # 预测
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba([normalized_features])[0]
        if isinstance(probability, list):
            prob = probability[1]  # 离职概率
        else:
            prob = probability

        prediction = 1 if prob >= threshold else 0
    else:
        prediction = model.predict([normalized_features])[0]
        prob = None

    result = {
        'prediction': prediction,
        'label': '离职' if prediction == 1 else '不离职',
        'threshold': threshold
    }

    if prob is not None:
        result['probability'] = prob

    return result
    ''')


def quick_demo():
    """快速演示"""
    print("快速演示：创建示例数据并分析查准率/查全率")

    # 创建示例数据
    X_train = [
        [25, 3, 2, 1, 0, 0],
        [30, 1, 5, 0, 1, 1],
        [35, 2, 8, 1, 0, 0],
        [40, 3, 10, 0, 1, 1],
        [28, 1, 3, 0, 0, 0],
    ]

    y_train = [0, 1, 0, 1, 0]

    X_test = [
        [32, 2, 6, 1, 1, 0],
        [45, 3, 12, 0, 1, 1],
        [26, 1, 2, 0, 0, 0],
    ]

    y_test = [1, 1, 0]

    # 训练模型
    print("\n1. 训练逻辑回归模型...")
    model = SimpleLogisticRegression(learning_rate=0.1, n_iterations=100)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    print(f"预测结果: {y_pred}")
    print(f"真实标签: {y_test}")

    # 计算查准率和查全率
    print("\n2. 计算查准率和查全率...")
    metrics = calculate_precision_recall_metrics(y_test, y_pred)

    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"查准率: {metrics['precision']:.4f}")
    print(f"查全率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")

    # 阈值分析
    print("\n3. 阈值分析...")
    y_proba = model.predict_proba(X_test)
    thresholds = [0.3, 0.5, 0.7]

    for threshold in thresholds:
        y_pred_thresh = [1 if prob >= threshold else 0 for prob in y_proba]
        metrics_thresh = calculate_precision_recall_metrics(y_test, y_pred_thresh)
        print(f"\n阈值 {threshold:.1f}:")
        print(f"  查准率: {metrics_thresh['precision']:.4f}")
        print(f"  查全率: {metrics_thresh['recall']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("找不到数据文件，运行快速演示...")
        quick_demo()
    except Exception as e:
        print(f"发生错误: {e}")
        print("运行快速演示代替...")
        quick_demo()
