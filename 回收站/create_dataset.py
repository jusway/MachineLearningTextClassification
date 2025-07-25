import os
import json
import random
import hashlib
from collections import defaultdict

def create_content_hash(content):
    """
    为内容创建哈希值，用于去重
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def create_train_val_dataset(dataset_path, train_output, val_output, train_samples=10000, val_samples=3000):
    """
    从THUCNews数据集中随机抽取训练集和验证集，并自动去除重复内容
    
    :param dataset_path: THUCNews数据集的根目录路径
    :param train_output: 训练集输出文件路径
    :param val_output: 验证集输出文件路径
    :param train_samples: 每个类别的训练样本数量
    :param val_samples: 每个类别的验证样本数量
    """
    
    # 定义12个类别
    categories = ['时尚', '房产', '游戏', '家居', '财经', '时政', '教育', '社会', '娱乐', '股票', '体育', '科技']
    
    print(f"开始处理数据集，源目录: {dataset_path}")
    
    # 收集每个类别的所有文件路径
    category_files = defaultdict(list)
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            print(f"正在扫描类别: {category}")
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    category_files[category].append(file_path)
            print(f"类别 {category} 共有 {len(category_files[category])} 个文件")
        else:
            print(f"警告: 找不到类别文件夹 {category}")
    
    # 检查每个类别是否有足够的文件
    total_needed = train_samples + val_samples
    for category in categories:
        if len(category_files[category]) < total_needed:
            print(f"警告: 类别 {category} 只有 {len(category_files[category])} 个文件，少于所需的 {total_needed} 个")
    
    # 先收集所有唯一内容，然后按需分配
    train_data = []
    val_data = []
    
    # 用于跟踪已见过的内容，避免重复
    seen_content_hashes = set()
    duplicate_count = 0
    
    for category in categories:
        print(f"正在处理类别 {category}")
        
        # 收集该类别的所有唯一内容
        unique_contents = []
        
        # 随机打乱文件列表以确保随机性
        random.shuffle(category_files[category])
        
        # 遍历所有文件，收集唯一内容
        for file_path in category_files[category]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 确保内容不为空
                        content_hash = create_content_hash(content)
                        if content_hash not in seen_content_hashes:
                            seen_content_hashes.add(content_hash)
                            unique_contents.append({
                                "内容": content,
                                "类别": category
                            })
                        else:
                            duplicate_count += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
        
        print(f"类别 {category} 收集到 {len(unique_contents)} 条唯一内容")
        
        # 检查是否有足够的唯一内容
        if len(unique_contents) < total_needed:
            print(f"警告: 类别 {category} 只有 {len(unique_contents)} 条唯一内容，少于所需的 {total_needed} 条")
            # 如果内容不足，全部用于训练集
            train_count = min(len(unique_contents), train_samples)
            val_count = max(0, len(unique_contents) - train_count)
        else:
            train_count = train_samples
            val_count = val_samples
        
        # 分配训练集和验证集
        train_data.extend(unique_contents[:train_count])
        val_data.extend(unique_contents[train_count:train_count + val_count])
        
        print(f"类别 {category}: 训练集 {train_count} 条，验证集 {val_count} 条")
    
    # 检查并确保严格的数量要求
    total_train_needed = train_samples * len(categories)  # 120000
    total_val_needed = val_samples * len(categories)      # 36000
    
    print(f"\n当前收集到: 训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")
    print(f"目标数量: 训练集 {total_train_needed} 条，验证集 {total_val_needed} 条")
    
    # 如果数量不足，需要调整策略
    if len(train_data) < total_train_needed or len(val_data) < total_val_needed:
        print("\n警告: 收集到的唯一内容不足以满足严格的数量要求")
        print("将调整分配策略以尽可能接近目标数量...")
        
        # 重新分配所有数据
        all_unique_data = train_data + val_data
        random.shuffle(all_unique_data)
        
        # 按比例重新分配
        train_ratio = total_train_needed / (total_train_needed + total_val_needed)
        actual_train_count = min(int(len(all_unique_data) * train_ratio), len(all_unique_data))
        actual_val_count = len(all_unique_data) - actual_train_count
        
        train_data = all_unique_data[:actual_train_count]
        val_data = all_unique_data[actual_train_count:]
        
        print(f"重新分配后: 训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")
    
    # 如果训练集数量超过需要，截取到精确数量
    if len(train_data) > total_train_needed:
        random.shuffle(train_data)
        train_data = train_data[:total_train_needed]
        print(f"训练集截取到 {total_train_needed} 条")
    
    # 如果验证集数量超过需要，截取到精确数量
    if len(val_data) > total_val_needed:
        random.shuffle(val_data)
        val_data = val_data[:total_val_needed]
        print(f"验证集截取到 {total_val_needed} 条")
    
    # 最终打乱顺序
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # 保存训练集
    print(f"保存训练集到 {train_output}，共 {len(train_data)} 条数据")
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证集
    print(f"保存验证集到 {val_output}，共 {len(val_data)} 条数据")
    with open(val_output, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计信息
    print("\n=== 数据集创建完成 ===")
    print(f"训练集: {len(train_data)} 条数据")
    print(f"验证集: {len(val_data)} 条数据")
    print(f"去重统计: 共发现并跳过了 {duplicate_count} 个重复内容")
    print(f"唯一内容总数: {len(seen_content_hashes)} 条")
    
    # 按类别统计
    train_category_count = defaultdict(int)
    val_category_count = defaultdict(int)
    
    for item in train_data:
        train_category_count[item['类别']] += 1
    
    for item in val_data:
        val_category_count[item['类别']] += 1
    
    print("\n训练集各类别数量:")
    for category in categories:
        print(f"  {category}: {train_category_count[category]}")
    
    print("\n验证集各类别数量:")
    for category in categories:
        print(f"  {category}: {val_category_count[category]}")

if __name__ == '__main__':
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 数据集路径
    DATASET_PATH = r"C:\D_disk\THUCNews"
    
    # 输出文件路径
    TRAIN_OUTPUT = "train.json"
    VAL_OUTPUT = "val.json"
    
    # 创建数据集
    create_train_val_dataset(
        dataset_path=DATASET_PATH,
        train_output=TRAIN_OUTPUT,
        val_output=VAL_OUTPUT,
        train_samples=10000,
        val_samples=3000
    )