#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据重复检查脚本
检查验证集中是否包含训练集中的数据
"""

import json
import hashlib
from typing import Set, Dict, List, Tuple

def load_jsonl_data(file_path: str) -> List[Dict[str, str]]:
    """
    加载JSON Lines格式的数据
    
    Args:
        file_path: JSON Lines文件路径
        
    Returns:
        包含所有数据条目的列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"警告: 第{line_num}行JSON解析错误: {e}")
                        continue
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return []
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时发生异常: {e}")
        return []
    
    return data

def create_content_hash(content: str) -> str:
    """
    为内容创建哈希值，用于快速比较
    
    Args:
        content: 文本内容
        
    Returns:
        内容的MD5哈希值
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def check_data_overlap(train_file: str, val_file: str) -> Tuple[bool, List[Dict]]:
    """
    检查验证集和训练集之间的数据重复
    
    Args:
        train_file: 训练集文件路径
        val_file: 验证集文件路径
        
    Returns:
        (是否有重复, 重复数据列表)
    """
    print("正在加载训练集数据...")
    train_data = load_jsonl_data(train_file)
    print(f"训练集加载完成，共 {len(train_data)} 条数据")
    
    print("正在加载验证集数据...")
    val_data = load_jsonl_data(val_file)
    print(f"验证集加载完成，共 {len(val_data)} 条数据")
    
    if not train_data or not val_data:
        print("数据加载失败，无法进行重复检查")
        return False, []
    
    # 创建训练集内容的哈希集合，用于快速查找
    print("正在创建训练集内容索引...")
    train_content_hashes = set()
    train_content_map = {}  # 哈希值到原始数据的映射
    
    for item in train_data:
        if '内容' in item:
            content = item['内容'].strip()
            content_hash = create_content_hash(content)
            train_content_hashes.add(content_hash)
            train_content_map[content_hash] = item
    
    print(f"训练集索引创建完成，共 {len(train_content_hashes)} 个唯一内容")
    
    # 检查验证集中的重复数据
    print("正在检查验证集中的重复数据...")
    overlapping_data = []
    
    for i, val_item in enumerate(val_data):
        if '内容' in val_item:
            val_content = val_item['内容'].strip()
            val_content_hash = create_content_hash(val_content)
            
            if val_content_hash in train_content_hashes:
                train_item = train_content_map[val_content_hash]
                overlap_info = {
                    'validation_index': i,
                    'validation_data': val_item,
                    'training_data': train_item,
                    'content_preview': val_content[:100] + '...' if len(val_content) > 100 else val_content
                }
                overlapping_data.append(overlap_info)
    
    has_overlap = len(overlapping_data) > 0
    return has_overlap, overlapping_data

def main():
    """
    主函数
    """
    train_file = "train.json"
    val_file = "val.json"
    
    print("=" * 60)
    print("数据重复检查工具")
    print("=" * 60)
    
    has_overlap, overlapping_data = check_data_overlap(train_file, val_file)
    
    print("\n" + "=" * 60)
    print("检查结果")
    print("=" * 60)
    
    if has_overlap:
        print(f"❌ 发现重复数据！验证集中有 {len(overlapping_data)} 条数据与训练集重复")
        print("\n重复数据详情:")
        print("-" * 60)
        
        for i, overlap in enumerate(overlapping_data, 1):
            print(f"\n重复数据 #{i}:")
            print(f"验证集索引: {overlap['validation_index']}")
            print(f"验证集类别: {overlap['validation_data'].get('类别', 'N/A')}")
            print(f"训练集类别: {overlap['training_data'].get('类别', 'N/A')}")
            print(f"内容预览: {overlap['content_preview']}")
            
            if i >= 10:  # 只显示前10个重复项
                print(f"\n... 还有 {len(overlapping_data) - 10} 个重复项未显示")
                break
        
        print("\n建议: 请从验证集中移除这些重复数据以确保模型评估的准确性。")
        
    else:
        print("✅ 未发现重复数据！验证集和训练集之间没有重复的内容。")
    
    print("\n检查完成。")

if __name__ == "__main__":
    main()