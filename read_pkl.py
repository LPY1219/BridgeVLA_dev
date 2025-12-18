import pickle
import sys

def read_pkl_file(file_path):
    """读取并打印pkl文件内容"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"成功读取文件: {file_path}")
        print(f"数据类型: {type(data)}")
        print("\n文件内容:")
        print(data)

        return data

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except pickle.UnpicklingError:
        print(f"错误: 无法解析pkl文件 {file_path}")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python read_pkl.py <pkl文件路径>")
        print("示例: python read_pkl.py data.pkl")
    else:
        file_path = sys.argv[1]
        read_pkl_file(file_path)
