import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

input_folder = ""  # 输入文件夹路径
mask_suffixes = ["_alpha", "_mask", "_a"]  # 蒙版文件后缀，不区分大小写
num_threads = 16  # 指定线程数

def merge_images(file, root, lower_files, original_files):
    if file.endswith(".png") and not any(file.lower().endswith(suffix.lower() + ".png") for suffix in mask_suffixes):
        base_name = file[:-4]
        mask_file = None

        for suffix in mask_suffixes:
            potential_mask_file = base_name + suffix + ".png"
            if potential_mask_file.lower() in lower_files:
                mask_file = original_files[lower_files.index(potential_mask_file.lower())]
                break

        if not mask_file:
            print(f"对于文件 '{file}' 未找到匹配的蒙版文件")
            return

        rgb_path = os.path.join(root, file)
        alpha_path = os.path.join(root, mask_file)

        src = cv2.imread(rgb_path)
        alpha = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取包含 alpha 通道的图片

        if src is None:
            print(f"无法读取源图像: {file}")
            return
        if alpha is None:
            print(f"无法读取蒙版图像: {mask_file}")
            return

        # 确保蒙版和源图像尺寸一致
        h1, w1, _ = src.shape
        alpha_resized = cv2.resize(alpha, (w1, h1), interpolation=cv2.INTER_CUBIC)

        # 检查 alpha 通道
        if len(alpha_resized.shape) == 2:
            # 如果是单通道图像（灰度图像），直接用它作为 alpha 通道
            mask = alpha_resized
        elif len(alpha_resized.shape) == 3 and alpha_resized.shape[2] == 4:
            # 如果 alpha 图像有 4 个通道，则使用最后一个通道
            _, _, _, mask = cv2.split(alpha_resized)
        else:
            print(f"蒙版文件 '{mask_file}' 格式不正确，无法找到 alpha 通道")
            return

        # 将灰度图像转换为与 src 相同的三维形状
        mask = np.expand_dims(mask, axis=2)

        # 合并源图像和蒙版通道
        dst = cv2.merge((src, mask))

        # 使用源图像名称覆盖原文件
        output_path = os.path.join(root, file)  # 覆盖源图像
        cv2.imwrite(output_path, dst)
        print(f"合并: {output_path}")

        # 删除蒙版文件，源图像已更新，无需保留蒙版文件
        os.remove(alpha_path)

def main():
    tasks = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for root, dirs, files in os.walk(input_folder):
            lower_files = [f.lower() for f in files]
            original_files = files
            for file in files:
                task = executor.submit(merge_images, file, root, lower_files, original_files)
                tasks.append(task)
        
        for task in as_completed(tasks):
            task.result()
    
    print("合并完成")

if __name__ == "__main__":
    main()
