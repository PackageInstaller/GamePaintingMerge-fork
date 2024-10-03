import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

input_folder = "./立绘" # 输入文件夹路径
mask_suffixes = ["_alpha", "_mask", "_alpha", "_a"]  # 蒙版文件后缀,不区分大小写
num_threads = 4  # 指定线程数

def merge_images(file, root, lower_files, original_files):
    if file.lower().endswith(".png") and not any(file.lower().endswith(suffix.lower() + ".png") for suffix in mask_suffixes):
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

        src = cv2.imread(os.path.join(root, file))
        mask = cv2.imread(os.path.join(root, mask_file), 0)

        if src is None:
            print(f"无法读取源图像: {file}")
            return
        if mask is None:
            print(f"无法读取蒙版图像: {mask_file}")
            return

        if src.shape[:2] != mask.shape:
            target_size = (
                max(src.shape[0], mask.shape[0]),
                max(src.shape[1], mask.shape[1])
            )

            src = cv2.resize(src, target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
            print(f"已调整尺寸: {file} 和 {mask_file} 到 {target_size}")

        # 将灰度图像转换为与 src 相同的三维形状
        mask = np.expand_dims(mask, axis=2)

        dst = cv2.merge((src, mask))

        output_path = os.path.join(root, file)
        cv2.imwrite(output_path, dst)
        print(f"合并: {output_path}")

        os.remove(os.path.join(root, mask_file))

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
