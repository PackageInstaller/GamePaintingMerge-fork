import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

input_folder = "./立绘" # 输入文件夹路径
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
        alpha = cv2.imread(alpha_path, -1)

        if src is None:
            print(f"无法读取源图像: {file}")
            return
        if alpha is None:
            print(f"无法读取蒙版图像: {mask_file}")
            return

        h1, w1, _ = src.shape
        alpha = cv2.resize(alpha, (w1, h1), cv2.INTER_CUBIC)

        alphach = cv2.split(alpha)

        dst = cv2.merge((src, alphach[3]))

        output_path = os.path.join(root, mask_file)
        cv2.imwrite(output_path, dst)
        print(f"合并: {output_path}")

        os.remove(rgb_path)

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
