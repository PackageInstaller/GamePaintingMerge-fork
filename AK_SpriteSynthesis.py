import cv2
import os

input_folder = "./立绘"

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(".png") and not file.endswith("_alpha.png"):
            mask_file = file.replace(".png", "_alpha.png")

            if mask_file not in files:
                print(f"对于文件 '{file}' 未找到匹配的_alpha文件 '{mask_file}'")
                continue

            src = cv2.imread(os.path.join(root, file))
            mask = cv2.imread(os.path.join(root, mask_file), 0)  # 灰度

            dst = cv2.merge((src, mask))

            output_path = os.path.join(root, file)  # 覆盖原始文件
            cv2.imwrite(output_path, dst)
            print(f"合并: {output_path}")

            os.remove(os.path.join(root, mask_file))  # 删除_alpha文件

print("合并完成")

