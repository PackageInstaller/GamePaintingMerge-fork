import cv2
import os
import numpy as np

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

            # 检查图像尺寸是否一致
            if src.shape[:2] != mask.shape:
                # 找到目标尺寸（较大的尺寸）
                target_size = (
                    max(src.shape[0], mask.shape[0]),
                    max(src.shape[1], mask.shape[1])
                )
                
                # 调整 src 和 mask 到目标尺寸
                src = cv2.resize(src, target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
                print(f"已调整尺寸: {file} 和 {mask_file} 到 {target_size}")

            # 将灰度图像转换为与 src 相同的三维形状
            mask = np.expand_dims(mask, axis=2)

            # 合并图像
            dst = cv2.merge((src, mask))

            output_path = os.path.join(root, file)  # 覆盖原始文件
            cv2.imwrite(output_path, dst)
            print(f"合并: {output_path}")

            os.remove(os.path.join(root, mask_file))  # 删除_alpha文件

print("合并完成")
