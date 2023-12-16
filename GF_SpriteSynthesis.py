import cv2
import os

input_folder = "./立绘"

for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith('.png') and not filename.endswith('_alpha.png'):
            mask_file = filename.replace('.png', '_alpha.png')
            mask_path = os.path.join(root, mask_file)


            if not os.path.exists(mask_path):
                print(f"对于文件 '{filename}' 未找到匹配的_alpha文件 '{mask_file}'")
                continue

            src = cv2.imread(rgb_path)
            alpha = cv2.imread(alpha_path, -1)

            h1, w1, _ = src.shape
            alpha = cv2.resize(alpha, (w1, h1), cv2.INTER_CUBIC)

            alphach = cv2.split(alpha)

            dst = cv2.merge((src, alphach[3]))

            output_path = os.path.join(root, f"{os.path.splitext(rgb_filename)[0]}_group.png")

            cv2.imwrite(output_path, dst)
            print(f"合并: {output_path}")

print("合并完成")

