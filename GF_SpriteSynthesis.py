import cv2
import os

input_folder = "./立绘"

for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith('_Alpha.png'):
            rgb_filename = filename.replace('_Alpha.png', '.png')
            rgb_path = os.path.join(root, rgb_filename)
            alpha_path = os.path.join(root, filename)

            if not os.path.exists(rgb_path):
                print(f"未在 {os.path.dirname(alpha_path)} 中找到对应的_alpha文件")
                continue

            src = cv2.imread(rgb_path)
            alpha = cv2.imread(alpha_path, -1)

            h1, w1, _ = src.shape
            alpha = cv2.resize(alpha, (w1, h1), cv2.INTER_CUBIC)

            alphach = cv2.split(alpha)

            dst = cv2.merge((src, alphach[3]))

            output_path = os.path.join(root, filename)


            cv2.imwrite(output_path, dst)
            print(f"合并: {output_path}")

            os.remove(os.path.join(root, rgb_filename))

print("合并完成")