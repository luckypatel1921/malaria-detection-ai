can you rewrite code with this

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

print("\n🧬 Loading Malaria Detection Model...")
model = load_model("malaria_model.h5")
print("✅ Model loaded successfully!\n")

folder_path = input("📂 Enter dataset folder path: ").strip('"')

if not os.path.exists(folder_path):
    print("❌ Folder not found!")
    exit()

infected_count = 0
uninfected_count = 0
total_images = 0

print("\n🔍 Starting Image Analysis...\n")

for root, dirs, files in os.walk(folder_path):

    for file in files:

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            file_path = os.path.join(root, file)

            img = cv2.imread(file_path)

            if img is None:
                continue

            img = cv2.resize(img,(64,64))
            img = img / 255.0
            img = np.reshape(img,(1,64,64,3))

            prediction = model.predict(img, verbose=0)[0][0]

            total_images += 1

            if prediction > 0.5:
                infected_count += 1
                print(f"{file} → ⚠ INFECTED ({prediction:.2f})")
            else:
                uninfected_count += 1
                print(f"{file} → ✅ UNINFECTED ({prediction:.2f})")


print("\n==============================")
print("       🧾 ANALYSIS REPORT")
print("==============================")

print("Total Images Processed :", total_images)
print("Infected Cells         :", infected_count)
print("Uninfected Cells       :", uninfected_count)

if total_images > 0:
    infection_rate = (infected_count / total_images) * 100
    print("Infection Rate         :", round(infection_rate,2), "%")

print("==============================")
print("✔ Analysis Complete")