import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preprocessing import utils

result = []

def main():
   results = pd.read_csv('results.csv') 

   result = load_checkpoint()
   
   for i in range(len(results)):
       if i < len(result):
           continue

       color, gray = get_images(results['label'][i])

       color, gray = cv2.imread(color), cv2.imread(gray)
       color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
       color = utils.template_matching(color, gray)

       fig = plt.figure(figsize=(10, 5))
       fig.canvas.mpl_connect('key_press_event', on_press)


       plt.subplot(1, 2, 1)
       plt.imshow(color)
       plt.xlabel(f"Image {i+1} / {len(results)}")

       plt.subplot(1, 2, 2)
       plt.imshow(gray)

       plt.show()

       with open("checkpoint.txt", "w") as f:
           f.write(str(result))

   print(len(result))
   results['result'] = result
   results.to_csv('results.csv', index=False)


def get_images(label):
    color = f"dataset/Color/{label}_4.png"
    gray = f"dataset/Gray/{label}_2.jpg"
    return (color, gray)


def load_checkpoint():
    try:
        with open("checkpoint.txt", "r") as f:
            result = f.read()

            li = []
            for ch in result:
                if ch == '1':
                    li.append(1)
                elif ch == '0':
                    li.append(0)
                else:
                    continue

            return li 
    except:
        return []


def on_press(event):
    if event.key == 'k':
        result.append(1)
        plt.close()
    elif event.key == 'l':
        result.append(0)
        plt.close()
    elif event.key == 'z':
        result.pop()
        result.append(0)

if __name__ == '__main__':
    main()
