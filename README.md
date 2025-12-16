# 人工智慧期末報告
# AI Projects｜利用神經網路輕鬆建立 Fashion MNIST 圖像辨識深度學習模型

<h2>
  組員:  11225035 謝詠任、11225006吳承祐、11225028呂偉勝
</h2>

這個專案將引導初學者透過使用 TensorFlow 框架，對 
Fashion MNIST 數據集進行圖像識別的深度學習模型的設計、
開發和評估，並且可以試著將模型下載分享與視覺化模型架構。
從基礎理論到實務操作，本專案旨在為初學者逐步了解深度學習的過程，
並且帶領讀者從數據預處理到模型訓練和評估，
進而掌握如何在實際情境中應用深度學習模型。


# 1. 學習目標

1.瞭解深度學習(Deep Learning)和神經網路(Neural Networks)的基本概念。

2.掌握使用 TensorFlow 進行進行數據處理、模型建構、訓練和評估的技巧。

3.學會進行數據正規化和模型參數的調整。

4.實作建構、訓練和評估深度學習模型。

5.提升問題解決和模型優化的能力。

</h2>

# 2. 步驟說明

## Step 1. 數據載入及預處理
載入數據集：使用 TensorFlow 的 Keras API 載入 Fashion MNIST 數據集，這是一個常提供給初學者使用的深度學習圖像分類數據集。是由 60,000 個圖像的訓練集(training set)和 10,000 個圖像的測試集(test set)所組成。每個圖像都是一個 28×28 灰階影像，與 10 個類別的標籤相關聯。


<img width="1916" height="412" alt="Image" src="https://github.com/user-attachments/assets/ba77b3dd-a90c-467c-beda-79c1bc9aa501" />

## Step 2. 顯示第一個訓練圖像

備註：本步驟提供想學習查看訓練資料圖像內容的讀者參考，若不需要則可以考慮跳至下一個步驟，不會影響本專案學習。

導入 matplotlib 函式庫來顯示圖像


import matplotlib.pyplot as plt

顯示第一個訓練圖像

plt.imshow(training_images[0])

<img width="1495" height="903" alt="Image" src="https://github.com/user-attachments/assets/3971baf3-1f54-4257-b5ff-5a565fbaa220" />

列印訓練標籤及圖像


print(training_labels[0]) # 列印第一個訓練標籤
9

輸出類別 9 即為 Ankle boot，其他類別可參考下表。

<img width="295" height="623" alt="Image" src="https://github.com/user-attachments/assets/6afafa66-9ca1-4a75-87fd-ce9cda0714a8" />


print(training_images[0]) # 列印第一個訓練圖像的像素數據

<img width="1236" height="885" alt="Image" src="https://github.com/user-attachments/assets/5c38e131-e316-4606-85de-dd2a62604962" />

## Step 3. 數據正規化
數據正規化：將圖像數據的像素值縮放到 0 到 1 之間，可以幫助模型更快更好地學習。

<img width="495" height="55" alt="Image" src="https://github.com/user-attachments/assets/dc154ccd-96a1-403d-aa54-69c5b8848b14" />


## Step 4. 定義模型結構
設計神經網路結構：使用 Sequential 模型來堆疊層(Layer)。首先是平坦層(Flatten Layer)將二維圖像轉換為一維陣列圖像數據，然後是兩個密集層(Dense Layer)進行特徵學習和分類。

<img width="1597" height="134" alt="Image" src="https://github.com/user-attachments/assets/833d5aec-3c8b-4033-a891-46875d89ba19" />

## Step 5. 模型編譯與訓練
編譯模型：在訓練之前，需要編譯模型，設置優化器、損失函數和評估指標。本篇文章將使用 Adam 優化器， sparse_categorical_crossentropy 作為損失函數，並追踪其準確率來做為評估指標。

<img width="491" height="67" alt="Image" src="https://github.com/user-attachments/assets/4e3d8f12-5109-4703-830f-ca5b3b9e29bc" />

## 訓練模型：使用 fit 方法來訓練模型，並且使用前面準備的圖像訓練數據來訓練模型，過程中模型將學習如何將輸入的訓練圖像映射到輸出類別，這就是監督式學習的基本概念。這裡設定迭代 5 個訓練週期，讀者可以根據自己需求調整訓練週期並觀察模型訓練效果。****

<img width="642" height="196" alt="Image" src="https://github.com/user-attachments/assets/15ad22ac-95ac-42d2-90d7-4b453e6d55dd" />

## Step 6. 模型儲存與載入
## 備註：本步驟提供有需要練習儲存模型的讀者參考，若不需要則可以考慮跳至下一個步驟，不會影響本專案學習。

### 儲存模型：訓練完成後，我們可以將模型儲存起來，以便未來使用或進行進一步的分析。

<img width="1626" height="58" alt="Image" src="https://github.com/user-attachments/assets/5647e196-e080-47c0-a363-4dceaa288b0f" />


### 載入模型：若要使用這個儲存的模型時，可以使用下列程式碼載入之前儲存的模型來使用。

<img width="1056" height="50" alt="Image" src="https://github.com/user-attachments/assets/2859fd4f-8a63-45a2-99a8-9e957c8e2945" />


## Step 7. 模型評估與優化
## 評估模型性能：使用測試數據集來評估模型的準確度，以了解模型在處理未見過的數據時的表現。下面程式碼將輸出模型在測試集上的準確度，讓我們能夠評估其泛化能力(Generalization)。

<img width="768" height="122" alt="Image" src="https://github.com/user-attachments/assets/f1b629be-1020-4e8f-9ae5-17487d5eae14" />

## 測試損失表示模型在測試數據上的平均損失值，測試準確率則表示模型正確預測標籤的比例。

## 性能優化：可以根據模型的表現，來調整模型結構（如增加層數、改變神經元數量）或調整學習參數（如學習率、批次大小），進行再次訓練和評估，以達到更好的性能。

</h2>

## Step 8. 檢視模型架構
## 備註：本步驟提供想了解神經網路結構的讀者參考，若不需要則可以暫時忽略此步驟，不會影響本專案學習。

### 檢視模型架構：若想要顯示模型的摘要資訊，可執行下面程式碼，將會顯示包括每層的名稱、輸出形狀和參數數量，對於理解模型的構造和複雜度非常有幫助。

<img width="823" height="236" alt="Image" src="https://github.com/user-attachments/assets/91f9f9bc-61b1-474d-ab4e-5d2d52740676" />






