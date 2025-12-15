# TimeCurve F1 Implementation

本專案旨在透過 Time Curves 技術視覺化 F1 賽車手的駕駛風格異同，目前專注於 2024 日本大獎賽 (Max Verstappen, Sergio Perez, Yuki Tsunoda)。

## 📁 專案結構

```
timecurve_f1/
```

## 🛠️ 資料接口使用方式 (Data Collection)

我們使用 [OpenF1 API](https://openf1.org/) 獲取比賽數據。相關代碼位於 `utils/fetch_f1_data.py`。

### 執行方式
```bash
python utils/fetch_f1_data.py
```

### 實作細節
1. **Meeting & Session**: 自動搜尋 2024 日本站 (Japan) 的正賽 (Race) Session Key。
2. **Chunked Fetching**: 由於 `car_data` 和 `location` 資料量大，API 容易回傳 422 或 500 錯誤，因此程式採用 **「每分鐘為一區塊 (Time Chunks)」** 的方式下載。
3. **Data Merging**: 下載後會自動去除重複的時間點 (基於 `date` 欄位)，並存入 CSV。
4. **Retry Logic**: 針對 500 錯誤進行簡單的錯誤記錄與重試機制（目前主要依賴縮小時間窗口來規避）。

---

## 📈 Time Curves 實作流程

本專案將依照以下學術定義的標準流程來實作 Time Curves：

### 1. 資料抽象化 (Data Abstraction)
將比賽數據定義為時序資料集 $P = \{p_0, p_1, ..., p_n\}$。
*   **時間戳記 ($t_i$)**: 對應遙測數據的每個採樣時間點 (約 3-4Hz)。
*   **資料快照 ($s_i$)**: 設定為高維特徵向量，目前簡化為 `[throttle, brake, n_gear]` 以聚焦於駕駛操作。

### 2. 建構距離矩陣 (Distance Matrix Construction)
計算所有時間點兩兩之間的相似度，生成 $N \times N$ 對稱矩陣 $D$。
*   **Metric**: 針對數值型遙測數據，計畫使用 **歐幾里得距離 (Euclidean Distance)** 或 **餘弦相似度 (Cosine Similarity)**（需先正規化）。

### 3. 降維投影 (Dimensionality Reduction)
將高維特徵轉化為 2D 平面座標。
*   **演算法**: 使用 **Classical MDS (Multidimensional Scaling)**。
*   **理由**: 收斂速度快且定位比 Force-directed 方法更精確，適合保留全域的時間結構。

### 4. 幾何優化與調整 (Geometric Refinement)
提升視覺可讀性 (Legibility) 的後處理步驟：
*   **移除重疊 (Overlap Removal)**: 使用迭代演算法推開重疊的點，並為被移動的點加上灰色光暈 (Halo)。
*   **旋轉 (Rotation)**: 自動旋轉圖形，使起始點 $p_0$ 位於左側，整體時間流向往右發展，符合閱讀習慣。

### 5. 曲線繪製 (Curve Drawing)
將散點連接成流暢的曲線。
*   **方法**: 使用 **Catmull-Rom 插值** 或 **貝茲曲線 (Bézier curves)**。
*   **參數**: 平滑參數 $\sigma = 0.3$，並在切線上加入微小隨機擾動，避免來回震盪的路徑完全遮擋。


### 6. 視覺編碼 (Visual Encoding)
將時間與狀態資訊編碼為視覺屬性：
*   **顏色 (Color)**: 代表時間 $t$ 的進程（例如：淺色 $\rightarrow$ 深色），使用 Viridis 色票，紫色為起始，黃色為結束。
*   **粗細 (Thickness)**: 代表「持續時間 (Duration)」。兩點間隔時間越長（速度慢），線條越粗。
*   **光暈 (Halo)**: 
    *   🔵 藍色: 完全相同的重複狀態。
    *   ⚪ 灰色: 被演算法推開的相似群聚。

---

## 📊 圖表解讀與分析結果

### 1. 全局 Time Curve (`output/time_curve_*.png`)
這張圖展示了車手在整個比賽或完整單圈中的「狀態演變」。
- **迴圈 (Loops)**: 代表車輛經歷了一系列相似的狀態後回到了原點。在 F1 中，**每個迴圈通常代表一圈 (Lap)**。
- **軌跡平滑度**:
    - **平滑**: 代表駕駛節奏穩定，每一圈的煞車、加速點都非常一致（例如 Max Verstappen）。
    - **雜亂/毛刺**: 代表每一圈的處理方式都有微小差異，可能受輪胎衰退或車流影響。

### 2. S 彎道比較分析 (`output/comparison_scurve.png`)
我們將三位車手 (Max, Perez, Tsunoda) 在鈴鹿賽道 S 彎 (Sector 1) 的數據投影到同一個相似度空間中。
- **重疊**: 若兩條線緊密重疊，代表兩位車手在該路段的駕駛方式（速度、檔位、油門控制）幾乎完全雷同。
- **分歧 (Divergence)**: 若線條分開，代表駕駛風格出現差異。

#### 🏁 關鍵發現：Max vs Perez (S-Curves)
根據我們的演算法分析 (詳見 `output/divergence_report.txt`)，兩者在入彎處出現了顯著分歧：
*   **Max Verstappen (Index 38)**: 採取了 **激進減速 (V-Style)**。
    *   速度: **209 km/h** | 煞車: **100% (重踩)** | 油門: **0% (全放)**
    *   *解讀*: Max 傾向於晚煞車並重煞，利用重心轉移快速讓車頭對準出彎點。
*   **Sergio Perez (Index 38)**: 保持 **高速滑行 (U-Style)**。
    *   速度: **225 km/h** | 煞車: **0%** | 油門: **94% (幾近全油)**
    *   *解讀*: 在同一時刻，Perez 尚未開始重煞或選擇以更高底速過彎，這顯示了兩者在車輛調校或駕駛習慣上的根本差異。

### 3. 車手一致性分析 (三位車手)
我們為每位車手生成了個別的一致性分析,展示各自在整場比賽中通過 S 彎的操作重疊圖。

#### 🟣 Max Verstappen (`output/consistency_Max.html`)
- **白色實線參考**: 代表 **Lap 50 (最快圈, 93.706秒)**，作為最佳表現的基準
- **背景軌跡**: 顯示了整場比賽的所有 52 圈,較窄的通道代表極高的穩定性
- **最快圈**: Lap 50 - 93.706秒 (三位車手中最快)

#### 🔵 Sergio Perez (`output/consistency_Perez.html`)
- **白色實線參考**: 代表 **Lap 35 (最快圈, 93.945秒)**
- **背景軌跡**: 52 圈疊加
- **最快圈**: Lap 35 - 93.945秒 (比 Verstappen 慢 0.239秒)

#### 🟡 Yuki Tsunoda (`output/consistency_Tsunoda.html`)
- **白色實線參考**: 代表 **Lap 51 (最快圈, 96.342秒)**
- **背景軌跡**: 50 圈疊加
- **最快圈**: Lap 51 - 96.342秒 (比 Verstappen 慢 2.636秒)

**共同特性:**
- **演變 (Evolution)**: 通過觀察顏色從紫色（早期）漸變到黃色（晚期），可以發現隨著輪胎耗損，操作軌跡是否發生偏移
- **互動功能**:
    - ▶️ **播放按鈕**: 自動以 0.8 秒間隔循環播放所有圈次
    - 滑桿或下拉選單可查看特定圈次與最快圈的差異
    - 紅色連結線顯示當前圈與最快圈起點的偏移距離
- 離群值 (Outliers) 通常對應到被套圈車阻擋或失誤的單圈

### 4. 車手風格對比 (兩個版本)

#### 4a. S 彎版本 (`output/driver_styles_comparison_S.html`)
比較三位車手在 **S 彎區域** (Sector 1, T3-T7) 的平均駕駛風格。
- **分析範圍**: 僅 S 彎區段 (~15-45秒)
- **顏色區分**:
    - 🟣 **Verstappen**: 紫色 - 最快車手 (93.706秒)
    - 🔵 **Perez**: 藍色 - 第二快 (93.945秒，慢 0.239秒)
    - 🟡 **Tsunoda**: 黃色 - AlphaTauri 車手 (96.342秒，慢 2.636秒)
- **風格差異觀察**:
    - **Red Bull 隊友 (Max vs Perez)**: 相同賽車，差異純粹來自個人駕駛習慣
    - **跨車隊對比 (vs Tsunoda)**: 包含賽車性能差異，AlphaTauri 明顯較慢
- **互動功能**: 點擊按鈕可單獨顯示某位車手或顯示全部對比

#### 4b. 全局版本 (`output/driver_styles_comparison.html`)
比較三位車手在**整場比賽**的平均駕駛風格。
- **分析範圍**: 完整單圈 (~90-96秒)
- **顏色與互動**: 與 S 彎版本相同
- **差異**: 展示完整賽道的綜合駕駛特性，而非特定彎角

### 5. 互動式視覺化總覽
靜態圖表難以呈現時間差。本專案提供**六個**互動式網頁：

**單圈動態追逐:**
- **`index.html`**: 三位車手 (Max, Perez, Tsunoda) 的 S 彎動態追逐，可拖動時間軸精確重現關鍵瞬間

**車手一致性分析 (附播放功能):**
- **`consistency_Max.html`**: Max Verstappen 的 52 圈演變
- **`consistency_Perez.html`**: Sergio Perez 的 52 圈演變
- **`consistency_Tsunoda.html`**: Yuki Tsunoda 的 50 圈演變

**車手風格對比 (可切換顯示):**
- **`driver_styles_comparison.html`**: 全局平均風格 (完整賽道)
- **`driver_styles_comparison_S.html`**: S 彎平均風格 (Sector 1)

**共同特性:**
- 網頁內均包含詳細的 **「圖表解讀指南 (How to Read)」**
- **注意**: 資料點之間的時間間隔約在 **0.4秒 ~ 1.3秒** 之間浮動，這是由於資料來源的不規則性，目前保留此特性以忠實呈現原始資料

---

## 🚀 Scripts 使用說明

本專案提供 4 個分析腳本,請按照以下順序執行以完成完整分析流程:

### 1. `generate_time_curve.py` - 生成 Time Curve

**功能:**  
為每位車手生成完整比賽的 Time Curve 視覺化,展示整個比賽過程中駕駛狀態的演變。

**執行方式:**
```bash
python scripts/generate_time_curve.py
```

**主要參數 (腳本內設定):**
- `drivers = [1, 11, 22]` - 分析的車手編號 (1=Verstappen, 11=Perez, 22=Tsunoda)
- `features = ['speed', 'throttle', 'brake', 'n_gear', 'rpm', 'drs']` - 使用的特徵
- `SAMPLE_RATE = 10` - 降採樣率 (每 10 個點取 1 個)

**輸出檔案:**
- `output/time_curve_{driver}.csv` - 2D 投影座標與時間進度
- `output/time_curve_{driver}.png` - Time Curve 視覺化圖表

**圖表解讀:**
- **迴圈 (Loops)**: 每個迴圈代表一圈比賽
- **顏色**: 紫色 (起始) → 黃色 (結束),顯示時間進程
- **起點/終點**: 綠色 (Start) / 紅色 (End) 標記

---

### 2. `analyze_corner.py` - S 彎道多車手比較

**功能:**  
分析多位車手在鈴鹿 S 彎 (Sector 1, T3-T7) 的駕駛風格差異,將不同車手投影到同一個相似度空間中。

**執行方式:**
```bash
python scripts/analyze_corner.py
```

**主要參數 (腳本內設定):**
- `TARGET_DRIVERS = [1, 11, 22]` - 比較的車手
- `FEATURES = ['throttle', 'brake', 'n_gear']` - 專注於操作特徵
- 時間窗口: Lap 5 的第 15-45 秒 (涵蓋 S 彎區段)

**輸出檔案:**
- `output/track_map_full.png` - 完整賽道地圖
- `output/roi_check.png` - ROI (S 彎區域) 確認圖
- `output/comparison_scurve.png` - 三位車手的 S 彎比較圖
- `output/comparison_scurve_data.csv` - 原始數據 (供後續分析使用)

**圖表解讀:**
- **軌跡重疊**: 駕駛方式相似
- **軌跡分歧**: 駕駛風格差異 (煞車點、入彎速度等)
- 每位車手以不同顏色標示,起點 (○) 和終點 (×) 清楚標記

---

### 3. `interpret_divergence.py` - 自動分析駕駛差異

**功能:**  
自動找出 Max Verstappen 與 Sergio Perez 在 S 彎中**差異最大的瞬間**,並輸出該時刻的詳細遙測數據。

**執行方式:**
```bash
python scripts/interpret_divergence.py
```

**前置條件:**  
必須先執行 `analyze_corner.py` 生成 `comparison_scurve_data.csv`

**輸出檔案:**
- `output/divergence_report.txt` - 詳細的差異分析報告

**報告內容:**
- 最大分歧點的索引位置
- 相似度空間中的歐幾里得距離
- 兩位車手在該瞬間的完整狀態:
  - 速度 (speed)
  - 油門 (throttle) / 煞車 (brake)
  - 檔位 (n_gear)
  - 引擎轉速 (rpm)

**關鍵發現範例:**
- Max: 激進重煞 (209 km/h, 100% brake)
- Perez: 高速滑行 (225 km/h, 94% throttle)

---

### 4. `analyze_driver_consistency.py` - 車手一致性分析

**功能:**  
深入分析**單一車手**在整場比賽中,每一圈通過 S 彎的操作一致性,生成互動式視覺化。

**執行方式:**
```bash
python scripts/analyze_driver_consistency.py
```

**主要參數 (腳本內設定):**
- `TARGET_DRIVER = 1` - 分析的車手 (預設為 Max Verstappen)
- `FEATURES = ['throttle', 'brake', 'n_gear']`
- `FIXED_POINTS = 30` - 每圈重採樣點數
- `HIGHLIGHT_LAPS = None` - 可指定特定圈次高亮 (例如 `[2, 18, 50]`)

**輸出檔案:**
- `output/driver_laps_consistency.png` - 所有圈次疊加的靜態圖
- `output/driver_laps_data.csv` - 每圈的 MDS 座標數據

**圖表解讀:**
- **白色虛線**: 代表最快圈 (Lap 50, 93.706秒) 作為參考基準
- **背景軌跡**: 所有 52 圈的淡化疊加,展示整體一致性
- **顏色映射**: 紫色 (早期圈次) → 黃色 (晚期圈次)
- **軌跡寬度**: 越窄代表越穩定
- **偏移趨勢**: 觀察輪胎衰退對操作的影響

**互動式網頁版:**  
搭配 `output/consistency_Max.html` 使用,可以:
- ▶️ **播放功能**: 以 0.8 秒間隔自動循環播放所有圈次
- 滑桿或下拉選單選擇特定圈次
- 查看該圈與最快圈的差異
- 紅色連結線顯示當前圈與最快圈起點的偏離距離

---

### 5. `compare_driver_styles.py` - 車手風格對比分析

**功能:**  
比較三位車手在 S 彎的**平均駕駛風格**,計算每位車手所有圈次的平均操作模式並投影到同一相似度空間。

**執行方式:**
```bash
python scripts/compare_driver_styles.py
```

**主要參數 (腳本內設定):**
- `TARGET_DRIVERS = [1, 11, 22]` - 比較的車手 (Verstappen, Perez, Tsunoda)
- `FEATURES = ['throttle', 'brake', 'n_gear']` - 分析的操作特徵
- `FIXED_POINTS = 30` - 每圈重採樣點數
- `COMMON_STEPS = 100` - 平均軌跡的最終點數

**輸出檔案:**
- `output/driver_average_styles.csv` - 三位車手的平均軌跡數據
- `output/driver_average_styles.png` - 靜態對比圖

**圖表解讀:**
- **顏色區分**:
  - 🟣 紫色 = Verstappen (最快: 93.706秒)
  - 🔵 藍色 = Perez (慢 0.239秒)
  - 🟡 黃色 = Tsunoda (慢 2.636秒)
- **軌跡重疊**: 代表相似的平均駕駛風格
- **軌跡分歧**: 顯示操作習慣或賽車性能的差異

**互動式網頁版:**  
搭配 `output/driver_styles_comparison.html` 使用,可以:
- 點擊按鈕切換顯示模式 (全部/單一車手)
- 清楚觀察 Red Bull 隊友間的風格差異
- 對比不同車隊(AlphaTauri)的性能影響

---

## 📖 快速開始 (Quick Start)

如果您是第一次使用本專案,請依照以下步驟操作:

1. **安裝環境:**
   ```bash
   conda create -n lix_f1 python=3.10 -y
   conda activate lix_f1
   pip install -r requirements.txt
   pip install matplotlib seaborn  # 額外依賴
   ```

2. **下載資料 (如果尚未下載):**
   ```bash
   python utils/fetch_f1_data.py
   ```

3. **執行分析流程:**
   ```bash
   # Step 1: 生成完整 Time Curve
   python scripts/generate_time_curve.py
   
   # Step 2: 分析 S 彎比較
   python scripts/analyze_corner.py
   
   # Step 3: 找出駕駛差異點
   python scripts/interpret_divergence.py
   
   # Step 4: 分析三位車手一致性
   python scripts/analyze_driver_consistency.py 1   # Verstappen
   python scripts/analyze_driver_consistency.py 11  # Perez
   python scripts/analyze_driver_consistency.py 22  # Tsunoda
   
   # Step 5: 比較三位車手平均風格 (S 彎)
   python scripts/compare_driver_styles.py
   
   # Step 6: 比較三位車手平均風格 (全局)
   python scripts/compare_driver_styles_full.py
   ```

4. **查看結果:**
   - 靜態圖表: `output/` 目錄下的 PNG 檔案
   - **圖庫預覽 (Quick Preview)**: [**GALLERY.md**](output/GALLERY.md) (直接在 GitHub 查看所有圖表)
   - 互動式網頁 (共6個): 在瀏覽器開啟
     - `output/index.html` - 三車手 S 彎動態追逐
     - `output/consistency_Max.html` - Verstappen 一致性 (播放功能)
     - `output/consistency_Perez.html` - Perez 一致性 (播放功能)
     - `output/consistency_Tsunoda.html` - Tsunoda 一致性 (播放功能)
     - `output/driver_styles_comparison.html` - 全局平均風格對比
     - `output/driver_styles_comparison_S.html` - S 彎平均風格對比

---
