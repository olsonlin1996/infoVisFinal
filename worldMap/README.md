# 世界地圖專案結構概覽

此資料夾包含多個以 F1 主題為核心的視覺化工具，依用途與產出分類如下：

- **animation/**：F1 動態賽道示意頁面，含前端資源、抓取腳本 (`fetch_2024_*`) 及標記圖示。
- **cache/**：FastF1 及其他下載的暫存檔，無需版本控管。
- **data/**：靜態資料來源（賽道 geojson、場次資料、天氣與結果等）。
- **scatter_plot/**：車手風格嵌入與分析工具，含產出的 json 與日誌。
- **timeCurve-f1/**：彎道與圈速分析工具，包含原始資料、腳本與輸出。
- **index.html / japan-insights.html / japan-style-insights.html**：主要網頁入口與主題頁面。
- **style.css / worldMap.js**：共用樣式與互動腳本。

建議透過 `.gitignore` 排除快取、日誌與大量產出檔，保持版本庫輕量。
