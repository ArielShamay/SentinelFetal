|Method|Recall|Precision|Alert Rate|F2-Score|Config / Notes|
|---|---|---|---|---|---|
|Old Hybrid (Simple OR)|87.6%|32.9%|58.3%|0.657|Baseline|
|XGBoost Solo (Simple OR)|83.8%|35.2%|52.1%|0.657|Did it win Exp A?|
|Smart Tiered Logic (Best)|81.9%|32.0%|56.0%|0.624|T_high=0.55, T_low=0.25, engine=ensemble|

⚠️ Logic Improvement Insufficient — Alert load remains high.