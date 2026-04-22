# Credit Portfolio Optimization Framework
## Compact implementation snapshot

## 项目摘要
为破解数字经济时代商业银行信贷管理面临的模型“黑箱”、评价主观与决策局限等挑战，提升金融服务实体经济的资源配置效率，本项目构建了一个集“可解释预测—客观量化—全局优化”于一体的三阶段集成式信贷决策框架。

### 核心特性
1.  **可解释预测 (Stage 1)**: 运用 **XGBoost-SHAP** 模型实现风险的透明化预测，解决模型“黑箱”问题。
2.  **客观量化 (Stage 2)**: 结合 **熵权TOPSIS法** 对企业实力进行客观量化，构建包含活力、盈利、杠杆、效率、现金流的五维评价体系。
3.  **全局优化 (Stage 3)**: 首创 **模拟退火-邻域 (SA-NA)** 混合智能算法，求解信贷组合的“收益-风险”双目标优化问题（最大化 RAROC，最小化 CVaR）。

### 主要成果
- **稳健预测**: 针对科技行业（如 NVIDIA）进行了专门的数据校准，显著提升了预测准确性。
- **效益提升**: 优化后的信贷组合在风险调整后收益（RAROC）上表现优异，尾部风险（CVaR）显著降低。
- **算法优势**: SA-NA 算法在解的质量与多样性上优于传统多目标优化算法，为信贷资源配置提供了帕累托改进方案。

## 快速开始

### 环境要求
- Python 3.8+
- 依赖库: pandas, numpy, xgboost, shap, sklearn, matplotlib

### 运行指南
1. **数据准备**: 确保 `data/corporate_credit_rating.csv` 存在。
2. **运行主程序**:
   ```bash
   python main.py
   ```
3. **查看结果**:
   - 控制台输出各阶段运行日志及优化结果。
   - `walkthrough.md` 中包含详细的运行记录和图表分析。

## 目录结构
- `stage1_prediction.py`: 风险预测模块 (XGBoost + SHAP)
- `stage2_evaluation.py`: 信用评级模块 (EWM + TOPSIS)
- `stage3_optimization.py`: 组合优化模块 (SA-NA)
- `config.py`: 系统配置文件
- `data_loader.py`: 数据加载与预处理
