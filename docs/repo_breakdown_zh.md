# DMDR 仓库解读（中文）

> 对应论文：Distribution Matching Distillation Meets Reinforcement Learning（arXiv:2511.13649）。

## 1. 仓库定位

这个仓库目前**公开的是 ImageNet class-conditional 的训练 demo**，主干模型是 SiT，不是完整的通用文生图工程。

- 根目录 `README.md` 明确说当前只开放 ImageNet 训练演示代码，并把训练入口指向 `train_cc/sit/README.md`。
- 推理部分主要引导到 Z-Image 仓库；本仓库核心价值在于 DMDR 训练逻辑。

## 2. 目录结构（最关键）

- `train_cc/sit/train_sitxl_refl_deepspeed.py`：核心训练循环（DMD + RL 联合训练）
- `train_cc/sit/models_sit.py`：SiT 模型定义、LoRA 注入 Q/V 注意力
- `train_cc/sit/samplers.py`：采样与反向模拟（`v2x0_sampler`、`euler_sampler`、`pred_v`）
- `train_cc/sit/utils.py`：时间步采样策略（含动态分布采样）、奖励模型预处理、checkpoint 路径映射
- `train_cc/sit/arguments.py`：所有训练超参配置入口
- `train_cc/sit/scripts/*.sh`：冷启动和第二阶段（ReFL）训练启动脚本
- `train_cc/sit/convert_weight/sit_conver.py`：把 REPA 预训练权重转换到当前 SiT 结构

## 3. 代码如何映射论文方法

### 3.1 两阶段训练（冷启动 + 联合 RL）

- `scripts/cold_start_sitxl.sh`：设置 `--cold-start-iter 20001`，意味着这段训练仅做冷启动分布蒸馏。
- `scripts/dmdr_sitxl_refl.sh`：从冷启动 checkpoint 恢复，开启 `--encoder-type dinov2` 和 `--dino-loss-weight`，进入 DMD+RL 联训。

### 3.2 DynaDG（动态分布引导）

论文里 DynaDG 的核心是初期把 real-score 分支往 student 分布方向拉近，后续再逐步衰减。

在代码中对应：

- Guidance 模型支持 LoRA，且生成器权重会复制到 guidance 分支，LoRA 参数独立学习；
- 训练中 `lora_scale_r` 使用余弦衰减：早期较大，后期趋近 0；
- 在 DMD 计算里，`v_pred_real` 走带 CFG 和 `lora_scale_r` 的分支，`v_pred_fake` 走 fake 分支，二者差值形成梯度方向。

### 3.3 DynaRS（动态重噪声采样）

论文里 DynaRS 是“先偏向大噪声 level，后逐步过渡”。

在代码对应为 `sample_discrete` / `sample_continue`：

- 当 `dynamic_step > 0` 时，用 cosine 退火把 Beta 分布参数逐步回退到更均匀的采样形态；
- `--s-type-gui logit_normal`、`gui-a/gui-b` 共同控制早期时间步采样分布。

### 3.4 RL 与 DMD 的联合损失

训练脚本中生成器最终损失：

- `dmd_loss_mean`（分布匹配蒸馏）
- `+ dino_loss_weight * reward_loss_dino_mean`（奖励项，默认 DINOv2 分类交叉熵）

这正对应论文强调的“DMD 作为 RL 正则，RL 反过来改善模式覆盖”的联合优化。

## 4. 训练脚本读法（实战）

建议按这个顺序读代码：

1. `arguments.py`：先理解参数面板（尤其 `cold-start-iter` / `dynamic-step` / `lora-scale-r` / `ratio-update` / `dino-loss-weight`）。
2. `scripts/cold_start_sitxl.sh` 与 `scripts/dmdr_sitxl_refl.sh`：看默认配方如何把参数拼起来。
3. `train_sitxl_refl_deepspeed.py`：
   - 建模部分：`gen_model`（学生生成器） + `guidance_model`（score 分支）
   - 采样部分：`v2x0_sampler` 产生反向模拟轨迹
   - guidance update：拟合 fake score
   - generator update：DMD 梯度 + reward 梯度联合
4. `samplers.py` 和 `utils.py`：补全 “时间步怎么采”“x0 怎么还原”“CFG/Lora 怎么作用” 等细节。

## 5. 你需要特别注意的工程点

- 该 demo 的 RL 奖励模型是 DINOv2 分类头（ImageNet 设定），不是通用文生图奖励模型。
- 这是分布蒸馏 + RL 的研究型代码，不是一键推理产品仓库。
- 脚本默认多卡 DeepSpeed，单卡用户需要改 `accelerate` 配置和 batch size。
- 输出目录里同时存了 deepspeed accelerator state 与 `torch_units/step-xxx.pt`（后者主要记录 global/inner step）。

## 6. 一句话总结

这个 repo 的核心思想是：**把“few-step 蒸馏”和“奖励优化”做成同一个训练过程**，并用 DynaDG + DynaRS 提升早期稳定性；代码上最关键就是 `train_sitxl_refl_deepspeed.py` 里 guidance/generator 两条更新路径如何交替与耦合。
