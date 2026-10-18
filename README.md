# 面向异质角色多智能体系统的不确定性传播与主动澄清

**Receiver-Conditioned Latent Uncertainty Propagation for Multi-Agent Clarification**

研究计划 v0.1 · 文献核查日期：2026-10-17 · 状态：**待验证的独立研究方向，没有实验结果**

> **核心问题：同一份上游候选相关的 latent uncertainty，能否帮助不同任务的接收者识别会改变自身判断的歧义，并在相同预算下，选择比丰富结构化文本和通用 latent communication 更有效的跨 agent 澄清？**

本方向延伸既有 Latent Uncertainty / LProp 的候选表示与证据更新思路。**FoodPath / EGPR 已停止；本计划不修改、不重启、不依赖该项目的实验、任务或仓库。** 本文仅交付研究设计，不表示已经训练模型、收集新轨迹或验证创新性。

本文使用三类表述：**文献事实**有原始论文或作者发布页支持；**候选设计**是拟实现的方案；**待验证假设**必须通过实验才能成立。对先前 LProp 的描述依据研究对话中的设计约定，尚未核对其完整论文、代码、checkpoint 或数据文件，不能据此宣称已有可直接复用的实现。

## 阅读入口

- [1. 研究动机与失败链](#1-研究动机与失败链)
- [2. 三种不同的预测与决策对象](#2-三种不同的预测与决策对象)
- [3. 与既有 LProp 的边界](#3-与既有-lprop-的边界)
- [4. 候选方法](#4-候选方法)
- [5. 相关工作与创新边界](#5-相关工作与创新边界)
- [6. 可否证假设与继续投入门槛](#6-可否证假设与继续投入门槛)
- [7. 强基线与公平比较](#7-强基线与公平比较)
- [8. 数据可行性与资产要求](#8-数据可行性与资产要求)
- [9. 实验设计与消融](#9-实验设计与消融)
- [10. 指标与统计协议](#10-指标与统计协议)
- [11. 近期执行计划](#11-近期执行计划)
- [12. 风险、限制与论文主张](#12-风险限制与论文主张)
- [13. 交付物与启动检查单](#13-交付物与启动检查单)
- [14. 原始文献与实现链接](#14-原始文献与实现链接)

## 1. 研究动机与失败链

### 1.1 为什么“不同角色的 uncertainty 尺度不同”还不够精确

检索、实体消歧、事实解释、关系推理和数值计算，具有不同的输出空间、错误事件和下游后果。某个分数增加 0.05，可能表示不同事件的概率变化，也可能只是未经校准的 entropy 变化。即使两个 agent 都准确预测自身有 10% 的错误概率，它们的错误也未必具有相同损失，更未必值得相同的澄清成本。

因此需要分别处理：

1. **概率语义**：原始分数究竟对应哪个可验证错误事件？
2. **任务相关性**：上游哪些候选差异会改变接收者的判断？
3. **行动价值**：哪个允许执行的问题能改变结果，收益是否抵得过成本？

温度缩放、按角色校准或归一化 entropy 都不能单独完成第 2、3 项。概率校准与决策校准已有成熟研究；风险与干预收益的区别也已有直接 agent 研究，不能作为本文首次提出的观点。[Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)；[Decision Calibration, 2021](https://papers.nips.cc/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf)；[Calibration Is Not Control, v2](https://arxiv.org/html/2606.21399v2)

### 1.2 要检验的失败链

~~~text
上游证据存在候选解释、实体歧义或字段冲突
  → handoff 压缩为一个答案和一个置信度
  → 下游看不到“哪项判断依赖哪种解释”
  → 局部生成可以很自信，但实际依赖仍然脆弱
  → 风险阈值只能决定多查或少查，不能可靠选择澄清内容
  → 问到无关问题、重复检查已知内容，或错过真正有用的澄清
  → 错误继续传播，或交互成本增加而最终结果不改善
~~~

这是一条**待检验的机制假设**，不是对所有文本 MAS 的断言。充分的结构化文本可能已经保留全部必要信息；generic latent 也可能已经足够。必须通过强对照定位失败发生在表示、接收者读出还是行动执行环节。

### 1.3 两个必须分开的诊断例子

**例子 A：证明接收者任务相关性。** 上游保留“成立于 1998 年”与“成立于 2001 年”，各占 0.5。接收者 A 判断“是否在 2000 年之前成立”，两种解释改变结论；接收者 B 判断“截至 2020 年是否已成立超过十年”，两种解释得到相同结论。固定相同上游信息，接收者 A 更可能需要澄清。这说明任务条件化有意义，**不能证明 latent 必要**。

**例子 B：检验澄清所需的信息。** 两个 handoff 都输出“1998 年”，报告同样风险；一个源于同名机构混淆，另一个源于注册年份与开始运营年份混淆。有效问题可能分别为“具体指哪个机构？”和“该年份对应哪个事件定义？”。单一错误概率不足以区分它们，但“候选＋概率＋歧义解释＋证据片段”可能足够。

以上为说明性例子，不是数据样本或实验结果。真正的数据还要包含：无需澄清、澄清无解、上游自信但错误、缺失正确候选、澄清反而改坏答案等情形。

## 2. 三种不同的预测与决策对象

### 2.1 记号与信息边界

将执行展开为带时间的消息图。发送者为 $i$，接收者为 $j$，其角色为 $\rho_i,\rho_j$。接收者当前可见状态为 $s_j$，包含子任务 $x_j$、已有消息、当前推理状态和可用工具；$I_i$ 是发送者当时实际可访问的信息。

$Z_i$ 表示拟传递的候选载体。后续执行策略 $\pi$ 在一次比较中固定；$a_0$ 表示直接继续。$S^{a,\pi}\in[0,1]$ 是执行行动 $a$ 后的最终任务得分；$Y^{a,\pi}\in\{0,1\}$ 是预先定义的最终成功事件。得分、成功条件、成本权重和评测终点必须在查看测试结果前固定。

**所有部署特征只能来自决策时刻。** 隐藏真值、未执行的问题答案、未来轨迹和最终评价标签只能用于训练或评估。

### 2.2 局部校准：当前输出是否可靠

对明确的原子输出或字段定义：

$$
E_i=\mathbf 1[\text{发送者选定输出违反预先定义的正确性标准}],\qquad
u_i=\Pr(E_i=1\mid I_i).
$$

例如实体 ID 是否正确、日期字段是否与证据一致、计算结果是否正确。不能把“检索覆盖不足”“实体错误”和“整段文字至少有一句错误”混成同一个未说明的标签。

按角色或 handoff 类型 $g$，校准要求近似满足：

$$
\mathbb E[E_i\mid \widehat u_i=p,g]\approx p.
$$

这是群体频率意义的性质，不是每个样本上的保证。若输出含多个字段，优先记录字段级标签，再明确聚合规则。这里的“局部”指局部评测对象，**不代表已因果隔离的内生错误**；观察到的局部错误可能已经受上游输入影响。

候选权重 $w_{ik}$、token entropy、语义 entropy、self-report confidence 和错误概率不是同一对象。若 $w_{ik}$ 只是支持度，需要单独校准，不能直接宣称为真实后验。

### 2.3 接收者条件化的下游风险：继续执行会发生什么

给定同一 handoff，预测固定策略下的最终失败：

$$
R^{\mathrm{cont}}_{i\rightarrow j}
=\Pr(Y^{a_0,\pi}=0\mid s_j,Z_i).
$$

另可定义接收者本地判断的错误概率，但须另记为 $R^{\mathrm{recv}}$，使用独立标签。本文默认主风险指标是 $R^{\mathrm{cont}}$，不能混用两个终点。

若错误后果不同，可另定义损失风险：

$$
\mathcal R^{\mathrm{cont}}_{i\rightarrow j}
=\mathbb E[L^{a_0,\pi}\mid s_j,Z_i].
$$

概率与损失有不同单位。10% 错误概率并不隐含相同的严重性；跨任务比较损失要说明归一化或效用权重。

**这不是“该 handoff 导致失败的概率”。** 预测最终失败只说明相关性。若要声称传播的因果影响，需要合法的消息替换或修复实验，并控制其他执行条件；仅把上游风险与最终失败相连不够。

### 2.4 澄清效用：问哪个问题值得

每个澄清行动包含收件人、目标字段和问题 $q$。定义：

$$
Q(a\mid s_j,Z_i)=
\mathbb E[S^{a,\pi}-\lambda C^{a,\pi}\mid s_j,Z_i],
$$

$$
G(q\mid s_j,Z_i)=Q(q\mid s_j,Z_i)-Q(a_0\mid s_j,Z_i).
$$

$C^{a,\pi}$ 计入行动及后续执行的成本，不能只计问题长度。若使用 token、延迟、检索次数等多维成本，可用预先固定的权重向量替代 $\lambda C$，同时单独报告各项原始成本。

控制器在共同可行动作集 $\mathcal A(s_j)$ 中选择最大 $Q$；问询只在预计优于继续且满足预算时执行。期望包含上游回答错误、无法回答、接收者误用答案和执行随机性。不能把“拿到 gold 答案后可能改善”当成现实澄清效用。

这是已有 action-value / value-of-information 思路的应用，**不是新公式或新理论**。[Calibration Is Not Control](https://arxiv.org/html/2606.21399v2)；[Value of Information: A Framework for Human–Agent Communication](https://aclanthology.org/2026.acl-long.1987/)

### 2.5 三者的实用区别

| 对象 | 标签来自哪里 | 回答的问题 | 不能据此推出 |
|---|---|---|---|
| $u_i$：局部错误概率 | 输出或字段正确性 | 上游这项输出可靠吗？ | 错误对接收者有多大影响 |
| $R^{\mathrm{cont}}$：继续执行风险 | 同一 prefix 的继续分支 | 当前接收者继续会失败吗？ | 哪个问题能挽救结果 |
| $G(q)$：澄清增益 | 相同 prefix 的行动分支 | 这个具体问题值得问吗？ | 其他问题、策略或成本下也值得 |

## 3. 与既有 LProp 的边界

### 3.1 继承什么

按此前研究约定，latent uncertainty 指：**最终文本未充分保留、但可能支持 trust、verify、clarify、abstain 或 reroute 的信息**；它不等于全部 hidden-state information。

拟继承的资产包括候选内容、候选支持度、候选相关 latent、证据到达后的 belief 更新和 source–receiver handoff 评测。sender risk reader 与候选 carrier 可独立存在，第一版不要求将它们合并为新的通用 uncertainty estimator。

### 3.2 新方向必须额外证明什么

| 维度 | 此前 LProp 的设计约定，待资产核对 | 本方向拟增加的实质问题 |
|---|---|---|
| 语义空间 | 在既定回答空间保留与修订候选 | 上游实体/事实候选映射为接收者的关系/数值/操作判断 |
| 接收者 | source–receiver handoff | 同一上游载体服务多个具有真实依赖的不同子任务 |
| 新证据 | 证据到达后更新 | 主动选择向哪个 agent 询问哪个信息项 |
| 监督目标 | 回答恢复、修订和可靠性 | 固定后续策略下的具体行动收益 |
| 评测终点 | belief 与最终回答质量 | 问题选择、净收益、成本及 harmful clarification |

如果原 LProp 已实现其中某项，该项应标记为复用，不能再次算新贡献。启动前需要记录原方法版本、已有实验和新增代码边界；若无法取得资产，只能称为“受 LProp 设计启发”，不能称为已完成的增量实验。

### 3.3 第一阶段的范围

- 同一开放权重 backbone，承担 3 类实质不同的任务角色。
- 单条 handoff 的接收者条件化读出，加一次 agent-to-agent clarification。
- 优先复用已有 carrier，冻结 backbone；训练小型 projection / risk / action-value heads。
- 首先比较表示与问题选择，再做闭环；复杂图、多轮规划和异构模型对齐放在后续。

不同 persona 不自动等于异质任务。不同 backbone 也不是验证角色异质性的前提。

## 4. 候选方法

### 4.1 候选相关载体

一种最小表示为：

$$
Z_i=\{(a_{ik},w_{ik},h_{ik},e_{ik},m_{ik})\}_{k=1}^{K}.
$$

- $a_{ik}$：语义去重后的候选实体、事实或解释。
- $w_{ik}$：候选支持度；有合适标签和校准后才能按概率解释。
- $h_{ik}$：候选对应的 latent slots，而非无区分的全轨迹平均。
- $e_{ik}$：可检查的文档 ID、片段定位和来源支持/冲突关系。
- $m_{ik}$：字段语义、时间/范围、候选生成方式与有效性信息。

**候选空间不完整必须显式处理。** 记录 other / unresolved 质量与候选覆盖率，不能仅将 top-$K$ 归一化后假装覆盖全部可能世界。候选解释要有互斥或事件关系的定义；多条可同时为真的事实不能直接套用互斥混合分布。

若现有 LProp 没有稳定的 $h_{ik}$，可以比较：候选生成末段的 hidden states、候选条件下对可见证据的编码、少量可学习池化 slots。三者都是候选实现，不能预设哪一种更有“uncertainty”。提取不能读取 gold 解释或澄清之后的信息；逐候选编码增加的计算必须计入。

**Hidden states 不是证据出处。** latent 中读出的“可能是注册年份”只能作为模型判断；最终澄清回答仍需外部可检查的来源或 unresolved 标记。候选标签、证据指针和文本摘要应同时给强文本基线。

### 4.2 接收者侧 projection

令 $b$ 是接收者任务的候选判断，可采用：

$$
p_j(b\mid s_j,Z_i)
=\sum_k w_{ik}\,
T_\theta(b\mid a_{ik},h_{ik},e_{ik},s_j,\rho_i,\rho_j).
$$

该式需要候选完整性和权重语义足够明确，或加入 other 分量。若不能满足，先将其作为可学习的混合近似，另行报告失配。它是条件边缘化，**不是新理论贡献**。

$T_\theta$ 的作用是把上游候选投影到接收者的实际判断空间。最小实现可用 receiver-task query 对 candidate slots 做注意力读出，再分别接 belief、risk 和 action-value heads。角色标签只提供辅助条件，不能替代具体子任务与执行状态。

不要求风险只由 $p_j$ 的 entropy 决定；接收者自身能力、其他上游依赖和执行状态也会影响失败。对开放式答案，可在共同的 receiver candidate set 上诊断 projection，再单独测自由生成，不把候选排名等同于完整任务成功。

第一版只验证一个 sender–receiver handoff。多父节点扩展应保留证据来源与共享祖先标记，不把相同证据重复计算成独立支持；联合依赖建模属于后续验证。

### 4.3 主动跨 agent 澄清

建议固定如下行动协议：

1. 接收者形成临时判断，读取候选载体及自身任务。
2. 共同问题生成器提出有限候选；第一轮用模板或固定生成器，所有表示看到相同候选集合。
3. 控制器为 continue、reread 和各个 targeted question 预测 $Q(a)$。
4. 若选择澄清，发送 sender ID、candidate IDs、目标字段和一个具体问题。
5. 上游只依据当时允许访问的来源回答，并返回证据指针、支持/冲突说明或 unresolved。
6. 接收者按共同更新规则修订候选与任务判断，然后用固定后续策略继续。

问题可覆盖：实体 ID、属性定义、时间范围、单位、证据冲突、某一步计算假设。第一阶段每个问题只询问一个信息项，限制答案长度；不能用一句“请核实所有事实并重做推理”隐藏无限预算。

必须区分三种行动：

| 行动 | 是否获取此前未传出的信息 | 公平性要求 |
|---|---|---|
| reread / recompute | 通常没有，只增加处理 | 与同等计算量的直接重新解释比较 |
| clarify upstream | 取得上游已见、但未充分传达的信息 | 上游无 gold 特权；来源权限相同 |
| retrieve / tool | 可能取得系统此前未见的信息 | 作为不同动作，独立计费和报告 |

主实验先限于第二类及其 continue/reread 对照。新检索可以作为后续扩展，不和澄清混为一个效果来源。若 latent 已包含所有相关内容，而询问只帮助解码，应报告为计算或接口收益。

### 4.4 最小训练方案

冻结 backbone 和既有 carrier，优先训练共享的小型读出器：

$$
\mathcal L=
\lambda_b\mathcal L_{\mathrm{belief}}+
\lambda_r\mathcal L_{\mathrm{risk}}+
\lambda_q\mathcal L_{\mathrm{value}}.
$$

- belief loss：仅在有可靠 receiver judgement 标签时使用。
- risk loss：针对固定 continue 分支的失败，用 proper scoring loss。
- value loss：用相同 prefix 下各行动的实际收益样本或均值回归，可辅以带噪声/近似并列处理的排序损失。

各项权重只在开发集选择；训练后在独立 calibration split 拟合概率校准。若样本不足，先分别训练 heads，避免多任务复杂度遮蔽主要问题。

所有主要基线获得相同训练标签、可行动作和后续策略。**仅对 proposed 训练澄清效用、对 Interlat 只训练回答准确率，会产生不公平的目标差异。** 不把最终失败标签直接当作“这个问题有用”的标签。

## 5. 相关工作与创新边界

### 5.1 最直接的比较对象

以下是截至核查日的证据摘要；“可能的区别”是本项目拟测试的目标，**不代表已经证实文献空白**。发表状态与版本见文末登记。

| 工作 | 已覆盖的关键能力 | 对本方向的约束与必要比较 |
|---|---|---|
| [CIPHER](https://arxiv.org/abs/2310.06272) | 通过词汇概率加权的 embedding 交流，避免只传一个采样 token | 保留比离散文本更丰富的分布信息已有先例；不能把 soft/latent message 本身作为创新 |
| [LatentMAS](https://arxiv.org/html/2511.20639v4) | latent thoughts、层级 KV 工作记忆传递、顺序和层级角色协作 | 已讨论中间错误及后续修正；比较其接入相同澄清控制器后的能力 |
| [Interlat](https://arxiv.org/html/2511.09149v5) | last-layer latent 通信、训练式适配与压缩、下游效用和 uncertainty-aware 目标、跨模型实验 | 最关键强对照；不能称其“不考虑 uncertainty 或 downstream utility” |
| [PropUQ-MAS](https://arxiv.org/html/2608.22130v3) | 沿执行图组合局部风险与继承风险，使用错误接收强度 | 已有传播及 receiver acceptance；拟比较内容级歧义与问题选择是否超越图风险标量 |
| [CAGE-Cal](https://arxiv.org/html/2605.30653v1) | 对比通信前后依赖图，以模型族、prompt role、答案组及拓扑校准 panel confidence | 角色/依赖感知校准已有直接先例；其最终 panel 目标与异质子任务 handoff 目标要分开 |
| [SAUCE](https://arxiv.org/abs/2610.08901) | 多 agent 同题推理中，用多轮一致性与生成不确定性顺序更新系统 belief | 不能声称首次动态系统 UQ；其 latent belief 不等于神经 hidden-state carrier |
| [MATU](https://arxiv.org/abs/2604.08708) | 将多次 MAS 完整轨迹的 embedding 组织为张量并估计不确定性 | 已超出最终答案置信度；重复 rollout 与完整轨迹成本、时间可用性需单列 |
| [AUQ](https://arxiv.org/abs/2601.15703) | 在记忆中保存 verbalized confidence 与语义解释，以解释触发针对性反思 | “不确定的原因指导恢复”已有文本实现；必须采用丰富文本＋相同行动监督的对照 |
| [Calibration Is Not Control](https://arxiv.org/html/2606.21399v2) | 风险与干预价值区分、相同 prefix 分支、匹配监督目标比较及在线控制 | 本项目借用其决策与评估思想；新的候选表示必须优于直接学习行动价值的紧凑读出 |
| [TarMAC](https://proceedings.mlr.press/v97/das19a.html) | 从下游任务奖励学习发什么、发给谁及多轮沟通 | recipient-aware communication 不是新概念；不同点只能落在此处具体表示、监督和证据 |

### 5.2 三个最容易误写的地方

**Interlat。** v5 的压缩目标包括 frozen actor 的任务交叉熵、完整与压缩 latent 所诱导行为的 KL 一致性，以及 latent 方向对齐；一致性权重与 latent 相对无 latent 的预测 entropy 降低有关。它已经考虑接收者使用价值。本文拟增加的是候选相关表示对多问题澄清选择的增量价值；需与保留原目标、再接相同效用监督和控制器的 Interlat 对照。[原文 §4、Appendix B](https://arxiv.org/html/2511.09149v5)

**PropUQ-MAS。** 其递推为：

$$
r_v=1-(1-u_v)\prod_{p\in\mathrm{Pa}(v)}(1-\alpha_{pv}r_p).
$$

$\alpha_{pv}$ 表示上游错误被接收的条件概率；实现使用自报 adoption score 代理。乘积依赖 incoming contamination 及 local/propagation 独立性假设。不能说它“无差别累加”或“完全忽略接收者”。本项目要检验标量及接收权重是否遗漏具体疑点的行动信息；其图建模可作为基线或可组合部件。[原文 §3、Appendix A–B](https://arxiv.org/html/2608.22130v3)

**CAGE-Cal。** 它校准的是多个 agent 对同一问题形成的最终 plurality prediction，已经纳入 role 与模型族等依赖；它的 no-communication 对照也需要额外运行。将其改造成异质子任务链的 prefix 风险模型属于适配，必须单独标记，不能把原论文完整结果与改造版混称。[原文 §3–4](https://arxiv.org/html/2605.30653v1)

### 5.3 必须纳入的邻近研究

| 工作 | 已有内容及本项目应承认的边界 |
|---|---|
| [SAUP / Uncertainty Propagation on LLM Agent，ACL 2025](https://aclanthology.org/2025.acl-long.302/) | 已研究多步 agent 的过程不确定性和情境相关传播；不能笼统声称既有 UQ 只看最终输出 |
| [T2MAC，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29438) | 选择性交互与 evidence-driven integration；说明针对对象、时机和可信证据的通信有更早 MARL 基础 |
| [Decision Calibration，NeurIPS 2021](https://papers.nips.cc/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf) | 从下游决策者需求定义校准；任务相关校准不是从零提出的新理论 |
| [Value of Information: A Framework for Human–Agent Communication，ACL 2026](https://aclanthology.org/2026.acl-long.1987/) | 用问题的预期效用与交互成本决定澄清；由人机改成 agent-to-agent 不自动构成方法创新 |
| [On Calibration of Modern Neural Networks，ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | 温度缩放等后处理属于标准校准工具，应作为可复用组件 |
| [LatCom，2026 预印本](https://arxiv.org/abs/2609.37017) | 多 sender latent 融合为固定 slots，按 receiver task utility 学习压缩；进一步限制“任务相关 latent 压缩”的新颖性主张 |

LatCom 是本轮补充检索发现的近邻；已核查原始摘要，尚未做方法和代码逐项复现。若扩展到多 sender 融合或以压缩效率为主张，必须进一步核对全文并加入直接实验。检索不是穷尽性证明；正式投稿前还需刷新候选表示、主动问询、MARL 通信与新 latent MAS 工作。

### 5.4 暂时允许的创新表述

> 我们研究异质子任务 handoff 中候选相关 latent 表示对接收者条件化澄清的作用，并通过匹配表示、控制器、监督及预算的实验，检验它是否保留了丰富文本和通用 latent 通信未有效利用的行动信息。

以下表述目前**不成立**：首次 latent MAS；首次角色校准；首次 uncertainty propagation；首次用 uncertainty 指导行动；latent 天然 grounded；latent 必然优于文本；新建小网络即方法创新；可以保证某个会议接收。

## 6. 可否证假设与继续投入门槛

### 6.1 研究假设

| 假设 | 可观察的支持证据 | 反证或失败后的处理 |
|---|---|---|
| H1：任务改变澄清价值 | 固定 sender 信息，在真实相关的 receiver 任务之间，最佳问题或问/不问选择发生可重复变化；任务条件化降低 regret | 若 role ID 改名有影响但实际任务依赖无影响，不能支持异质任务主张 |
| H2：候选相关 latent 有增量价值 | 超越丰富结构化文本、普通语义 embedding 和匹配 generic latent / Interlat；候选–latent 对应关系有可重复贡献 | 若只能胜裸标量，撤回 latent 独特性；若 generic latent 相同，撤回 uncertainty-specific carrier 主张 |
| H3：问题选择改善最终净收益 | 相同问题预算和可访问证据下，提高任务得分或净效用，并降低 regret；非仅增加提问数量 | 若只提高离线 probe 或问题相关性，不能声称闭环澄清有效 |
| H4：收益跨任务而非记忆模板 | 对未见实体、文档、任务组合或 handoff 类型仍有效；有足够组内样本和不确定区间 | 若只在训练角色模板成立，将结论限定为该分布 |
| H5：校准支持可解释风险语义 | role/handoff 分组 proper scores 与可靠性图合理，且校准前后分别报告 | 若全局 ECE 好但组内差，不能声称解决跨角色可靠性 |

H1–H3 是主线；H4 约束泛化范围，H5 约束 calibration 主张。不能用 H5 替代 H3，也不能用 H1 证明 H2。

### 6.2 分阶段 gate

以下数值是**建议的资源管理门槛，不是文献结论或通用统计标准**。Day 1 可依据资产与成本调整一次并登记；在测试集封存后不得逐结果调整。所有门槛目前均为“未执行”。

| Gate | 需要完成的证据 | 继续条件与停止条件 |
|---|---|---|
| G0：资产与可重放性 | 审计约 50 个 handoff；核对模型版本、来源、候选、latent、prefix | 能重放或合法重建；若 hidden states 不可得且重建超预算，停止 latent 实验路线，不悄悄替换为文本后继续称 latent |
| G1：实际机会 | 约 200 个 receiver decision-prefix 的探索性 pilot；同源配对按 parent 分组，含自然与机制样本 | 建议至少 20 个有可回答且正收益的澄清，并至少 20 个合法 receiver 配对显示稳定的行动差异；不足时只允许预登记的一轮数据审计，不无限制造歧义 |
| G2：表示增量 | 在 pilot 开发划分比较强文本、generic latent 与候选 latent，检查容量与信息权限 | 存在可复核的效应及明确成本预估才扩大；正向点估计不算确认新颖性，结构化文本或 generic latent 已足够时停止该表示主张 |
| G3：独立确认 | 封存测试集、匹配控制器与预算，至少 3 个训练随机种子 | 建议最小实用效应 $\delta_U=0.02$（得分归一到 [0,1]、成本权重固定）；对两个主要强对照，平均净效用增益达到 $\delta_U$，经多重比较处理的 95% 区间下界 > 0 |
| G4：机制与适用范围 | 闭环复验、关键消融、自然分布评估与未见任务组合 | 关闭 latent 对应关系或任务条件后有符合预测的变化；闭环收益维持。harmful clarification 的增幅上侧 95% 界建议 ≤1 个百分点，否则报告权衡并撤回“更可靠”主张 |

G1 只是检查是否值得研究，不是显著性检验。G3 的最终样本量由 pilot 方差、配对设计和预先选择的检验确定；约 200 个 pilot 样本通常不足以确认小效应。若区间过宽，应判为**证据不足**，而不是“证明无效”或“证明有效”。

预算到限、gate 未过或强基线消除收益时，保留负结果、配置和成本记录；不靠增加角色、模块或改写故事绕过门槛。

## 7. 强基线与公平比较

### 7.1 必须包含的表示阶梯

所有可部署基线都保留正常的继续、重读、修订、恢复和同一组澄清动作，不能人为削弱文本系统。

| 编号 | 发送/读取的信息 | 回答的替代解释 |
|---|---|---|
| B0 | 普通文本 MAS，直接继续；另测 always-ask、random-ask | 是否仅仅多花预算就改善 |
| B1 | 局部校准风险＋receiver task/state；增加标准 uncertainty 特征 | 标量及任务条件是否已经足够 |
| B2 | 候选＋权重＋字段范围＋证据指针 | 候选结构本身的贡献 |
| B3，主要强对照 | B2＋结构化 uncertainty explanation＋支持/冲突证据摘要 | 显式解释是否已保留澄清所需信息 |
| B4 | B3＋普通语义 embedding，或匹配长度的文本 latent 编码 | 额外语义内容/维度是否足以解释增益 |
| B5 | PropUQ-MAS 原始传播分数；另测其风险特征＋B3 | 图风险传播与接收权重是否已经足够 |
| B6，主要强对照 | B3＋generic sender hidden-state payload，匹配 slots、层、维度与训练预算 | 是否只需更多 hidden states，无需候选相关结构 |
| B7，关键文献对照 | Interlat 表示与压缩＋共同问题集、行动价值监督和控制器 | 是否只是 Interlat 加澄清读出器 |
| B8 | LatentMAS 接入共同澄清能力；兼容时加入 CIPHER | 现有 latent 协作能否覆盖同一用途 |
| B9 | 既有 LProp 原版，外接相同 action-value reader；再加 receiver conditioning | 原 carrier 与新增条件化各贡献多少 |
| P | B3＋候选对应 latent＋receiver projection | 拟研究的完整最小方案 |
| O1 / O2，诊断上界 | 合法来源内最佳问题；另列具有 gold 信息的回答上界 | 问题集/回答能力限制与表示限制分别多大 |

B3 必须认真优化提示、摘要长度和字段表达，且有充分开发预算。增加一个 full-evidence text 条件，允许接收者读取上游实际已见的完整证据，以诊断损失是否主要来自人为压缩；它可以更贵，但需画出成本–效用曲线。

对 B1/B3/B6/B7/P 都训练行动价值读出，不能让 P 独占效用标签。加入**直接预测每个问题收益的紧凑标量或多标量接口**；这可检验是否需要显式 belief/projection 中间层，而非只需要更合适的监督目标。

### 7.2 “相同控制器”到底固定什么

主实验将系统拆为：

~~~text
表示 R
  → 相同模型族、容量上限与监督的 action-value estimator
  → 完全相同的 controller C(Q, costs, remaining_budget)
  → 同一回答协议与 continuation policy
~~~

控制器 C 的问题集合、成本权重、阈值、并列处理和预算规则完全一致，不能按方法手工优化。各表示的输入适配器可不同；action-value estimator 使用相同架构和训练/调参预算，但各自训练权重。**“相同架构与协议”不等于“所有神经网络权重相同”，报告中须写清。**

补充一个更严格的共享读出实验：在均衡混合的训练表示上训练公共读出器，冻结其权重，只匹配训练各方法输入适配器；避免只用 P 训练公共读出器后要求其他表示迁就它。再给基线独立调优的读出结果，排除共享接口不公平。

Interlat 需区分：

- **原版复现**：保留官方训练和压缩目标，确认实现有效。
- **共同任务适配**：使用相同 backbone、数据、问题集、容量及行动效用监督。
- **差异隔离实验**：保留其原目标，再加入与 P 相同的效用监督；额外训练成本计入。

若无法在资源内完成 B7，pilot 可先用 B6 筛选；最终必须明确“尚未排除 Interlat”，不能宣称已超越最直接先例。LatentMAS 的 KV cache 与 fixed slots 不能简单按 token 数等价，需同时报告字节、显存和端到端延迟。

### 7.3 比较的两个预算视角

1. **内容与训练匹配**：相同可见来源、候选、解释、问题集、训练 parent tasks、监督量、head 参数预算和随机种子。
2. **资源曲线匹配**：在不同通信字节、GPU 时间和问询预算下比较 Pareto 曲线；文本 token 与 latent slot 不能直接视为等成本。

候选抽样、解释生成、latent 提取、adapter 训练、压缩、校准和离线分支数据都要登记成本。统一输出长度不是充分公平条件。

### 7.4 系统 UQ 工作如何纳入

CAGE-Cal、SAUCE、MATU 的原生评价对象可能是完整 panel 或完整轨迹，不宜直接假装为 handoff 时刻可用的读出器：

- 在兼容的同题 panel / trajectory 子实验中按原任务使用。
- 用于在线 prefix 时，只使用当时可得信息；适配版本单独命名、说明限制。
- 重复运行、无通信对照、额外 entropy 或 embedding 提取的成本全部报告。
- 不将完整未来轨迹生成的特征送入在线 controller；做不到时仅作为离线分析基线。

## 8. 数据可行性与资产要求

### 8.1 优先数据路线

**HotpotQA。** 多跳问答及 supporting facts 可用于构建检索、事实解释、比较推理的依赖。已有 LProp 轨迹若确实保存候选与来源，可先审计复用。但原始 QA 数据没有 agent clarification 的行动收益标签。[原始论文](https://aclanthology.org/D18-1259/)

**MuSiQue。** 单跳问题组合形成的多跳依赖有利于构造合法 sender–receiver 关系。仍需设计角色、信息权限和问题协议，不能把它宣称为现成的异质角色澄清 benchmark。[原始论文](https://aclanthology.org/2022.tacl-1.31/)；[数据与实现](https://github.com/stonybrooknlp/musique)

**可控机制集。** 使用独立、可验证的小型实体/日期/单位/关系事实集，构造“相同上游、不同下游决策边界”和“相同风险、不同歧义原因”。它适合验证机制，不足以单独支持真实工作流结论。合成样本与自然样本分别报告。

第一阶段不引入新的复杂应用领域。若只剩合成数据可跑，结论必须限定为机制验证。

### 8.2 真实角色和信息权限

最小链可以包含：

1. **证据 agent**：读取许可文档，返回候选证据与来源。
2. **解释 agent**：区分实体、字段定义、时间与单位。
3. **推理 agent**：完成比较、关系组合或计算，形成最终答案。

角色必须有不同的可评测输出契约，不能只修改“你是一名谨慎专家”等 persona。sender 不能预先看到全部 receiver 的 gold 答案；receiver 可以看到自己的任务。paired-receiver 诊断固定 sender 输入、生成和载体，分别接入两个都真实依赖该信息的子任务。

上游回答澄清时只能使用其已有许可来源；若允许检索，所有方法拥有同一权限并把它记为 retrieve 动作。答案允许 unresolved，禁止总能回答的 oracle sender。疑问与证据摘要在测试时不能由 gold 标签生成。

### 8.3 现有资产审计清单

| 资产 | 必需记录 | 缺失时的可行处理 |
|---|---|---|
| 模型与接口 | checkpoint/revision、tokenizer、层、dtype、推理配置 | 不能取得 hidden states 的 API 无法直接验证 latent carrier |
| 候选 | 候选文本/ID、生成条件、去重规则、支持度、other 信息 | 可重建但计入费用；不能从 gold 反推候选 |
| latent | 提取时间、候选对应关系、张量形状、mask、文件 hash | 只有普通文本日志不能无损恢复原 hidden states |
| evidence | 原文或许可缓存、文档 hash、span、可见角色 | 来源失效时标记不可重放，不用当前网页冒充旧证据 |
| receiver prefix | 当前任务、消息序列、局部状态、工具状态 | 需要可重放快照或明确限定为文本 QA |
| labels | 字段正确性、最终成功、每个行动分支结果 | 只有终局正确性不足以监督具体澄清效用 |
| 版本与权限 | 数据版本、许可、split、代码版本 | 先记录可发布范围，避免交付不可复现资产 |

目前未验证上述资产是否存在，也未把任何历史“已有 HotpotQA 数据”的叙述当作可运行证据。

### 8.4 最小日志契约

以下是字段契约，**不是已生成数据的样例**：

~~~yaml
identity:
  parent_task_id: string
  source_cluster_id: string
  prefix_id: string
  split: train | calibration | validation | test
state:
  sender_role: string
  receiver_role: string
  receiver_task: string
  visible_message_ids: list
  visible_source_ids: list
  replay_snapshot_ref: string
carrier:
  candidates: list  # id, content, weight, scope, evidence references
  unknown_mass: number_or_null
  latent_ref: string_or_null
  extraction_manifest_ref: string_or_null
actions:
  candidates: list  # action id, recipient, target field, question, access policy
  decision_features_timestamp: string
branches:
  outcomes_ref: string  # restricted evaluation/training side only
  continuation_policy_id: string
  responder_policy_id: string
  random_seeds: list
costs:
  input_tokens: integer
  output_tokens: integer
  transmitted_bytes: integer
  tool_calls: integer
  gpu_seconds: number
  wall_seconds: number
provenance:
  model_revision: string
  prompt_hash: string
  evidence_snapshot_hash: string
  code_commit: string
~~~

另存 schema version、候选顺序、问题生成器版本、评分器版本与错误记录。tensor 可存 safetensors/其他明确格式，元数据存 JSONL；不要把未来分支 outcome 字段直接拼入训练输入。

### 8.5 分支标签与泄漏控制

- 从完全相同且可重放的 prefix 分别执行 continue、reread、q1…qM。
- 答案来源权限、responder 和后续策略固定；不能只改 P 的后续模型。
- 单次分支是随机收益的一个样本，不能当成真实期望。对近似并列及关键样本重复运行。
- 用 train 收集监督，用 calibration 做概率校准，用 validation 选择模型与预算策略，test 只作最终评估。
- 同一 parent task 的所有 prefix、receiver 变体、释义、候选变化与行动分支必须在同一 split；必要时按共享文档/实体族成组隔离。
- 不从测试失败案例挑选“最好解释故事”的训练样本；记录所有过滤规则与过滤率。
- 对天然歧义富集样本报告条件结果，同时保留未筛选自然任务集及实际歧义发生率。

### 8.6 先计算数据成本

若有 $N$ 个 prefix、每个有 $M$ 个问题、另有 continue 和 reread、每个行动重复 $R$ 次：

$$
N_{\mathrm{branch}}=N(M+2)R.
$$

这里的 $N$ 按 receiver decision-prefix 计数，同一 parent 的两个接收者计为两个 prefix。例如 $N=200,M=3,R=2$ 需要 2,000 次后续分支，**还未计入**上游生成、问题生成、失败重试、latent 提取和闭环评估。这只是计划算例，不代表已经执行或已获预算。

用 20 个开发样本实测平均/高分位 token、时间和存储，再估算总成本。建议 pilot 默认上限为上述 2,000 分支及预登记 GPU/API 预算，先到者停止。第一轮不要为了训练一个小 head 而启动未经估算的大规模轨迹采集。

## 9. 实验设计与消融

### E0：接口与可重放性审计

确认输出契约、证据可见性、candidate–latent 对应、快照重放和成本记录；检查 hidden states 中没有未来信息。先测不训练的结构化文本，确认问题是任务本身需要，而非 latent 代码才制造出来。

**输出**：资产清单、失败样本清单、版本登记、预算估算、G0 决定。

### E1：同一上游、不同接收任务

固定 sender 输入、消息、候选、权重与 latent，仅改变真实相关的 receiver 子任务。比较 task-aware、role-only 和 task-blind 读出，测候选对接收者判断的影响以及最佳问题变化。

分别记录：候选分歧不影响判断、影响但无法澄清、影响且可澄清。随机替换不相关 receiver 不算合法证据。此实验只验证任务条件化的需要。

**输出**：合法配对清单、效用矩阵、任务条件化增益及区间、G1 决定。

### E2：共同问题池上的表示诊断

每个 prefix 使用相同 2–4 个可执行问题，加 continue 和 reread；问题来源不使用任何方法专有 latent 或 gold 标签。比较 B1/B3/B4/B6/B9/P，G2 通过后补齐 Interlat 强对照。

同时测风险预测、问题收益估计与 regret。问题顺序随机化，recipient 与字段分布平衡；手工审计一部分问题是否因措辞或长度泄露正确动作。测 oracle question coverage，防止固定问题池遗漏真正有用的问题。

**输出**：表示阶梯结果、容量与训练匹配记录、错误分类、G2 决定。

### E3：闭环 agent-to-agent 澄清

冻结控制器、问题生成器、成本权重与模型选择，在未见任务上真实执行决策。第一轮每个任务最多一次 targeted clarification，下一轮才扩展到 0/1/2 次预算曲线。

比较同一 ask 数量预算下的净效用，也比较达到相同成功率所需成本。记录 no-ask、无解、回答错误、重问、恢复成功和从正确变错误的样本。继续分支用于评价反事实时属于离线评估成本，不计入部署成本，但必须单列。

**输出**：闭环任务得分、净效用、成本曲线、harmful clarification、G3/G4 决定。

### E4：泛化与传播深度

按资源顺序测试未见实体/文档、未见任务组合、未见 handoff 类型、2–3 条消息依赖，再考虑新 backbone。第一阶段的同模型异质角色结论不能外推为跨模型 latent 对齐成立。

对多父节点加入共享证据、相关错误、冲突证据与缺失来源；对多轮通信按时间展开，避免循环传播中重复累计同一证据。若没有这些实验，只声称在指定局部 handoff 上有效。

### 必需消融

| 消融 | 控制方式 | 试图区分的机制 |
|---|---|---|
| 去掉 receiver task | 保留角色标签和同容量网络 | 是否只识别角色平均难度 |
| 去掉/改名 role ID | 保留真实任务描述 | 角色标签是否只是捷径 |
| 去掉候选 latent | 保留完整 B3 内容与参数匹配读出 | latent 是否有增量价值 |
| generic latent 替换 | 匹配层、slots、来源与训练 | 候选相关结构是否必要 |
| candidate–latent 对应打乱 | 在适当匹配组内置换并保留边际统计 | 是否利用候选对应关系 |
| 仅候选均值/单候选 | 总载体容量尽量匹配 | 是否依赖多候选差异，而非额外容量 |
| 去掉权重 / 去掉解释 / 去掉证据关系 | 每次只移除一项 | 概率、显式语义与 grounding 各自贡献 |
| 无校准 / 全局 / 分组或层级校准 | 同一原始分数与独立 calibration split | 校准改善与表示改善是否混淆 |
| 风险目标 vs 行动效用目标 | 表示、head、控制规则和数据规模固定 | 是否主要由监督目标造成收益 |
| 去掉中间 belief projection | 直接从同一表示预测各问题收益 | 中间结构是否必要 |
| reread 替代提问 | 匹配计算与输出预算 | 澄清是否只是额外推理 |
| carrier 长度、候选 K 与提问预算 | 预登记小范围 sweep | 效果是否只来自带宽/计算 |
| unresolved 与错误回答压力测试 | 相同响应协议 | 是否依赖完美上游回答 |

零向量、噪声或任意置换可能造成分布外输入；其性能下降不能单独证明“latent 中编码了某种真实 uncertainty”。需要结合匹配替代表示、自然样本和行为证据解释。

## 10. 指标与统计协议

### 10.1 概率校准

针对每个明确的二元错误事件 $e_n$ 和预测 $\hat p_n$：

$$
\mathrm{Brier}=\frac1N\sum_n(\hat p_n-e_n)^2,
\qquad
\mathrm{NLL}=-\frac1N\sum_n
[e_n\log\hat p_n+(1-e_n)\log(1-\hat p_n)].
$$

实现时对概率 clipping 的数值范围预先登记。报告全局、role、handoff 类型和关键任务组的 Brier、NLL、可靠性图及分箱 ECE；固定分箱规则并给样本量与区间。组内样本不足时不能做确定性结论；可采用层级校准，并单列未见组。

Brier 和 NLL 是 proper scores，也受区分能力影响；下降不单独证明“纯校准误差下降”。ECE 对分箱和样本量敏感，不能作为唯一指标。AUROC、AUPRC、AURC / risk–coverage 衡量排序或选择性预测，不代替概率校准。

候选分布另测正确候选覆盖率、other 命中、候选级 NLL/Brier；若正确答案未在候选集中，必须计入覆盖失败，不能排除该样本后只报校准。

**时间对齐**：$R^{\mathrm{cont}}$ 必须对继续分支标签校准；不能用澄清后的实际成功去评判干预前 continue-risk。干预后 risk 使用新的状态、明确的新预测目标另报。

### 10.2 澄清质量与行动价值

| 指标 | 定义或评测要求 |
|---|---|
| 最终任务得分 | QA 的 EM/F1 或任务专属成功规则；预登记主指标 |
| 净效用 | 实际任务得分减预登记成本，原始成本另报 |
| 决策 regret | $\max_{a\in\mathcal A}\bar U_a-\bar U_{\hat a}$，仅针对实际评估的共同动作集 |
| useful-question selection | 在确有正收益问题的样本中，选中正收益行动的比例；并报告无可用正收益问题的比例 |
| 问题排序 | top-choice agreement、Kendall/Spearman 或 NDCG；近似并列不强制唯一 gold |
| 收益估计质量 | 预测 $G(q)$ 与重复分支平均增益的 MAE/RMSE、分箱预测–实际增益曲线 |
| recovery rate | 继续会失败的样本中，澄清后成功的比例 |
| harmful clarification | 继续会成功的样本中，澄清后失败的比例；同时报告全任务口径 |
| 可回答性与证据支持 | 问题是否单一明确、回答有无来源支持、是否询问真正分歧 |
| 无效交互 | unresolved、重复问题、与决策无关问题及答非所问比例 |
| 成本 | 问题数、信息项数、token、传输字节、工具调用、GPU 时间、墙钟延迟、峰值显存 |

$G$ 是连续收益差，不能直接当概率计算 binary ECE。若额外预测“回答可解决目标字段”的概率，可以用相应可观察事件校准；若预测“正收益概率”，必须明确定义随机事件与重复分支标签，不能把单次幸运改善当成期望正收益。

oracle regret 受有限问题池和分支噪声影响。用独立重复样本估计行动均值与评估所选行动，或报告重复次数敏感性与 oracle 不确定区间，避免对同一批噪声取 max 后夸大 regret。gold-answer oracle 仅用于 headroom 分析，不能进入可部署主表。

### 10.3 统计与结果呈现

- 主比较预登记为 P 对 B3，以及 P 对匹配 generic latent / Interlat；完整方法主张要求后者实际完成。
- 用 parent task 或更高层 source cluster 做配对 bootstrap，保留同源 prefix/receiver 的相关性；不能把每条分支当独立样本。
- 至少 3 个训练随机种子，报告种子间变异及任务级区间；关键分支重复运行以估计响应噪声。
- 先定最小实用效应和功效目标，再依据 pilot 方差确定确认性样本量；探索性分析单独标记。
- 多个主要比较使用预登记的校正方案，例如 Holm；不从多个成本权重和预算点中挑最有利结果。
- 报告所有任务、歧义子集、自然样本、合成样本及失败类型；不能只展示被 controller 选择提问的样本。
- 问题/回答质量以可检查来源为主，盲评一部分样本并记录评审一致性；不能只使用生成同一答案的模型自评。
- 至少输出四组图表：分组可靠性、效用–成本、表示阶梯及消融、receiver 配对效用矩阵。结果为空时保留“未运行”，不填写推测分数。

## 11. 近期执行计划

这是一个**约 10 个工作日的可行性冲刺建议**，从资产可用后开始计算，不代表既定投稿期限，也不承诺在此期间完成全文复现或确认性实验。先通过 gate，再扩大投入。

| 时间 | 具体行动 | 可审阅交付物 | 决策点 |
|---|---|---|---|
| Day 1 | 审计 50 个既有 handoff；20 个样本测成本；核对 LProp 边界和访问权限 | 资产清单、模型/数据版本、主指标、预算上限、未解决依赖 | G0；不能重放或无法合法取得 latent 则停止对应路线 |
| Day 2 | 固定 3 类角色、错误标签、问题协议、same-prefix 分支与分组 split | 任务契约、日志 schema、评测规则、封存 split manifest | 审阅是否存在真实任务依赖和可回答的澄清 |
| Day 3–4 | 按预算采集约 200 个 pilot receiver decision-prefix；验证 paired receivers 与问题收益 | 分支数据、成本账本、自然/合成样本比例、oracle headroom | G1；没有稳定可恢复空间则不训练复杂方法 |
| Day 5–6 | 运行 B1/B3/B4/B6/B9/P 的冻结主模型读出实验 | 表示阶梯、训练容量表、候选覆盖与泄漏审计 | G2；强文本或 generic latent 足够则撤回独特性主张 |
| Day 7 | 完成最关键的任务条件、候选对应、监督目标及 reread 消融 | 消融结果、失败原因分类、确认性样本量估计 | 若收益来自额外计算或目标差异，收窄结论 |
| Day 8–9 | 在 G2 通过后适配 Interlat；小规模未见任务闭环；校准及成本曲线 | Interlat 适配记录、在线日志、延迟/字节对照 | 尚未完成强对照时保留“未排除直接先例” |
| Day 10 | 综合证据、复核负结果、决定停止或进入确认性实验 | 一页 go/no-go memo、更新后的 README、可复现配置与资产索引 | 不以写出论文故事替代 G3/G4 |

G3/G4 可超出这 10 天；如果硬件、现有 LProp 实现或分支数据不足，不压缩强对照来制造“按期完成”。下一阶段只在估计成本、实用效应和最直接对照均清楚后启动。

**工作量优先级：**

1. 验证可回答问题确实改变最终结果。
2. 验证 receiver task 改变最值得问的内容。
3. 用强文本与 generic latent 尽早否证 latent 独特性。
4. 再做 Interlat 对照、闭环与泛化。
5. 最后才考虑多轮规划、多 sender 融合、异构 backbone 或更复杂训练。

## 12. 风险、限制与论文主张

### 12.1 必须诚实面对的失败可能

- **丰富文本已经足够。** 若 B3 与 P 等效，就没有证据支持 latent 在当前任务上的必要性；可保留任务相关澄清分析，但不能包装成 latent 方法成功。
- **generic latent 已经足够。** 若 B6/B7 复现增益，收益可能来自带宽、语义内容或行动监督，不是 uncertainty-specific carrier。
- **表示有信息但接收者不会用。** probe 可预测不代表闭环会改善；还需验证读出、答案使用和执行链。
- **干预本身没有价值。** 问题不可回答、来源不包含答案或错误已不可恢复时，优质 UQ 也不能创造收益。
- **候选缺失或支持度失真。** latent 不能自动弥补正确候选从未出现；必须看覆盖和 other 事件。
- **低 entropy 也会错。** 自信错误、共同盲点与相关来源可能让所有 agent 一起收敛到错误结论。
- **校准发生分布漂移。** 新角色、证据域、模型或控制策略会改变风险事件频率，原校准不保证继续有效。
- **数据成本可能主导。** 行动分支、重放与人工来源核验可能远贵于训练小模块。
- **表示无法跨模型直接复用。** 相同维度也不意味着 latent 语义对齐；跨模型结论需要单独实验。
- **自然歧义稀少。** 在人工富集数据上的效应可能无法转化为自然任务平均收益。
- **共享证据被重复计算。** 多 agent 同意不等于独立证据；本项目也不能靠简单候选混合宣称解决所有依赖。
- **不透明载体的可检查性有限。** 保留证据指针和可读的候选/问题/回答；不把 latent 的解释性猜测当作事实依据。

### 12.2 不同结果应对应什么结论

| 实验结果 | 可以说什么 | 不可以说什么 |
|---|---|---|
| H1 成立，H2 不成立 | 不同任务需要不同澄清；强文本足以支持 | latent 提供独特价值 |
| P 优于标量但不优于 B3 | 显式候选与解释优于单一风险摘要 | latent uncertainty 方法有效 |
| P 与 generic latent / Interlat 相当 | 通信内容或共同效用监督有用 | 候选 carrier 是必要的新机制 |
| 离线排序提高，闭环无收益 | 表示含有诊断信息，行动链尚未受益 | 实现更有效的主动澄清 |
| 成功率提高但成本过大 | 在额外资源下有准确率收益 | 相同预算更高效 |
| 自然集无效、合成集有效 | 受控条件下的机制证据 | 普遍适用真实 MAS |
| H1–H3 成立且强对照与消融通过 | 在明确任务、模型、预算范围内有增量方法证据 | 首次解决 MAS uncertainty 或已保证发表 |

**可能的贡献组合，必须由结果选择：**

- 一个精确、可重放的异质子任务澄清评测协议。
- 一组说明上游风险、接收者风险与行动价值分离的实证结果。
- 若强对照通过，一个候选相关 latent 载体及接收者读出的方法增量。
- 有价值的负结果：何时结构化文本、紧凑行动值或现有 latent communication 已经足够。

前两项也需要与已有数据集和评测协议继续比较，不能因为方法没有增益就自动宣布 benchmark 新颖。

## 13. 交付物与启动检查单

### 13.1 当前交付与未来资产分开

**当前已交付：本 README 研究计划及文献链接。** 尚未生成实验数据、实现训练代码、执行模型复现或创建新 GitHub 仓库。

未来若建立独立研究仓库，建议使用以下目录；这是规划，不是现成可执行代码：

~~~text
independent-mas-uncertainty/
  README.md
  docs/
    novelty-ledger.md          # 文献版本、已有能力、拟主张、对应证据
    lprop-boundary.md          # 复用资产与新增部分
    protocol.md                # 信息权限、动作、标签、评测终点
    gate-decisions.md          # go/no-go、成本、负结果
  configs/
    pilot.yaml
    baselines.yaml
    evaluation.yaml
  manifests/
    assets.json
    splits.json
    versions.json
  data/
    prefixes.jsonl
    carriers/
    branch-outcomes/
  src/
    carrier/
    receiver/
    clarification/
    evaluation/
  results/
    calibration/
    clarification/
    ablations/
    costs/
~~~

发布最小资产应包括数据生成/过滤脚本、问题生成规则、合法来源快照或可重建索引、模型与代码版本、分组 split、全部对照配置、原始结果与错误日志。若不能发布 tensor 或来源正文，说明限制并提供可再生成步骤。

### 13.2 启动前

- [ ] 独立目录与资源账本已建立，未修改或复用停止项目的任务配置。
- [ ] 原 LProp 论文/代码/资产已核对，新增贡献边界写明。
- [ ] backbone hidden-state 接口与硬件可用。
- [ ] 候选、证据、receiver prefix 可关联并重放。
- [ ] 局部事件、continue-risk、任务效用、成本分别定义。
- [ ] 共同问题集、上游回答权限与 unresolved 规则已冻结。
- [ ] 主要强对照至少覆盖 B3、B6，以及最终所需 B7。
- [ ] 训练、校准、开发和测试按 parent/source cluster 隔离。
- [ ] pilot 预算、分支上限、gate 阈值与停止规则已登记。

### 13.3 对外宣称方法成立前

- [ ] 有实际执行的强对照；没有把原版与适配版混称。
- [ ] 闭环净收益通过预登记标准，成本与损害同时报告。
- [ ] 风险校准标签与预测时间一致，没有未来分支泄漏。
- [ ] receiver task 与 candidate–latent 对应的贡献有证据。
- [ ] 自然任务结果、失败案例、未见任务及负结果均保留。
- [ ] 对文献作最后一次更新，尤其是 Interlat、LatentMAS、LatCom 与主动澄清近邻。

## 14. 原始文献与实现链接

下列链接以原始论文、正式论文页面和作者仓库为主。2026 年文献更新较快，复现实验应 pin 版本和 commit。作者在 arXiv/仓库中声明的接收信息，与正式 proceedings 核验分开记录；本 README **未复现任何论文性能数值**。

### 14.1 直接相关工作

1. **CIPHER — Let Models Speak Ciphers: Multiagent Debate through Embeddings.** 2023 预印本；ICLR 2024 论文。核查其分布加权 embedding 通信机制。  
   [arXiv](https://arxiv.org/abs/2310.06272) · [ICLR 论文 PDF](https://openreview.net/pdf/44f4dae92e4c22176324d6eb9893ab81be3ad031.pdf)

2. **LatentMAS — Latent Collaboration in Multi-Agent Systems.** 2025 首稿；本次按 arXiv v4（2026-08-03）核查；arXiv 与作者仓库标注 ICML 2026 Spotlight。  
   [v4 全文](https://arxiv.org/html/2511.20639v4) · [版本记录](https://arxiv.org/abs/2511.20639) · [作者实现](https://github.com/Gen-Verse/LatentMAS)

3. **Interlat — Enabling Agents to Communicate Entirely in Latent Space.** ACL 2026；本次方法比较按 arXiv v5（2026-07-13），不要把 v5 的所有细节未经检查地归入早期版本。  
   [ACL 页面](https://aclanthology.org/2026.acl-long.1248/) · [v5 全文](https://arxiv.org/html/2511.09149v5) · [作者实现](https://github.com/XiaoDu-flying/Interlat)

4. **PropUQ-MAS — Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems.** 本次按 v3（2026-09-01）；arXiv 注明 accepted to EMNLP 2026 Main，本次未另核对正式 proceedings。  
   [v3 全文](https://arxiv.org/html/2608.22130v3) · [版本与接收备注](https://arxiv.org/abs/2608.22130)

5. **CAGE-Cal — Counterfactual Graph for Multi-Agent LLM Calibration.** arXiv v1，2026-05-28；本次按预印本处理。  
   [全文](https://arxiv.org/html/2605.30653v1) · [作者实现](https://github.com/J-Huang01/Counterfactual-Graph-Calibration-for-Multi-Agent-LLMs)

6. **SAUCE — Sequential Probabilistic Uncertainty Estimation for Parallel Multi-Agent Reasoning Systems.** arXiv v1，2026-10-06；作者仓库标注 NeurIPS 2026，本次未独立核对正式 proceedings。核查原始摘要与作者算法说明。  
   [论文](https://arxiv.org/abs/2610.08901) · [作者实现](https://github.com/Tyrion58/SAUCE)

7. **MATU — Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition.** arXiv v1，2026-04-09；arXiv 与作者仓库标注 ACL 2026，未另核对 anthology 条目。核查轨迹张量表示及重复运行输入要求。  
   [论文](https://arxiv.org/abs/2604.08708) · [作者实现](https://github.com/tiejin98/MATU)

8. **AUQ — Agentic Uncertainty Quantification.** arXiv v1，2026-01-22；本次按预印本处理。核查 uncertainty-aware memory 与 reflection 的设计。  
   [论文](https://arxiv.org/abs/2601.15703)

9. **Calibration Is Not Control: Intervention Value for LLM-Agent Oversight.** 本次按 v2（2026-10-06）；arXiv 注明 NeurIPS 2026，未另核对正式 proceedings。v1 标题表述不同，引用与实验协议应固定版本。  
   [v2 全文](https://arxiv.org/html/2606.21399v2) · [版本记录](https://arxiv.org/abs/2606.21399) · [作者实现](https://github.com/bennidict23/calibration-is-not-control)

10. **TarMAC: Targeted Multi-Agent Communication.** ICML 2019，PMLR 97。  
    [正式论文页面](https://proceedings.mlr.press/v97/das19a.html)

### 14.2 邻近方法与基础

11. **T2MAC: Targeted and Trusted Multi-Agent Communication through Selective Engagement and Evidence-Driven Integration.** AAAI 2024。  
    [正式论文页面](https://ojs.aaai.org/index.php/AAAI/article/view/29438)

12. **Uncertainty Propagation on LLM Agent.** ACL 2025；方法名 SAUP。  
    [正式论文页面](https://aclanthology.org/2025.acl-long.302/)

13. **Calibrating Predictions to Decisions: A Novel Approach to Multi-Class Calibration.** NeurIPS 2021。  
    [正式论文 PDF](https://papers.nips.cc/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf)

14. **Value of Information: A Framework for Human–Agent Communication.** ACL 2026。此处引用其澄清价值与成本框架，不引入其应用场景。  
    [正式论文页面](https://aclanthology.org/2026.acl-long.1987/)

15. **On Calibration of Modern Neural Networks.** ICML 2017，PMLR 70。  
    [正式论文页面](https://proceedings.mlr.press/v70/guo17a.html)

16. **LatCom: Cross-Agent Latent Compression for Efficient Multi-Agent Collaboration.** arXiv v1，2026-09-29；本次只核查原始摘要，方法和代码细节待深入审计。  
    [论文](https://arxiv.org/abs/2609.37017)

### 14.3 数据来源

17. **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.** EMNLP 2018。  
    [正式论文页面](https://aclanthology.org/D18-1259/)

18. **MuSiQue: Multihop Questions via Single-hop Question Composition.** TACL 2022。  
    [正式论文页面](https://aclanthology.org/2022.tacl-1.31/) · [数据与实现](https://github.com/stonybrooknlp/musique)

---

**当前研究判断：值得做一次预算受控、可否证的可行性诊断；尚未确认一个未被现有工作覆盖的新 gap。主要判据是：在相同控制与资源条件下，候选相关 latent 是否比丰富文本和匹配的 Interlat / generic latent 更好地选择澄清，并把这种优势转化为真实任务净收益。**
