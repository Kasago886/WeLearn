# 单节点重生成

学习者对学习路径里的**某一个节点**不满意，需要你只为它重新设计内容。其余节点保持不变。

## 输入

概念：$concept
学习目标：$goal
掌握度画像：
$profile

整条路径的上下文（这个节点在其中的位置）：
$neighbours

需要重新设计的节点（第 $node_index 个，共 $node_total 个）：
- 当前标题：$node_title
- 当前目标：$objective
- 当前要点：$key_points

学习者的额外要求：$requirement

## 输出要求

只输出这一个节点，不要输出整条路径。要求：

1. `title` 简明、是学习者能一眼看懂的知识点名称（不要带"第 N 步"这类序号）；
2. `objective` 写"学完后能做到什么"，可验证；
3. `key_points` 3~6 条，每条是一个可独立讲解的要点；
4. `checkpoints` 至少 1 个，每个都必须**可判定**：问题明确、`expected_points` 是判定要点而非参考答案、
   `reference_answer` 供判定者参考；
5. 与前后节点的衔接合理，不要重复相邻节点已经讲过的内容；
6. `is_quick_review` 仅在"这个知识点学习者已经掌握、只需快速带过"时为 true。

## 输出

只输出 json，不要任何额外文字或代码块标记。格式如下：

{"title": "先验与后验", "objective": "能说清先验与后验的关系", "prerequisites": ["条件概率"], "key_points": ["先验来自原本的相信程度", "后验是折入证据后的更新", "分母负责归一化"], "checkpoints": [{"question": "请解释 P(A|B) 里每一项的作用。", "expected_points": ["说明后验的定义", "说明先验的作用"], "reference_answer": "P(A|B) 是后验……"}], "is_quick_review": false}
