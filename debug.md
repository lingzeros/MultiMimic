**当前问题**：在dual decoder架构中，无论是否加上fk_pose监督，都会有提前松手的情况。

调试顺序：
1. 在单 decoder ACT 和 dual decoder ACT 中，修复raw_action的索引，对比原先的索引与新的“错误”索引是否都能work

2. 选择能够正常work的索引，做无temporal agg的对照
    - 如果无temporal agg时没有提前松手的情况，则问题来自temporal agg

3. 尝试arm分支继续temporal agg，hand分支优先使用最新chunk的query 1

