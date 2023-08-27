
from dataclasses import dataclass


@dataclass
class ContractsPromptsTemplate:
    INTERMEDIATE="""这是该问题的历史处理记录："""

    PREFIX = """请使用下列给出的工具回答问题，你能使用的工具如下："""

    TOOLS  ="""Comparison: 支持2个数值的大小比较，如果不等式成立，返回值为True，如果不等式不成立，返回值为False"""

    FORMAT_INSTRUCTIONS = """你需要通过传入一个特定格式的JSON来调用这些工具。\
这个被传入的$JSON_BLOB应该包含两个关键词：`action`(指定你想调用的工具的名称)和`action_input`(工具的输入)

对于这个问题，你能在"action"里面填入的工具有：Comparison

这个特定格式的JSON只能包含一个action，不能包括多个action。以下是标准的$JSON_BLOB输出样例:

```
{
    "action": $TOOL_NAME,
    "action_input": $INPUT
}
```

输出的格式要求：
Question: 需要回答的问题
Thought: 对问题拆解后的回复。
Action:
```
$JSON_BLOB
```
Observation: 工具的输出结果

... (这个 Thought/Action/Observation 的过程可以重复多次)
Thought: 我知道最终答案了
Final Answer: 对原始问题给出的最终解答。
"""

    SUFFIX = """开始回答问题，记住在给出最后答案的时候，一定要包含关键词`Final Answer`。""" 