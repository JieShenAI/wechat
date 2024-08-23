@[toc]
## few-shot

相比大模型微调，在有些情况下，我们更想使用 Few-shot Learning 通过给模型喂相关样本示例，让模型能够提升相应任务的能力。

**固定样本提示  VS 动态样本提示**：

<font color="red">固定样本提示：每次都用同样的样本提示去推理；
动态样本提示：根据当前要推理的样本，基于向量相似度算法，在训练集中找出相似的样本作为提示去推理。
</font>


**Few-shot Learning (少样本提示学习)**：

- **定义**：Few-shot learning 是通过给模型提供少量示例（例如 1-5 个）来进行任务的学习方式。这些示例通常包括输入和相应的输出。
- **实现方式**：在大多数情况下，few-shot learning 是在模型的输入中直接包含这些示例作为提示。这意味着模型本身没有经过任何额外的训练或调整。
- **优点**：可以快速适应新任务，无需额外的训练时间和资源。



##  Fixed Examples 固定样本
以聊天模型为例，
```python
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

parser = StrOutputParser()

model = ChatOpenAI(model="gpt-4o-mini")
```



```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]
```



🦜 代表加法。想让大模型根据给出的例子学会🦜 代表加法。



```python
# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```


```python
few_shot_prompt.invoke({}).messages
```

Output:

```python
[HumanMessage(content='2 🦜 2'),
 AIMessage(content='4'),
 HumanMessage(content='2 🦜 3'),
 AIMessage(content='5')]
```



```python
few_shot_prompt.format()
```

Output:

```python
'Human: 2 🦜 2\nAI: 4\nHuman: 2 🦜 3\nAI: 5'
```



```python
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
# chain = model | final_prompt
chain = final_prompt | model

chain.invoke({"input": "What's 3 🦜 3?"})
```

Output:

```python
AIMessage(content='Based on the previous pattern, the 🦜 operation appears to be addition. Therefore:\n\n\\[ 3 🦜 3 = 3 + 3 = 6 \\]', response_metadata={'token_usage': {'completion_tokens': 37, 'prompt_tokens': 30, 'total_tokens': 67}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='run-xxx', usage_metadata={'input_tokens': 30, 'output_tokens': 37, 'total_tokens': 67})
```

如上模型的输出结果所示，模型已经能够学到🦜是加法，并返回  3 🦜 3 = 3 + 3 = 6 。



## Dynamic few-shot prompting 动态样本提示

> 为什么要有一个动态的 few-shot 呢？
>
> 在上一节 Fixed Examples中，无论输入什么问题，都只使用固定的例子作为提示。
>
> 动态例子提示是：针对不同的问题，使用不同的例子进行提示。目的是为了提高模型的性能。

如果你想评估 动态few-shot的效果，那么便逐个遍历测试集的样本数据，根据测试集的样本使用向量相似度算法从训练集中拿到最相似的几个样本，再去做 few-shot prompting。

> 我们考虑在下一篇文章，为大家评估动态few-shot的效果。当前文章只是教学文章，不想整的太复杂。

在前一个章节中使用：
`ChatPromptTemplate` 和`FewShotChatMessagePromptTemplate`，

在本章节中使用：
`PromptTemplate` 和 `FewShotPromptTemplate` 

上述一一对应，不能混用。



```python
from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
```
下述代码展示了 example_prompt 使用效果：

```python
print(example_prompt.invoke(qa_examples[0]).text)
```

Output:

```python
Question: Who lived longer, Muhammad Ali or Alan Turing?

            Are follow up questions needed here: Yes.
            Follow up: How old was Muhammad Ali when he died?
            Intermediate answer: Muhammad Ali was 74 years old when he died.
            Follow up: How old was Alan Turing when he died?
            Intermediate answer: Alan Turing was 41 years old when he died.
            So the final answer is: Muhammad Ali
```

下述的 `qa_examples` 是一个训练集，供模型推理时，在其中选择向量最相似的样本。

```python
qa_examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
            Are follow up questions needed here: Yes.
            Follow up: How old was Muhammad Ali when he died?
            Intermediate answer: Muhammad Ali was 74 years old when he died.
            Follow up: How old was Alan Turing when he died?
            Intermediate answer: Alan Turing was 41 years old when he died.
            So the final answer is: Muhammad Ali
            """,
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Who was the founder of craigslist?
            Intermediate answer: Craigslist was founded by Craig Newmark.
            Follow up: When was Craig Newmark born?
            Intermediate answer: Craig Newmark was born on December 6, 1952.
            So the final answer is: December 6, 1952
            """,
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Who was the mother of George Washington?
            Intermediate answer: The mother of George Washington was Mary Ball Washington.
            Follow up: Who was the father of Mary Ball Washington?
            Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
            So the final answer is: Joseph Ball
            """,
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Who is the director of Jaws?
            Intermediate Answer: The director of Jaws is Steven Spielberg.
            Follow up: Where is Steven Spielberg from?
            Intermediate Answer: The United States.
            Follow up: Who is the director of Casino Royale?
            Intermediate Answer: The director of Casino Royale is Martin Campbell.
            Follow up: Where is Martin Campbell from?
            Intermediate Answer: New Zealand.
            So the final answer is: No
            """,
    },
]
```

`example_prompt` 作为参数 放入到 FewShotPromptTemplate 模版中，实现对 `qa_examples`中的数据进行封装。

```python
from langchain_core.prompts import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=qa_examples,
    example_prompt=example_prompt,
    # prefix="You are a helpful assistant.",
    suffix="Question: {input}",
    input_variables=["input"],
    
)

print(
    prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)
```

这里是不使用向量筛选器prompt。若调用 invoke 方法，FewShotPromptTemplate会把qa_examples中**所有的样本**都封装好作为上下文。

Output:

```python
Question: Who lived longer, Muhammad Ali or Alan Turing?

            Are follow up questions needed here: Yes.
            Follow up: How old was Muhammad Ali when he died?
            Intermediate answer: Muhammad Ali was 74 years old when he died.
            Follow up: How old was Alan Turing when he died?
            Intermediate answer: Alan Turing was 41 years old when he died.
            So the final answer is: Muhammad Ali
            			......
Question: Who was the father of Mary Ball Washington?
```

使用编码模型构建向量筛选器，将qa_examples经过编码后，保存到 Chroma 向量数据库中。

```python
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    qa_examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1,
)
```

使用 `example_selector` 根据用户输入的问题，找一个最相似的样本出来：

```python
# Select the most similar example to the input.
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print("\n")
    print('【')
    for k, v in example.items():
        print(f"{k}: {v}")
    print('】')
```

Output:

```python
Examples most similar to the input: Who was the father of Mary Ball Washington?


【
answer: 
            Are follow up questions needed here: Yes.
            Follow up: Who was the mother of George Washington?
            Intermediate answer: The mother of George Washington was Mary Ball Washington.
            Follow up: Who was the father of Mary Ball Washington?
            Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
            So the final answer is: Joseph Ball
            
question: Who was the maternal grandfather of George Washington?
】
```

使用向量选择器example_selector和提示词封装器example_prompt，构建最终的prompt。

同时可以在 `FewShotPromptTemplate` 添加后缀和前缀。一般前缀用来添加系统提示词，后缀用来添加问题。

```python

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    # prefix="You are a helpful assistant.",
    suffix="Question: {input}",
    input_variables=["input"],
)

print(
    prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)
```

Output:

```python
Question: Who was the maternal grandfather of George Washington?

            Are follow up questions needed here: Yes.
            Follow up: Who was the mother of George Washington?
            Intermediate answer: The mother of George Washington was Mary Ball Washington.
            Follow up: Who was the father of Mary Ball Washington?
            Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
            So the final answer is: Joseph Ball
            

Question: Who was the father of Mary Ball Washington?
```



```python
chain = prompt | model
chain.invoke({"input": "Who was the father of Mary Ball Washington?"})
```

Output:

```python
AIMessage(content='The father of Mary Ball Washington was Joseph Ball.', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 103, 'total_tokens': 113}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0f03d4f0ee', 'finish_reason': 'stop', 'logprobs': None}, id='run-ae96f9c7-ac89-47ba-8074-69197b89bef5-0', usage_metadata={'input_tokens': 103, 'output_tokens': 10, 'total_tokens': 113})
```



## 辅助

与huggingface 通过代理连接

```python
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
```



## 参考资料

下述是2个langchain的官方说明文档，均写的很不错：

* [https://python.langchain.com/v0.2/docs/how_to/few_shot_examples_chat/](https://python.langchain.com/v0.2/docs/how_to/few_shot_examples_chat/) How to use few shot examples in chat models

* [https://python.langchain.com/v0.2/docs/how_to/few_shot_examples/#pass-the-examples-and-formatter-to-fewshotprompttemplate](https://python.langchain.com/v0.2/docs/how_to/few_shot_examples/#pass-the-examples-and-formatter-to-fewshotprompttemplate) How to use few shot examples