---
title: LangChain
date : 2024-04-24 15:42:38 +0800
categories: [机器学习]
tags: [计算机, LLM, LangChain]
---

<!-- TOC -->

- [大模型](#大模型)
  - [Azure](#azure)
  - [ChatGLM3](#chatglm3)
- [Langchain](#langchain)
  - [提示和提示模板](#提示和提示模板)
  - [链](#链)
  - [嵌入查询](#嵌入查询)
  - [记忆](#记忆)
  - [langchain表达式](#langchain表达式)
  - [函数和工具](#函数和工具)
    - [自带的工具](#自带的工具)
    - [函数调用](#函数调用)
    - [工具调用](#工具调用)
    - [BaseTool](#basetool)
- [LTP](#ltp)

<!-- /TOC -->

## 大模型

### Azure

环境变量配置：

```py
import os

os.environ['AZURE_OPENAI_API_KEY'] = API_KEY
os.environ['AZURE_OPENAI_ENDPOINT'] = ENDPOINT
```

创建实例

```py
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
```

续写类AI：

```py
start_phrase = 'def bubble_sort(array):'
response = client.completions.create(model=DEPLOYMENT_NAME, prompt=start_phrase, max_tokens=40)
print(response.choices[0].text)
```

对话类AI：

```py
response = client.chat.completions.create(model=DEPLOYMENT_NAME, messages=[
    # ['system', 'assistant', 'user', 'function']
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
    {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
    {"role": "user", "content": "Do other Azure AI services support this too?"},
])
print(response.choices[0].message.content)
```

文本需要的tokens数

```py
llm.get_num_tokens(text)
```

### ChatGLM3

直接调用：

```py
glm_path = '../chatglm3-6b'

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(glm_path, trust_remote_code=True)
model = AutoModel.from_pretrained(glm_path, trust_remote_code=True).half().cuda()
model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
```

启动openai服务，然后用langchain：

```py
from langchain.chains import LLMChain
from langchain.schema.messages import AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.prompts import PromptTemplate

template = """{question}"""
prompt = PromptTemplate.from_template(template)

endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

messages = [
    AIMessage(content="我将从美国到中国来旅游，出行前希望了解中国的城市"),
    AIMessage(content="欢迎问我任何问题。"),
]

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    prefix_messages=messages,
    top_p=0.9,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "北京和上海两座城市有什么不同？"

llm_chain.run(question)
```

## Langchain

### 提示和提示模板

使用llm

```py
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.llms.openai import AzureOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

messages = [
    SystemMessage(content="你是一个回答问题的机器人，请用简短的语言回答下面的问题。"),
]

llm = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    api_version="2024-02-15-preview")

user_input = '说出一个位于欧洲的国家。'
messages.append(HumanMessage(content=user_input))
response = llm(messages).content
print(response)
messages.append(AIMessage(content=response))

user_input = '这个国家的首都在哪里？'
messages.append(HumanMessage(content=user_input))
response = llm(messages).content
print(response)
messages.append(AIMessage(content=response))
```

模板使用

```py
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template = '我需要你为新开店面起一个好听的名字，一家卖{product}的商店可以起什么名字？'
prompt = PromptTemplate(
    input_variables=['product'],
    template=template
)
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm
)

print(llm_chain.invoke('家具')['text'])
input_list = [
    {"product": "水果"},
    {"product": "蛋糕"},
    {"product": "蔬菜"}
]
llm_chain.apply(input_list) # 可对列表运行
llm_chain.generate(input_list) # 附带更多生成信息
```

少样本上下文提示

```py
llm = AzureOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    api_version="2024-02-15-preview",
    max_tokens=5)


prefix = '''你是一个考研政治多项选择题回答AI，以下是若干例题供你参考，请你回答问题的时候只用选择题的选项A、B、C、D来回答问题，每道题至少有两个正确选项。
'''

examples = [{
    'query': '''随着新一代人工智能技术的发展，基于大模型的生成式人工智能(AIGC)在快速回答提问、创作代码、翻唱经典歌曲等方面取得了新的突破。但是，随着技术迭代，人工智能高效地应用于各行各业时，其带来的风险也不容忽视，比如人工智能生成近似原画的内容、构图，可能侵犯原创作者的知识产权;人工智能技术被恶意使用，可能用来从事制造虚假信息、诈骗等违法活动。守住法律和伦理底线，推动人工智能朝着科技向善的方向发展，关键还在于人们更智慧地使用人工智能工具。“更智慧地使用人工智能工具”意味着
A.技术进步要以维护人民的根本利益为最高标准
B.人类活动能够实现合目的性与合规律性的统一
C.科技发展是由主观意志决定的客观物质活动
D.成功的实践是真理尺度与价值尺度的统一
''', 'answer': 'ABD'}, {'query': '''习近平指出：“人类文明多样性是世界的基本特征，也是人类进步的源泉。世界上有200多个国家和地区、2500多个民族、多种宗教。不同历史和国情，不同民族和习俗，孕育了不同文明，使世界更加丰富多彩。”唯物史观关于社会形态的理论中，内在地包含着文明多样性的思想。下列关于人类文明多样性表述正确的有
A.独特的生产方式和生活方式决定着文明发展的不同样态
B.各种文明都具有独自的比其他文明更优越、更强大的文化基因
C.每一种文明都代表着一方文化的独特性，是人类文明的重要组成部分
D.每一种文明都是在与其他文明相隔离的状态下独自产生、发展和演变的
''', 'answer': 'AC'}, {'query': '''商品经济是社会经济发展到一定阶段的产物。在资本主义社会之前的发展阶段，商品经济只是一种简单商品经济，这一阶段商品经济发展的基础是
A.生产资料公有制
B.个体劳动
C.生产资料私有制
D.雇佣劳动
''', 'answer': 'BC'}, ]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"], template=example_template
)

suffix = """
User: {query}
AI:"""

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

query = '''今年以来，我国经济持续回升向好，高质量发展扎实推进，我国仍是全球增长最大引擎。据权威部门统计，前三季度我国国内生产总值同比增长5.2%;全国居民人均可支配收入同比实际增长5.9%;高技术产业投资增长11.4%。前10个月社会物流总额同比增长4.9%,物流需求恢复向好，行业提质升级加速。总体上看，我国经济长期向好的基本面没有变也不会变，因为我国具有
A.超大规模市场的需求优势
B.产业体系配套完整的供给优势
C.社会主义市场经济的体制优势
D.大量高素质劳动者和企业家的人才优势
'''

print(prompt.format(query=query))

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm
)

print(llm_chain.invoke(input={'query': query})['text'])
```

这里有工具可以通过选择器来从大量示例中选择和问题合适的对应的示例来当做提示，不过略。

输出解析器指导输出json格式：

```py
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

response_schemas = [
    ResponseSchema(name="bad_string", description='这是一个格式不正确的用户输入符串'),
    ResponseSchema(name='good_string', description='这是重新格式化后的字符串')
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = """
你将从用户那里得到一个格式不正确的字符串，重新格式化并确保所有单词拼写正确。

{format_instructions}

User：
{user_input}

AI："""

prompt = PromptTemplate(
    input_variables=['user_input'],
    partial_variables={'format_instructions': format_instructions},
    template=template
)

prompt_string = prompt.format(user_input='im a student.')
print(prompt_string)
print(llm.invoke(prompt_string).content)
```

实际使用的format_instructions是

````
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{
	"bad_string": string  // 这是一个格式不正确的用户输入符串
	"good_string": string  // 这是重新格式化后的字符串
}
```
````

另一个例子

```py
response_schemas = [
    ResponseSchema(name="name", description="学生的姓名"),
    ResponseSchema(name="age", description="学生的年龄")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="回答下面问题.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

_input = prompt.format_prompt(question="给我一个学生的档案")
output = cllm.invoke(_input.to_string()).content
print(output)
print(output_parser.parse(output))
```

另一种是使用`PydanticOutputParser`，不过之后用到再细看。

### 链

数学链：

```py
from langchain.chains import LLMMathChain

llm_math = LLMMathChain(llm=llm, verbose=True)
llm_math('what is 13 to the power of 0.4')
```

顺序链，有利于分解任务，保持LLM专注

```py
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate

template = '''根据用户输入的位置，推荐一道该地区的经典菜肴，回复菜名。
User: {user_input}
AI:'''
prompt = PromptTemplate(
    input_variables=['user_input'],
    template=template
)
location_chain = LLMChain(llm=llm, prompt=prompt)

template = '''根据用户输入的菜名，说明如何做这道菜。
User: {user_input}
AI:'''
prompt = PromptTemplate(
    input_variables=['user_input'],
    template=template
)
dish_chain = LLMChain(llm=llm, prompt=prompt)

chain = SimpleSequentialChain(chains = [location_chain, dish_chain], verbose=True)
chain.run('北京')
```

总结链

```py
from langchain.chains import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader(path, encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

texts = text_splitter.split_documents(documents)
texts = texts[:5]

# map_reduce是指每段总结 最后再把所有总结做一个总结
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts)
```

SQL数据库链

这里是代码，但是没执行过，暂时跳过

```py
from langchain_experimental.sql.base import SQLDatabaseChain, SQLDatabase
db = SQLDatabase.from_uri(f"sqlite:///{path}")
db_chain = SQLDatabaseChain(llm=cllm, database=db, verbose=True)
```

API链

```py
from langchain.chains.api.base import APIChain
from langchain.chains.api import open_meteo_docs

api_chain = APIChain.from_llm_and_api_docs(
    cllm,
    open_meteo_docs.OPEN_METEO_DOCS,
    verbose=True,
    limit_to_domains=["https://api.open-meteo.com/"],
)
api_chain.invoke("上海当前气温是？")
```

另一例

```py
API_DOCS_BAIDU_FANYI = """BASE URL: https://fanyi.baidu.com/

API Documentation
/sug 接受一个英文或中文单词，返回一个json格式的相关词汇的英译中或中译英翻译，如下是所有的url参数：

Parameter	Format	Required	Default	Description
kw	String	Yes		要翻译的词汇
"""


api_chain = APIChain.from_llm_and_api_docs(
    cllm,
    API_DOCS_BAIDU_FANYI,
    verbose=True,
    limit_to_domains=["https://fanyi.baidu.com/"],
)
api_chain.invoke("'TEST'是什么的缩写？")
```

### 嵌入查询

读取文档和分词

```py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader('./国产剧.txt', encoding='utf8')
doc = loader.load()
print(f'字数{len(doc[0].page_content)}')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
docs = text_splitter.split_documents(doc)
```

嵌入模型、装到向量库、检索链

```py
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

embeddings = HuggingFaceBgeEmbeddings(model_name='./bge-large-zh-v1.5/')
base = FAISS.from_documents(docs, embeddings)
retriever= base.as_retriever(search_kwargs={'k': 5})
qa = RetrievalQA.from_chain_type(llm=cllm, chain_type='stuff', retriever=retriever, return_source_documents=True)
```

带记忆的对话

```py
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(cllm, retriever=retriever, memory=memory)

print(qa.invoke('2月5日有哪部电视剧杀青了？')['answer'])
print(qa.invoke('导演是谁？')['answer'])
```

下面进行批量的提问和判断是否正确，也就是评估

```py
examples = [
    {'question': '判断下面关于原文内容的理解和分析是否正确：《中国目录学史》的各篇“采多样之体例”，好处是尊重史事，缺点是强立名义。',
     'answer' : '错误，“《中国目录学史》......缺点是强立名义”错误，根据原文“因为在他看来，中国目录学虽然源远流长，但......硬要划分时期，区别特点，‘强立名义，反觉辞费’”可见，并不是《中国目录学史》强立名义，而是中国一直以来的目录学有“强立名义”的嫌疑，《中国目录学》是跳出了通常的中国目录学方法创新而作的“主题分述法”，根据主题选用合适体制而不强求一律。”' },

    {'question': '根据原文内容判断论述是否正确：与主题分述法相比，使用断代法来写中国目录学史，更能接近历史的本来面貌。',
     'answer' : '错误，原文只说“中国目录学史也未尝不可用‘断代法”来编写（吕绍虞《中国目录学史稿》即用分期断代法论述”，但并没有证据表明其比主题分述法“更能接近历史的本来面貌”。' },

]

qa = RetrievalQA.from_chain_type(llm=cllm, chain_type='stuff', retriever=base.as_retriever(), input_key='question')
predictions = qa.batch(examples)
predictions
```

```
[{'question': '判断下面关于原文内容的理解和分析是否正确：《中国目录学史》的各篇“采多样之体例”，好处是尊重史事，缺点是强立名义。',
  'answer': '错误，“《中国目录学史》......缺点是强立名义”错误，根据原文“因为在他看来，中国目录学虽然源远流长，但......硬要划分时期，区别特点，‘强立名义，反觉辞费’”可见，并不是《中国目录学史》强立名义，而是中国一直以来的目录学有“强立名义”的嫌疑，《中国目录学》是跳出了通常的中国目录学方法创新而作的“主题分述法”，根据主题选用合适体制而不强求一律。”',
  'result': '部分正确。原文中提到，《中国目录学史》的各篇采用适宜各自主题的体制，而不强求一律，这是为了尊重史事，使其源流毕具，一览无余。但文章并没有明确提到“强立名义”这个缺点，只是说如果按照时代顺序来划分各篇，可能会使读者迷乱。因此，该观点并不准确。'},
 {'question': '根据原文内容判断论述是否正确：与主题分述法相比，使用断代法来写中国目录学史，更能接近历史的本来面貌。',
  'answer': '错误，原文只说“中国目录学史也未尝不可用‘断代法”来编写（吕绍虞《中国目录学史稿》即用分期断代法论述”，但并没有证据表明其比主题分述法“更能接近历史的本来面貌”。',
  'result': '原文中并没有明确表示使用断代法来写中国目录学史更能接近历史的本来面貌。作者提到了中国目录学史可以使用“断代法”来编写，但并没有对此提出正面的或负面的评价。因此，论述不正确。'}]
```

进行评估，用的是examples的question和answer，以及predictions的result：

```py
from langchain.evaluation.qa import QAEvalChain

eval_chain = QAEvalChain.from_llm(cllm)
eval_chain.evaluate(examples, predictions, question_key='question', answer_key='answer', prediction_key='result')
```

这是个reranker的实现，不知道标不标准。

```py
from __future__ import annotations
from typing import Optional, Sequence
from langchain.schema import Document
from langchain.pydantic_v1 import Extra

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from sentence_transformers import CrossEncoder


class BgeRerank(BaseDocumentCompressor):
    model_name:str = './bge-reranker-v2-m3/'
    """Model name to use for reranking."""
    top_n: int = 4
    """Number of documents to return."""
    model:CrossEncoder = CrossEncoder(model_name)
    """CrossEncoder instance to use for reranking."""

    def bge_rerank(self,query,docs):
        model_inputs =  [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results

from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever

compressor = BgeRerank()
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
kc = RetrievalQA.from_llm(llm=cllm, retriever=compression_retriever, return_source_documents=True)
```

### 记忆

```py
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='chat_history'),
    HumanMessagePromptTemplate.from_template('{question}')
])

memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=True)

llm_chain = LLMChain(llm=cllm, memory=memory, prompt=prompt)

print(llm_chain.predict(question='苹果是什么颜色的？'))
print(llm_chain.predict(question='主要产地在哪里？'))
```

```py
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")
])

chain = prompt | cllm | output_parser

history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

print(chain_with_history.invoke({'question': '什么是图计算？'},
                                config={"configurable": {"session_id": None}}))
print(chain_with_history.invoke({'question': '刚才我问了什么问题？'},
                                config={"configurable": {"session_id": None}}))
```

### langchain表达式

重载了`__or__`来构建

```py
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("{question}")
model = AzureChatOpenAI(deployment_name=DEPLOYMENT_NAME, api_version="2024-02-15-preview")
output_parser = StrOutputParser()
chain = {"question": RunnablePassthrough()} | prompt | model | output_parser
```

可以可视化链

```py
chain.get_graph().print_ascii()
```

利用langsmith来监视

```py
import os
from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
```

测试一条链：

```py
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 输出解析器
output_parser = StrOutputParser()

# Prompt
topic_prompt = ChatPromptTemplate.from_template("生成一种'{input}'的名称，只回复一种{input}的名称")
good_prompt = ChatPromptTemplate.from_template("列举{topic}的好处:")
bad_prompt = ChatPromptTemplate.from_template("列举{topic}可能有的坏处:")
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("ai", "{topic}"),
        ("human", "好处:\n{good}\n\n坏处:\n{bad}"),
        ("system", "总结前文"),
    ]
)

# 链
topic_chain = topic_prompt | model | output_parser | {"topic": RunnablePassthrough()}
goods_chain = good_prompt | model | output_parser
bads_chain = bad_prompt | model | output_parser
summary_chain = summary_prompt | model | output_parser
chain = (
    topic_chain
    | {
        "good": goods_chain,
        "bad": bads_chain,
        "topic": itemgetter("topic"),
    }
    | summary_chain
)

# 调用
chain.invoke({"input": '水果'})
```

本质上，组件实现了`Runnable`，才能被串在一起，通用的方法有：
- `invoke`
- `stream`
- `batch`
以及相应的异步版本。

```py
prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)

chain = prompt | llm | StrOutputParser()

chain.batch([{"topic": "bears"}, {"topic": "frogs"}])

for t in chain.stream({"topic": "bears"}):
    print(t)
```


对不同的组件来说，其输入类型和输出类型是不一样的

|组件|输入类型|输出类型|
|-|-|-|
|`Prompt`|字典|提示词|
|`Retriever`|字符串|文档列表|
|`LLM`|字符串、消息列表或提示词|字符串|
|`ChatModel`|字符串、消息列表或提示词|ChatMessage|
|`Tool`|字符串/字典|取决于具体工具|
|`OutputParser`|LLM或ChatModel的输出|取决于具体解析器|

LCEL的好处：
- 立即支持异步、批量、流式
- Fallbacks
- 并行处理
- 内置日志

### 函数和工具

#### 自带的工具

使用谷歌搜索

```py
from langchain.agents import load_tools
from langchain.agents import initialize_agent

# 也有其他类型的工具
tools = load_tools(['google-search'], google_api_key=google_api_key, google_cse_id=google_cse_id)
agent = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True, return_intermediate_steps=True)
response = agent({'input': '特朗普称自己是“关税的信徒”是怎么回事'})
```

使用arxiv搜索

```py
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools

tools = load_tools(
    ["arxiv"],
)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(cllm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "What's the paper 1605.08386 about?",
    }
)
```

查看工具的详细信息

```py
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tool.run({"query": "langchain"})
```

```py
tool.name  # 'Wikipedia'
tool.description  # 'A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.'
tool.args  # {'query': {'title': 'Query', 'type': 'string'}}
tool.return_direct  # False
```

#### 函数调用

下面的内容是在旧版本的langchain（0.0.312）和DeepLearning.AI提供的OpenAI服务上运行的，我稍后再自己给Azure和ChatGLM做适配。

让LLM学会调用函数，首先要知道怎么把函数信息告知LLM。使用`pydanic`来描述这样的结构信息：

```py
# 一个函数
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# 告知LLM所应有的格式
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

# 使用pydanic类
from pydantic import BaseModel, Field

class pUser(BaseModel):
    name: str
    age: int
    email: str

# pydanic类可以嵌套
from typing import List
class Class(BaseModel):
    students: List[pUser]

# 一个描述函数的pydantic类 以前要求必须写文档 现在好像不用了
class WeatherSearch(BaseModel):
    """传入机场编码，调用本函数后返回机场所在地的天气"""
    airport_code: str = Field(description="要查询天气的机场编码")

# 转换为openai_function
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
weather_function = convert_pydantic_to_openai_function(WeatherSearch)
weather_function

'''
{'name': 'WeatherSearch',
 'description': 'Call this with an airport code to get the weather at that airport',
 'parameters': {'title': 'WeatherSearch',
  'description': 'Call this with an airport code to get the weather at that airport',
  'type': 'object',
  'properties': {'airport_code': {'title': 'Airport Code',
    'description': 'airport code to get weather for',
    'type': 'string'}},
  'required': ['airport_code']}}
'''

# 字段可以不写描述
class WeatherSearch2(BaseModel):
    """传入机场编码，调用本函数后返回机场所在地的天气"""
    airport_code: str
```

首先`convert_pydantic_to_openai_function`已经弃用了，其次我这里调用的结果也和教程格式不一样：

```py
from langchain_core.utils.function_calling import convert_to_openai_function

class WeatherSearch(BaseModel):
    """传入机场编码，调用本函数后返回机场所在地的天气"""
    airport_code: str = Field(description="要查询天气的机场编码")

weather_function = convert_to_openai_function(WeatherSearch)
weather_function
'''
{'name': 'WeatherSearch',
 'description': '传入机场编码，调用本函数后返回机场所在地的天气',
 'parameters': {'type': 'object',
  'properties': {'airport_code': {'description': '要查询天气的机场编码',
    'type': 'string'}},
  'required': ['airport_code']}}
'''

def get_temperature(city: str) -> int:
    """获取指定城市的当前气温"""
    return 20
convert_to_openai_function(get_temperature)
'''
{'name': 'get_temperature',
 'description': '获取指定城市的当前气温',
 'parameters': {'type': 'object',
  'properties': {'city': {'type': 'string'}},
  'required': ['city']}}
'''

from langchain_core.tools import tool
@tool
def get_temperature(city: str) -> int:
    """获取指定城市的当前气温"""
    return 20
convert_to_openai_function(get_temperature)

{'name': 'get_temperature',
 'description': 'get_temperature(city: str) -> int - 获取指定城市的当前气温',
 'parameters': {'type': 'object',
  'properties': {'city': {'type': 'string'}},
  'required': ['city']}}
```

将这个json传给LLM，可以让LLM知道是否要调用这个函数

```py
model_with_function = model.bind(functions=[weather_function])
model_with_function.invoke("what is the weather in sf?")

# 强制调用
model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})
```

旧版本（0.0.312）的返回值相当简洁，现在（0.1.5）多了很多信息，但是也能看到其中要调用的函数名和参数名

```
AIMessage(content='',
additional_kwargs={
    'function_call': {
        'name': 'WeatherSearch',
        'arguments': '{"airport_code":"SFO"}'}},
response_metadata={
    'token_usage': <OpenAIObject at 0x7f94b05d4c20> JSON: {
        "prompt_tokens": 70,
        "completion_tokens": 17,
        "total_tokens": 87},
    'model_name': 'gpt-3.5-turbo',
    'system_fingerprint': 'fp_b28b39ffa8',
    'finish_reason': 'function_call',
    'logprobs': None},
id='run-d8bd80d9-3aba-4344-859c-65c10061a8ef-0')
```

Azure说他们已经弃用了`functions`，现在要传入的参数是`tools`和`tool_choice`。传入的格式也有些不同，我找不到资料，但是强凑了一个出来（在`OpenChatAI`上）：

```py
weather_tool = {"type": "function",
                "function":  convert_pydantic_to_openai_function(WeatherSearch),}

model_with_function = model.bind(tools=[weather_tool])
model_with_function.invoke("what is the weather in sf?")
```

```
AIMessage(content='', additional_kwargs={'tool_calls': [<OpenAIObject id=call_6bwQedmrEtU8yz7RGWlia1ZG at 0x7f94b06d76d0> JSON: {
  "id": "call_6bwQedmrEtU8yz7RGWlia1ZG",
  "type": "function",
  "function": {
    "name": "WeatherSearch",
    "arguments": "{\"airport_code\":\"SFO\"}"
  }
}]}, response_metadata={'token_usage': <OpenAIObject at 0x7f94b06d70e0> JSON: {
  "prompt_tokens": 69,
  "completion_tokens": 17,
  "total_tokens": 86
}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-8e446556-f225-40ac-95d0-49206fce67a5-0')
```

但是在`AzureChatAI`上失败了。返回的东西完全不能用，尽管Azure的官网说可以。

总之，这样的类可以给信息提取和打标签做准备，例如：

```py
class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")

model_with_functions = model.bind(
    functions=[convert_pydantic_to_openai_function(Tagging)],
    function_call={"name": "Tagging"}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])
tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

tagging_chain.invoke({"input": "non mi piace questo cibo"})
# {'sentiment': 'neg', 'language': 'it'}
```

可选参数

```py
from typing import Optional
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")

class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

# 略去一些内容

extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
# [{'name': 'Joe', 'age': 30}, {'name': 'Martha'}]
```


#### 工具调用

工具 = 函数 + 函数调用

大模型会根据prompts，选择合适的工具，这一过程叫做路由。langchain现在有很多现成的工具，如数学、搜索、SQL等。可以设计自己的工具。

tool装饰器的使用
```py
from langchain.agents import tool

@tool
def get_word_length(word: str) -> int:
    """返回单词的长度，也就是字母个数"""
    return len(word)

get_word_length.name  # get_word_length
get_word_length.description  # get_word_length(word: str) -> int - 返回单词的长度，也就是字母个数
get_word_length.args  # {'word': {'title': 'Word', 'type': 'string'}}
```

结合pydanic，指定args_schema

```py
from pydantic import BaseModel, Field
class GetWordLengthInput(BaseModel):
    word: str = Field(description="要统计长度的单词")

@tool(args_schema=GetWordLengthInput)
def get_word_length(word: str) -> int:
    """返回单词的长度，也就是字母个数"""
    return len(word)

get_word_length.args
# {'word': {'title': 'Word', 'description': '要统计长度的单词', 'type': 'string'}}
```

工具调用输出解析：

```py
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
chain = prompt | model | OpenAIFunctionsAgentOutputParser()
result = chain.invoke({"input": "what is the weather in sf right now"})
type(result)  # langchain.schema.agent.AgentActionMessageLog
result.tool  # 'get_current_temperature'
result.tool_input  # {'latitude': 37.7749, 'longitude': -122.4194}
```

路由

```py
from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)

chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route
```

这两个无论新旧版本都是适配的。

这是OpenAI可以用的会话机器人：

```py
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)
agent_executor.invoke({"input": "whats the weather in sf?"})
```

#### BaseTool

我模仿了langchaian内置工具的调用构造了自己的工具，它可以使用CoT来处理，这样应该即使是较差的大模型也能处理函数调用。

```py
from langchain_core.tools import BaseTool

class FactorizationTool(BaseTool):
    name: str = 'factorization'
    description:str = '一个用于大于1的正整数进行质因数分解的工具，传入一个大于1的整数，返回它的质因数组成的列表。'
    def _run(self, n):
        return prime_factors(n)

class CountLetterTool(BaseTool):
    name: str = 'count_letter'
    description:str = '一个用于对单词内字母进行计数的工具，传入一个单词，返回它的长度。'
    def _run(self, word):
        return count_letter(word)
```

用[hub](https://smith.langchain.com/hub/)获取提示词，并

```py
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

tools = [FactorizationTool(), CountLetterTool()]
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

agent_executor.invoke({"input": "数一下pliuahdifdhjcbhasjkh有多少个字母"})
```

这是使用到的[提示词](https://smith.langchain.com/hub/hwchase17/react)：

```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

记忆功能和hub还在看。

## LTP

[文档](https://ltp.readthedocs.io/zh-cn/stable/)


