---
theme: seriph
background: ./image.png
# some information about your slides (markdown enabled)
title: LLMs in Remote Sensing
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# open graph
seoMeta:
  # By default, Slidev will use ./og-image.png if it exists,
  # or generate one from the first slide if not found.
  ogImage: auto
  # ogImage: https://cover.sli.dev
fonts:
  sans: PingFang SC
  serif: PingFang SC
  mono: SF Mono
---

# LLMs in Remote Sensing
by [yaoyhu](https://github.com/yaoyhu)


<div class="abs-br m-6 text-xl">
  <a href="https://github.com/yaoyhu/slides/tree/main/llm-rs" target="_blank" class="slidev-icon-btn">
    <carbon:logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
layout: image-right
---

# Overview
1. 🧑‍💻 Implementation
   - Retrieval-Augmented Generation
   - Fine-tuning a Pre-trained Model
2. 💬 Applications
   - Image Understanding
   - Cross-Modal Applications
3. 😵 Unfinished tasks...
4. 📚 TODOs

<!--
这次我要汇报的内容分三点：

1. 我动手实现了简单的 RAG 和 微调流程，上次我先把微调排除了，这次我又把它搬回来了，后面会介绍原因。

2. 第二点更偏向于讨论，就是做一个什么样的 LLM？

3. 最后是和师姐请教一下，上次那篇论文我的复现进展很慢。
-->

---
transition: fade-out
layout: image-right
image: https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/diagram_4_mermaid--1446345905-light-mermaid.svg
backgroundSize: 30em 90%
---

# Retrieval-Augmented Generation

Tutorial: [Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/make-your-own-rag)

1. **Dense Retrieval**: better embedding model + query rewriting
2. **Indexing**: [FAISS](https://github.com/facebookresearch/faiss) + compression strategies
3. **Ranking**: Precision@k evaluation + user feedback
4. **Prompt**: RAG-specific optimization + answer-aware retrieval

<!--
首先是 RAG，以前一直把它当作一个很简单的外部数据库。

很久之前的一次组会我用 LLaMAindex 库提供的 API 几十行代码就实现了一个很简单的 RAG。

最近在知乎看到一个提问：RAG 会不会消亡？
- 有个人从面试的角度：“我在这个项目中使用LangChain搭建RAG的链路，”
- 看起来是做了一点事，但主要工作都在库里而不是自己做了什么。所以我就找了一些资源，写了一个简化的全流程RAG，体会一下整个 RAG 的过程。
-->

---

# Fine-Tuning a Pre-trained Model

1. Why Fine-Tuning?
   - [SpectralGPT](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT): The **pretraining** experiments were run on 8 NVIDIA RTX 4090 GPUs.
   - Finetune Dataset: EuroSAT & OSCD & SegMunich
   - ...
2. [Seven Stage Fine-Tuning Pipeline for LLM](https://arxiv.org/html/2408.13296v1#Ch2)
   - Data Preparation 
   - Model Initialization
   - Training Setup
   - Fine-tuning
   - Validation & Evaluation
   - Deployment
   - Monitoring

<!--
我在调研的过程中发现：微调并没有那么麻烦。
-->

---
transition: slide-up
layout: image-right
image: https://github.com/ZhanYang-nwpu/Awesome-Remote-Sensing-Multimodal-Large-Language-Model/raw/main/images/1-timeline.jpg
backgroundSize: 30em 30%
---

# Current Applications

Reference: [Awesome-Remote-Sensing-Multimodal-Large-Language-Model](https://github.com/ZhanYang-nwpu/Awesome-Remote-Sensing-Multimodal-Large-Language-Model)

1. Image Understanding
   - Scene Description
   - Object/Change  Detection
2. Cross-Modal Applications
   - Text-to-Image Retrieval
   - Image-to-Text Generation
   - Visual Question Answering
3. ~~Chatbot~~: [GPT-5 not only matches but surpasses the performance of pre-licensed medical professionals in controlled QA/VQA evaluations, which raises both potential benefits and caution.](https://arxiv.org/pdf/2508.08224)

<!--
目前LLM在遥感方面的应
-->

---

# 🤯 预处理：时间对齐

1. EMI 的实际观测时间？
   - 文件名 `GF5B_EMI_20220601_003884_L10000140424_VI1.h5`
   - 文件属性
   - 内部数据
   - LV1 数据读取
2. 固定过境时间 + 经度调整后的时区
3. GAIA = MAE + DINO


---
layout: center
class: text-center
---

# TODOs

[LoRA](https://arxiv.org/abs/2106.09685) & [Q-LoRA](https://arxiv.org/abs/2305.14314)
