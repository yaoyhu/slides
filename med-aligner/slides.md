---
theme: Seriph
background: https://images.ctfassets.net/kftzwdyauwt9/7f37b3a2-c7fa-4e5d-85517bd29a33/cc8ca2aa652e9edc6e724977f64b2c64/SuperAlignmentBlog_Artwork_Transparent.png?w=3840&q=90&fm=webp
# some information about your slides (markdown enabled)
title: Aligner
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
  # basically the text
  sans: PingFang SC
  # use with `font-serif` css class from UnoCSS
  serif: PingFang SC
  # for code blocks, inline code, etc.
  mono: SF Mono
---

# Alignment for LLMs
by yaoyhu
<!-- Thanks to [Jiaming Ji](https://github.com/zmsn-2077) -->

<!-- <div @click="$slidev.nav.next" class="mt-12 py-1" hover:bg="white op-10">
  Press Space for next page <carbon:arrow-right />
</div> -->

<div class="abs-br m-6 text-xl">
  <a href="https://github.com/yaoyhu/slides" target="_blank" class="slidev-icon-btn">
    <carbon:logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
---

# Table of contents

1. MedAligner
2. AI Alignment
   - Pre-training and Post-training
   - Reinforcement Learning with Human Feedback (RLHF)
3. Aligner
   - Correction is easier than generation
   - The training process of Aligner
   - Experiment Results
   - Weak-to-Strong Generalization via Aligner
4. Building a Large Language Model (LLM)
   - RAG
   - Fine-tuning
   - Alignment

---
transition: fade-out
---

# MedAligner

[Med-Aligner](https://www.sciencedirect.com/science/article/pii/S266667582500205X) empowers LLM medical applications for complex medical scenarios

1. Reliability
   - limited high-quality data
   - closed-source model rigidity
   - reasoning degradation during fine-tuning
2. Achievements
   - plug-and-play, even for closed-source models
   - without requiring full re-optimization
   - significant enhancements across all 3H dimensions—helpfulness, harmlessness, and honesty
3. No technical details (even wrong huggingface link)

<!--
You can have `style` tag in markdown to override the style for the current page.
Learn more: https://sli.dev/features/slide-scope-style
-->

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

<!--
Here is another comment.
-->

---
transition: slide-up
level: 2
---

# AI Alignment

How can we build AI systems that behave in line with human intentions and values?

1. Training process of LLMs:
   - **Pre-training**: Utilize large-scale text data to train a model for general capabilities through an autoregressive approach.
   - **Post-training**: Align the pre-trained model with specific tasks using instruction fine-tuning and **reinforcement learning with human feedback** (RLHF).

<!--
通用扩写能力：给定一个上文，自回归下文扩写
-->

2. OpenAI demonstrated that RLHF enabled a smaller 1.3B parameter model to outperform a much larger 175B model.

---
layout: image-right
level: 2
image: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png
backgroundSize: 30em 90%
---

# Reinforcement Learning with Human Feedback ([RLHF](https://huggingface.co/blog/zh/rlhf))

1. Requires access to model parameters
2. Optimization redundancy: target model, reward model, critic model...
3. Full parameter tuning is challenging
4. Reward models have poor generalization
5. Alignment objectives are hard to define

<!--
1. Step 1. 预训练语言模型训：具备问答能力
2. Step 2. 训练奖励模型：通过人类反馈进行优化
3. Step 3. 用强化学习微调：PPO
-->

---
layout: image-right
image: https://pku-aligner.github.io/figures/aligner_figure1.jpg
backgroundSize: 25em 50%
---

# Aligner: Efficient Alignment by Learning to Correct

1. Correction is easier than generation 
2. A lightweight model to correct the target model's response
   - applicable across different base models
   - Completely bypassing RLHF, Aligner requires only a single line of code modification from SFT
   - For a 70B-parameter model, using Aligner saves 22.5 times the resources compared to RLHF.

<!-- 2.  -->


---
level: 2
---

# The training process of Aligner

1. SFT: high-quality instruction dataset: $\{x^{(i)}, y^{(i)}; i = 1, \dots, n\}$
$$
\min_{\theta} \mathcal{L}(\theta; \mathcal{D}_{\text{sft}}) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{sft}}} [ \log \pi_{\theta}(y|x) ]
$$
2. Aligner: $\{x^{(i)}, y_o^{(i)}, y_c^{(i)}; i = 1, \dots, n\}$
   - Copy: directly output the response from the upstream model
   - Correction:residual correct the response from the upstream model.
$$
\min_{\phi} \mathcal{L}_{\text{aligner}}(\phi; \mathcal{M}) = -\mathbb{E}_{\mathcal{M}}[ \log \mu_{\phi}(y_c \mid y_0, x) ]
$$

<!--
1. 让模型预测的答案 y 与数据集中的标准答案越接近越好
-->

---
layout: image-right
image: https://pku-aligner.github.io/figures/w2s_performance.png
backgroundSize: 30em 55%
level: 2
---

# Experiment Results
1. Improvements:
   - Improved Helpfulness
   - Enhanced Harmlessness
   - Reduced Hallucinations
2. Aligner exhibits a Scale Law trend as the model parameters increase
3. ~~bigger is better~~: e.g. the improvement from 2B to 13B is significant, but the gain from 13B to 70B is relatively limited.

---
layout: image
image: https://pku-aligner.github.io/figures/w2s_illustration.png
# backgroundSize: 30em 50%
backgroundSize: contain
---

# [Weak-to-Strong Generalization](https://openai.com/index/weak-to-strong-generalization/) via Aligner

<!-- > If I have seen further it is by standing on the shoulders of giants. (Isaac Newton) -->

---
layout: center
# class: text-center
---

# Building a Large Language Model

1. RAG: providing specific information for the model to draw from when answering questions
2. Fine-tuning: training the model on specific tasks
3. Alignment: ensuring the model's behavior aligns with human values and intentions


<!-- <PoweredBySlidev mt-10 /> -->
