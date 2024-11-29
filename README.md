# A Survey on Data Synthesis and Augmentation for Large Language Models

Work In Progress.


## Table of Contents
* [Types of Data Synthesis and Augmentation](#types-of-data-synthesis-and-augmentation)
* [Applications](#applications)
  * [Data preparation](#data-preparation)
  * [Pretraining](#pretraining)
  * [Prompt](#prompt)
  * [Fine-Tuning](#fine-tuning)
  * [Alignment with Human Preferences](#alignment-with-human-preferences)
  * [Knowledge Base](#knowledge-base)
  * [RAG and Other Tools](#rag-and-other-tools)
  * [Evaluation](#evaluation)
  * [Optimization and Deployment](#optimization-and-deployment)
  * [Applications](#applications-1)
  * [Agent](#agent)
* [Dataset](#dataset)
  * [Dataset Examples](#dataset-examples)

# Types of Data Synthesis and Augmentation
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[example paper](https://arxiv.org/abs/...)|arxuv 2023|-|

# Applications
## Data preparation
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
| [Tinystories: How small can language models be and still speak coherent english?](https://arxiv.org/abs/2305.07759) |      arxiv 2023       |    https://huggingface.co/roneneldan    |
| [Controllable dialogue simulation with in-context learning](https://arxiv.org/abs/2210.04185) |      arxiv 2022       |    https://github.com/Leezekun/dialogic    |
| [Genie: Achieving human parity in content-grounded datasets generation](https://arxiv.org/abs/2401.14367) |      arxiv 2024       |    -    |
| [Case2Code: Learning Inductive Reasoning with Synthetic Data](https://arxiv.org/abs/2407.12504) |      arxiv 2024       |    https://github.com/choosewhatulike/case2code    |
| [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120) |      41 ICML       |    https://github.com/ise-uiuc/magicoder    |
| [ Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) |      arxiv 2023       |    https://arxiv.org/abs/2212.10560    |
| [Wizardlm: Empowering large language models to follow complex instructions](https://arxiv.org/abs/2304.12244) |      arxiv 2023       |    https://github.com/nlpxucan/WizardLM    |
| [Augmenting Math Word Problems via Iterative Question Composing](https://arxiv.org/abs/2401.09003) |      arxiv 2024       |    https://huggingface.co/ datasets/Vivacem/MMIQC    |
| [Common 7b language models already possess strong math capabilities](https://arxiv.org/abs/2403.04706) |      arxiv 2024       |    https://github.com/Xwin-LM/Xwin-LM    |
| [Mammoth: Building math generalist models through hybrid instruction tuning](https://arxiv.org/abs/2309.05653) |      arxiv 2023       |    https://tiger-ai-lab.github.io/MAmmoTH/    |
| [Enhancing chat language models by scaling high-quality instructional conversations](https://arxiv.org/abs/2305.14233) |      arxiv 2024       |    https://github.com/thunlp/UltraChat    |
| [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) |      arxiv 2024       |    https://magpie-align.github.io/    |
| [GenQA: Generating Millions of Instructions from a Handful of Prompts](https://arxiv.org/abs/2406.10323) |      arxiv 2024       |    https://huggingface.co/datasets/tomg-group-umd/GenQA    |
| [Sharegpt4v: Improving large multi-modal models with better captions](https://arxiv.org/abs/2311.12793) |      arxiv 2023       |    https://sharegpt4v.github.io/    |
| [What makes for good visual instructions? synthesizing complex visual reasoning instructions for visual instruction tuning](https://arxiv.org/abs/2311.01487) |      arxiv 2023       |    https://github.com/RUCAIBox/ComVint    |
| [Stablellava: Enhanced visual instruction tuning with synthesized image-dialogue data](https://arxiv.org/abs/2308.10253) |      arxiv 2023       |    https://github.com/icoz69/StableLLAVA    |
| [Anygpt: Unified multimodal llm with discrete sequence modeling](https://arxiv.org/abs/2402.12226) |      arxiv 2024       |    https://junzhan2000.github.io/AnyGPT.github.io/    |
| [ Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model](https://arxiv.org/abs/2407.07053) |      arxiv 2024       |    https://github.com/zwq2018/Multi-modal-Self-instruct    |
| [Chartllama: A multimodal llm for chart understanding and generation](https://arxiv.org/abs/2311.16483) |      arxiv 2023       |    https://tingxueronghua.github.io/ChartLlama/    |
| [Genixer: Empowering Multimodal Large Language Models as a Powerful Data Generator](https://arxiv.org/abs/2312.06731) |      arxiv 2023       |    https://github.com/zhaohengyuan1/Genixer    |
| [Open-Source LLMs for Text Annotation: A Practical Guide for Model Setting and Fine-Tuning](https://arxiv.org/abs/2307.02179) |      arxiv 2024       |    https://osf.io/ctgqx/    |
| [ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks](https://www.pnas.org/doi/10.1073/pnas.2305016120) |      NAS 2023       |    -    |
| [Can Large Language Models Aid in Annotating Speech Emotional Data? Uncovering New Frontiers](https://arxiv.org/abs/2307.06090) |      arxiv 2023       |    -    |
| [Can ChatGPT Reproduce Human-Generated Labels? A Study of Social Computing Tasks](https://arxiv.org/abs/2304.10145) |      arxiv 2023       |    -    |
| [Chatgpt-4 outperforms experts and crowd workers in annotating political twitter messages with zero-shot learning](https://arxiv.org/abs/2304.06588) |      arxiv 2023       |    -    |
| [Unraveling chatgpt: A critical analysis of ai-generated goal-oriented dialogues and annotations](https://arxiv.org/abs/2305.14556) |      ICIAAI       |    -    |
| [FullAnno: A Data Engine for Enhancing Image Comprehension of MLLMs](https://arxiv.org/abs/2409.13540) |      arxiv 2024       |    https://arcana-project-page.github.io/    |
| [DISCO: Distilling counterfactuals with large language models](https://arxiv.org/abs/2212.10534) |      arxiv 2023       |    https://github.com/eric11eca/disco    |
| [Tinygsm: achieving> 80% on gsm8k with small language models](https://arxiv.org/abs/2312.09241) |      arxiv 2023       |    -    |
| [Gpt3mix: Leveraging large-scale language models for text augmentation](https://arxiv.org/abs/2104.08826) |      arxiv 2021       |    https://github.com/naver-ai/hypermix    |
| [CORE: A retrieve-then-edit framework for counterfactual data generation](https://arxiv.org/abs/2210.04873) |      arxiv 2022       |    https://github.com/tanay2001/CORE    |
| [Diversify your vision datasets with automatic diffusion-based augmentation](https://arxiv.org/abs/2305.16289) |      arxiv 2023       |    https://github.com/lisadunlap/ALIA    |
| [Closing the loop: Testing chatgpt to generate model explanations to improve human labelling of sponsored content on social media](https://arxiv.org/abs/2306.05115) |      arxiv 2023       |    https://github.com/thalesbertaglia/chatgpt-explanations-sponsored-content/    |
| [Toolcoder: Teach code generation models to use api search tools](https://arxiv.org/abs/2305.04032) |      arxiv 2023       |    -    |
| [Coannotating: Uncertainty-guided work allocation between human and large language models for data annotation](https://arxiv.org/abs/2310.15638) |      arxiv 2023       |    https://github.com/SALT-NLP/CoAnnotating    |
| [Does Collaborative Human-LM Dialogue Generation Help Information Extraction from Human Dialogues?](https://arxiv.org/abs/2307.07047) |      arxiv 2023       |    https://boru-roylu.github.io/DialGen    |
| [Measuring mathematical problem solving with the math dataset](https://arxiv.org/abs/2103.03874) |      arxiv 2021       |    -    |
| [Llemma: An open language model for mathematics](https://arxiv.org/abs/2310.10631) |      arxiv 2023       |    https://github.com/EleutherAI/math-lm    |
| [Code Less, Align More: Efficient LLM Fine-tuning for Code Generation with Data Pruning](https://arxiv.org/abs/2407.05040) |      arxiv 2024       |    -    |

## Pretraining
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
| [VILA2: VILA Augmented VILA](https://arxiv.org/abs/2407.17453)|arxiv 2024|https://github.com/NVlabs/VILA|
| [Textbooks are all you need](https://arxiv.org/abs/2306.11644)|arxiv 2023|-|
| [Textbooks are all you need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)|arxiv 2023|-|
| [Is Child-Directed Speech Effective Training Data for Language Models](https://arxiv.org/abs/2408.03617) |      arxiv 2024       |    https://babylm.github.io/index.html    |
| [SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding](https://arxiv.org/abs/2408.15545) |      arxiv 2024       |    https://github.com/dptech-corp/Uni-SMART    |
| [Anygpt: Unified multimodal llm with discrete sequence modeling](https://arxiv.org/abs/2402.12226) |      arxiv 2024       |    https://junzhan2000.github.io/AnyGPT.github.io/    |
| [Is synthetic data from generative models ready for image recognition](https://arxiv.org/abs/2210.07574) |      arxiv 2023       |    https://github.com/CVMI-Lab/SyntheticData    |
| [Rephrasing the web: A recipe for compute and data-efficient language modeling](https://arxiv.org/abs/2401.16380) |      arxiv 2024       |    -    |
| [Physics of language models: Part 3.1, knowledge storage and extraction](https://arxiv.org/abs/2309.14316) |      arxiv 2024       |    https://physics.allen-zhu.com/part-3-knowledge/part-3-1    |
| [Llemma: An open language model for mathematics](https://arxiv.org/abs/2310.10631) |      arxiv 2023       |    https://github.com/EleutherAI/math-lm    |
| [Enhancing multilingual language model with massive multilingual knowledge triples](https://arxiv.org/abs/2111.10962) |      arxiv 2021       |    https://github.com/ntunlp/kmlm.git    |
| [Large language models, physics-based modeling, experimental measurements: the trinity of data-scarce learning of polymer properties](https://arxiv.org/abs/2407.02770) |      arxiv 2024       |    -    |

## Fine-Tuning
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Large language models can self-improve](https://arxiv.org/abs/2210.11610)|arxiv 2022|-|
|[STaR: Bootstrapping reasoning with reasoning](https://arxiv.org/abs/2203.14465)|arxiv 2022|-|
|[Language models can teach themselves to program better](https://arxiv.org/abs/2207.14502)|arxiv 2022|https://github.com/microsoft/PythonProgrammingPuzzles|
|[Self-Instruct: Aligning language models with self-generated instructions](https://arxiv.org/abs/2212.10560)|arxiv 2023|https://github.com/yizhongw/self-instruct|
|[Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)|github 2023|https://github.com/tatsu-lab/stanford_alpaca|
|[Code alpaca: An instruction-following llama model for code generation](https://github.com/sahil280114/codealpaca)|github 2023|https://github.com/sahil280114/codealpaca|
|[Code Llama: Open foundation models for code](https://arxiv.org/abs/2308.12950)|arxiv 2023|https://github.com/meta-llama/codellama|
|[WizardLM: Empowering large language models to follow complex instructions](https://arxiv.org/abs/2304.12244)|arxiv 2023|https://github.com/nlpxucan/WizardLM|
|[WizardCode: Empowering code large language models with Evol-Instruct](https://arxiv.org/abs/2306.08568)|arxiv 2023|https://github.com/nlpxucan/WizardLM|
|[WizardMath: Empowering mathematical reasoning for large language models via reinforced evol-instruct](https://arxiv.org/abs/2308.09583)|arxiv 2023|https://github.com/nlpxucan/WizardLM|
|[Magicoder: Source code is all you need](https://arxiv.org/abs/2312.02120)|arxiv 2023|https://github.com/ise-uiuc/magicoder|
|[MetaMeth: Bootstap your own mathematical questions for large language models](https://arxiv.org/abs/2309.12284)|arxiv 2024|https://meta-math.github.io/|
|[DeepSeek-Prover: Advancing theorem proving in LLMs through large-scale synthetic data](https://arxiv.org/abs/2405.14333v1)|arxiv 2024|-|
|[Conmmon 7B language models already possess strong math capabilities](https://arxiv.org/abs/2403.04706)|arxiv 2024|https://github.com/Xwin-LM/Xwin-LM|
|[Augmenting math word problems via iterative question composing](https://arxiv.org/abs/2401.09003)|arxiv 2024|https://huggingface.co/datasets/Vivacem/MMIQC|
| []() |      arxiv 2024       |    -    |
| []() |      arxiv 2024       |    -    |
| []() |      arxiv 2024       |    -    |
| []() |      arxiv 2024       |    -    |
| []() |      arxiv 2024       |    -    |
| []() |      arxiv 2024       |    -    |



## Prompt
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|


## Alignment with Human Preferences
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|


## Knowledge Base
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|



## RAG and Other Tools
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|


## Evaluation
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|


## Optimization and Deployment
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|

## Applications

### Math

| Paper                                                        |     Published in      |                 Code/Project                  |
| ------------------------------------------------------------ | :-------------------: | :-------------------------------------------: |
| [Galactica: A Large Language Model for Science](http://arxiv.org/abs/2211.09085) |      arxiv 2022       |                       -                       |
| [STaR: Bootstrapping Reasoning With Reasoning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) |     NeurIPS 2022      |       https://github.com/ezelikman/STaR       |
| [Multilingual Mathematical Autoformalization](https://arxiv.org/abs/2311.03755) |      arxiv 2023       |                       -                       |
| [WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](http://arxiv.org/abs/2308.09583) |      arxiv 2023       |     https://github.com/nlpxucan/WizardLM      |
| [MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning](https://arxiv.org/abs/2309.05653) |      arxiv 2023       |                       -                       |
| [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](http://arxiv.org/abs/2309.12284) |      arxiv 2023       |         https://meta-math.github.io/          |
| [Synthetic Dialogue Dataset Generation using LLM Agents](https://arxiv.org/abs/2401.17461) |  EMNLP Workshop 2023  |                       -                       |
| [Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://openreview.net/forum?id=TPtXLihkny) | NeurIPS Workshop 2024 |                       -                       |
| [Synthetic Dialogue Dataset Generation using LLM Agents](http://arxiv.org/abs/2401.17461) |      arxiv 2024       |                       -                       |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) |      arxiv 2024       | https://github.com/deepseek-ai/DeepSeek-Math  |
| [DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://arxiv.org/abs/2405.14333) |      arxiv 2024       |                       -                       |
| [Augmenting Math Word Problems via Iterative Question Composing](http://arxiv.org/abs/2401.09003) |      arxiv 2024       | https://huggingface.co/datasets/Vivacem/MMIQC |



### Science

| Paper                                                        |           Published in           |                         Code/Project                         |
| ------------------------------------------------------------ | :------------------------------: | :----------------------------------------------------------: |
| [Galactica: A Large Language Model for Science](http://arxiv.org/abs/2211.09085) |            arxiv 2022            |                              -                               |
| [Reflection-Tuning: Recycling Data for Better Instruction-Tuning](https://openreview.net/forum?id=xaqoZZqkPU) | NeurIPS Workshop 2023 / ACL 2024 |       https://github.com/tianyi-lab/Reflection_Tuning        |
| [Reflexion: language agents with verbal reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) |           NeurIPS 2023           |          https://github.com/noahshinn024/reflexion           |
| [SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding](https://arxiv.org/abs/2408.15545) |      NeurIPS Workshop 2024       | https://github.com/dptech-corp/Uni-SMART/tree/main/SciLitLLM |
| [SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models](https://arxiv.org/abs/2401.07950) |           NeurIPS 2024           |               https://github.com/THUDM/SciGLM                |
| [ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852) |            arxiv 2024            |                              -                               |





### Code

| Paper                                                        |     Published in     |             Code/Project              |
| ------------------------------------------------------------ | :------------------: | :-----------------------------------: |
| [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html) |      NIPS 2022       | https://github.com/salesforce/CodeRL  |
| [Generating Programming Puzzles to Train Language Models](https://openreview.net/forum?id=H8cx0iO-y-9) | ICLR 2022 (Workshop) |                   -                   |
| [Language Models Can Teach Themselves to Program Better](http://arxiv.org/abs/2207.14502) |      ICLR 2023       |                   -                   |
| [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) |      Arxiv 2023      |                   -                   |
| [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463) |      Arxiv 2023      |                   -                   |
| [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) |      ICLR 2023       |                   -                   |
| [Learning Performance-Improving Code Edits](http://arxiv.org/abs/2302.07867) |      ICLR 2024       |         https://pie4perf.com/         |
| [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](http://arxiv.org/abs/2306.08568) |      ICLR 2024       | https://github.com/nlpxucan/WizardLM  |
| [Magicoder: Source Code Is All You Need](http://arxiv.org/abs/2312.02120) |      ICML 2024       | https://github.com/ise-uiuc/magicoder |



### Medical

| Paper                                                        | Published in  |                    Code/Project                     |
| ------------------------------------------------------------ | :-----------: | :-------------------------------------------------: |
| [MedDialog: Large-scale Medical Dialogue Datasets](https://aclanthology.org/2020.emnlp-main.743/) |  EMNLP 2020   | https://github.com/UCSDAI4H/Medical-Dialogue-System |
| [HuatuoGPT, towards Taming Language Model to Be a Doctor](https://arxiv.org/abs/2305.15075) |  EMNLP 2023   |  https://github.com/FreedomIntelligence/HuatuoGPT   |
| [HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs](https://arxiv.org/abs/2311.09774) |  arxiv 2023   | https://github.com/FreedomIntelligence/HuatuoGPT-II |
| [ChatCounselor: A Large Language Models for Mental Health Support](https://arxiv.org/abs/2309.15461) |  arxiv 2023   |                          -                          |
| [DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation](https://arxiv.org/abs/2308.14346) |  arxiv 2023   |      https://github.com/FudanDISC/DISC-MedLLM       |
| [Biomedical discovery through the integrative biomedical knowledge hub (iBKH)](https://www.cell.com/iscience/fulltext/S2589-0042(23)00537-0) | iScience 2023 |         https://github.com/wcm-wanglab/iBKH         |
| [Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models](https://arxiv.org/abs/2311.00287) |  arxiv 2023   |         https://github.com/ritaranx/ClinGen         |
| [ShenNong-TCM](https://github.com/michael-wzhu/ShenNong-TCM-LLM) |  Github repo  |  https://github.com/michael-wzhu/ShenNong-TCM-LLM   |
| [ZhongJing(仲景)](https://github.com/pariskang/CMLM-ZhongJing) |  Github repo  |     https://github.com/pariskang/CMLM-ZhongJing     |



### Law

| Paper                                                        | Published in |                   Code/Project                   |
| ------------------------------------------------------------ | :----------: | :----------------------------------------------: |
| [DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services](http://arxiv.org/abs/2309.11325) |  arxiv 2023  |     https://github.com/FudanDISC/DISC-LawLLM     |
| [Lawyer LLaMA Technical Report](http://arxiv.org/abs/2305.15062) |  arxiv 2023  |                        -                         |
| [LawGPT: A Chinese Legal Knowledge-Enhanced Large Language Model](http://arxiv.org/abs/2406.04614) |  arxiv 2024  |     https://github.com/pengxiao-song/LaWGPT      |
| [WisdomInterrogatory](https://github.com/zhihaiLLM/wisdomInterrogatory) | Github repo  | https://github.com/zhihaiLLM/wisdomInterrogatory |

### Education

| Paper                                                        |                         Published in                         | Code/Project |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :----------: |
| [A Comparative Study of AI-Generated (GPT-4) and Human-crafted MCQs in Programming Education](https://doi.org/10.1145/3636243.3636256) | Proceedings of the 26th Australasian Computing Education Conference |      -       |



### Financial

| Paper                                                        | Published in |           Code/Project            |
| ------------------------------------------------------------ | :----------: | :-------------------------------: |
| [FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models](http://arxiv.org/abs/2402.10986) |  Arxiv 2024  |  http://arxiv.org/abs/2402.10986  |
| [FinGLM Competition](https://github.com/MetaGLM/FinGLM)      | Github repo  | https://github.com/MetaGLM/FinGLM |



## Agent

| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|


# Dataset
## Dataset Examples
| Dataset                                            |  Home/Github | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
