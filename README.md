# A Review for Data Synthesis and Augmentation for Large Models

This is a repository for ...

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

## Data Augmentation

| Paper                                                                                                                                                                                                                                                                | Published in | Code/Project                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:--------------------------------------------------------------------------:|
| [Open-source large language models outperform crowd workers and approach ChatGPT in text-annotation tasks](https://arxiv.org/abs/2307.02179)                                                                                                                         | arxiv 2023   | -                                                                          |
| [ChatGPT outperforms crowd workers for text-annotation tasks](https://www.pnas.org/doi/pdf/10.1073/pnas.2305016120)                                                                                                                                                  | PNAS 2023    | -                                                                          |
| [Q: How to Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!](https://openaccess.thecvf.com/content/CVPR2023/papers/Khan_Q_How_To_Specialize_Large_Vision-Language_Models_to_Data-Scarce_VQA_CVPR_2023_paper.pdf) | CVPR 2023    | https://github.com/codezakh/SelTDA                                         |
| [Mind's eye: Grounded language model reasoning through simulation](https://arxiv.org/pdf/2210.05359)                                                                                                                                                                 | arxiv 2022   | -                                                                          |
| [Chatgpt-4 outperforms experts and crowd workers in annotating political twitter messages with zero-shot learning](https://arxiv.org/pdf/2304.06588)                                                                                                                 | arxiv 2023   | -                                                                          |
| [Can chatgpt reproduce human-generated labels? a study of social computing tasks](https://arxiv.org/pdf/2304.10145)                                                                                                                                                  | arxiv 2023   | -                                                                          |
| [CORE: A retrieve-then-edit framework for counterfactual data generation](https://arxiv.org/pdf/2210.04873)                                                                                                                                                          | EMNLP 2022   | https://github.com/tanay2001/CORE                                          |
| [Diversify your vision datasets with automatic diffusion-based augmentation](https://arxiv.org/pdf/2210.04873)                                                                                                                                                       | NIPS 2023    | https://github.com/lisadunlap/ALIA                                         |
| [Llamax: Scaling linguistic horizons of llm by enhancing translation capabilities beyond 100 languages](https://arxiv.org/pdf/2407.05975)                                                                                                                            | EMNLP 2024   | https://github.com/CONE-MT/LLaMAX/                                         |
| [Gpt3mix: Leveraging large-scale language models for text augmentation](https://arxiv.org/pdf/2104.08826)                                                                                                                                                            | EMNLP 2021   | https://github.com/naver-ai/hypermix                                       |
| [Closing the loop: Testing chatgpt to generate model explanations to improve human labelling of sponsored content on social media](https://arxiv.org/pdf/2306.05115)                                                                                                 | xAI 2023     | https://github.com/thalesbertaglia/chatgpt-explanations-sponsored-content/ |
| [Data augmentation using llms: Data perspectives, learning paradigms and challenges](https://arxiv.org/pdf/2403.02990)                                                                                                                                               | arxiv 2024   | -                                                                          |
| [Coannotating: Uncertainty-guided work allocation between human and large language models for data annotations](https://arxiv.org/pdf/2310.15638)                                                                                                                    | EMNLP 2023   | https://github.com/SALT-NLP/CoAnnotating                                   |

## Data Synthesis

| Paper                                                                                                                                        | Published in       | Code/Project                                 |
| -------------------------------------------------------------------------------------------------------------------------------------------- |:------------------:|:--------------------------------------------:|
| [AlpaGasus: Training A Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701)                                                      | arxiv 2023         | https://lichang-chen.github.io/AlpaGasus/    |
| [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2307.08701)                          | arxiv 2023         |                                              |
| [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)                                                  | arxiv 2023         |                                              |
| [Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model](https://arxiv.org/abs/2407.07053) | arxiv 2023         | https://multi-modal-self-instruct.github.io/ |
| [Genixer: Empowering Multimodal Large Language Models as a Powerful Data Generator](https://arxiv.org/abs/2312.06731)                        | arxiv 2023         | https://github.com/zhaohengyuan1/Genixer     |
| [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)                                             | arxiv 2022         |                                              |
| [WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](https://arxiv.org/abs/2308.09583)     | arxiv 2023         |                                              |
| [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](arXiv:2306.08568)                                                    | arxiv 2023         | https://github.com/nlpxucan/WizardLM.        |
| [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120)                                                  | arxiv 2023         |                                              |
| [VILA$^2$: VILA Augmented VILA](https://arxiv.org/abs/2407.17453)                                                                            | arxiv 2024         |                                              |
|                                                                                                                                              |                    |                                              |
| [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380)                            | arxiv 2024         |                                              |
| [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/)                      | ACL Anthology 2023 | https://github.com/yizhongw/self-instruct    |
| [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)                                                             | arxiv 2022         |                                              |

# Applications

## Data preparation

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Pretraining

| Paper                                                                                                              | Published in | Code/Project                      |
| ------------------------------------------------------------------------------------------------------------------ |:------------:|:---------------------------------:|
| [TinyStories: How small can language models be and still speak coherent English](https://arxiv.org/abs/2305.07759) | arxiv 2023   | https://huggingface.co/roneneldan |
| [Textbooks are all you need](https://arxiv.org/abs/2306.11644)                                                     | arxiv 2023   | -                                 |
| [Textbooks are all you need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)                        | arxiv 2023   | -                                 |

## Fine-Tuning

| Paper                                                                                                                                    | Published in | Code/Project                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:-----------------------------------------------------:|
| [Large language models can self-improve](https://arxiv.org/abs/2210.11610)                                                               | arxiv 2022   | -                                                     |
| [STaR: Bootstrapping reasoning with reasoning](https://arxiv.org/abs/2203.14465)                                                         | arxiv 2022   | -                                                     |
| [Language models can teach themselves to program better](https://arxiv.org/abs/2207.14502)                                               | arxiv 2022   | https://github.com/microsoft/PythonProgrammingPuzzles |
| [Self-Instruct: Aligning language models with self-generated instructions](https://arxiv.org/abs/2212.10560)                             | arxiv 2023   | https://github.com/yizhongw/self-instruct             |
| [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)                                    | github 2023  | https://github.com/tatsu-lab/stanford_alpaca          |
| [Code alpaca: An instruction-following llama model for code generation](https://github.com/sahil280114/codealpaca)                       | github 2023  | https://github.com/sahil280114/codealpaca             |
| [Code Llama: Open foundation models for code](https://arxiv.org/abs/2308.12950)                                                          | arxiv 2023   | https://github.com/meta-llama/codellama               |
| [WizardLM: Empowering large language models to follow complex instructions](https://arxiv.org/abs/2304.12244)                            | arxiv 2023   | https://github.com/nlpxucan/WizardLM                  |
| [WizardCode: Empowering code large language models with Evol-Instruct](https://arxiv.org/abs/2306.08568)                                 | arxiv 2023   | https://github.com/nlpxucan/WizardLM                  |
| [WizardMath: Empowering mathematical reasoning for large language models via reinforced evol-instruct](https://arxiv.org/abs/2308.09583) | arxiv 2023   | https://github.com/nlpxucan/WizardLM                  |
| [Magicoder: Source code is all you need](https://arxiv.org/abs/2312.02120)                                                               | arxiv 2023   | https://github.com/ise-uiuc/magicoder                 |
| [MetaMeth: Bootstap your own mathematical questions for large language models](https://arxiv.org/abs/2309.12284)                         | arxiv 2024   | https://meta-math.github.io/                          |
| [DeepSeek-Prover: Advancing theorem proving in LLMs through large-scale synthetic data](https://arxiv.org/abs/2405.14333v1)              | arxiv 2024   | -                                                     |
| [Conmmon 7B language models already possess strong math capabilities](https://arxiv.org/abs/2403.04706)                                  | arxiv 2024   | https://github.com/Xwin-LM/Xwin-LM                    |
| [Augmenting math word problems via iterative question composing](https://arxiv.org/abs/2401.09003)                                       | arxiv 2024   | https://huggingface.co/datasets/Vivacem/MMIQC         |

## Prompt

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Alignment with Human Preferences

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Knowledge Base

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## RAG and Other Tools

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Evaluation

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Optimization and Deployment

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Applications

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Agent

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

# Dataset

## Dataset Examples

| Dataset | Home/Github | Download link |
| ------- |:-----------:|:-------------:|
