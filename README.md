# A Survey on Data Synthesis and Augmentation for Large Language Models

Work In Progress.

## Table of Contents

* [Taxonomy](#Taxonomy)
  * [Data Augmentation](#Data-Augmentation)
  * [Data Synthesis](#Data-Synthesis)
* [Full Lifecycle of LLM](#Full-Lifecycle-of-LLM)
  * [Data preparation](#Data-preparation)
  * [Pretraining](#Pretraining)
  * [Fine-Tuning](#Fine-tuning)
  * [Instruction-Tuning](#Instruction-Tuning)
  * [Preference Alignment](#Preference-Alignment)
  * [Applications](#applications-1)
* [Functionality](#Functionality)
  * [Understanding](#Understanding)
  * [Logic](#Logic)
  * [Memory](#Memory)
  * [Generation](#Generation)
- [Challenges and Limitations](#Challenges-and-Limitations)
  
  - [Synthesizing and Augmenting Method](#Synthesizing-and-Augmenting-Method)
  
  - [Data Quality](#Data-Quality)
  
  - [Impact of Data Synthesis and Augmentation](#Impact-of-Data-Synthesis-and-Augmentation)
  
  - [Impact on Different Applications and Tasks](#Impact-on-Different-Applications-and-Tasks)

- [Future Directions](#Future-Directions)

# Taxonomy

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
| [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/arXiv:2306.08568)                              | arxiv 2023         | https://github.com/nlpxucan/WizardLM.        |
| [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120)                                                  | arxiv 2023         |                                              |
| [VILA$^2$: VILA Augmented VILA](https://arxiv.org/abs/2407.17453)                                                                            | arxiv 2024         |                                              |
| [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380)                            | arxiv 2024         |                                              |
| [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/)                      | ACL Anthology 2023 | https://github.com/yizhongw/self-instruct    |
| [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)                                                             | arxiv 2022         |                                              |

# Full Lifecycle of LLM

## Data preparation

| Paper                                                                                                                                                                | Published in | Code/Project                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:--------------------------------------------------------------------------:|
| [Tinystories: How small can language models be and still speak coherent english?](https://arxiv.org/abs/2305.07759)                                                  | arxiv 2023   | https://huggingface.co/roneneldan                                          |
| [Controllable dialogue simulation with in-context learning](https://arxiv.org/abs/2210.04185)                                                                        | arxiv 2022   | https://github.com/Leezekun/dialogic                                       |
| [Genie: Achieving human parity in content-grounded datasets generation](https://arxiv.org/abs/2401.14367)                                                            | arxiv 2024   | -                                                                          |
| [Case2Code: Learning Inductive Reasoning with Synthetic Data](https://arxiv.org/abs/2407.12504)                                                                      | arxiv 2024   | https://github.com/choosewhatulike/case2code                               |
| [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120)                                                                          | 41 ICML      | https://github.com/ise-uiuc/magicoder                                      |
| [ Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)                                                        | arxiv 2023   | https://arxiv.org/abs/2212.10560                                           |
| [Wizardlm: Empowering large language models to follow complex instructions](https://arxiv.org/abs/2304.12244)                                                        | arxiv 2023   | https://github.com/nlpxucan/WizardLM                                       |
| [Augmenting Math Word Problems via Iterative Question Composing](https://arxiv.org/abs/2401.09003)                                                                   | arxiv 2024   | https://huggingface.co/datasets/Vivacem/MMIQC                              |
| [Common 7b language models already possess strong math capabilities](https://arxiv.org/abs/2403.04706)                                                               | arxiv 2024   | https://github.com/Xwin-LM/Xwin-LM                                         |
| [Mammoth: Building math generalist models through hybrid instruction tuning](https://arxiv.org/abs/2309.05653)                                                       | arxiv 2023   | https://tiger-ai-lab.github.io/MAmmoTH/                                    |
| [Enhancing chat language models by scaling high-quality instructional conversations](https://arxiv.org/abs/2305.14233)                                               | arxiv 2024   | https://github.com/thunlp/UltraChat                                        |
| [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)                                             | arxiv 2024   | https://magpie-align.github.io/                                            |
| [GenQA: Generating Millions of Instructions from a Handful of Prompts](https://arxiv.org/abs/2406.10323)                                                             | arxiv 2024   | https://huggingface.co/datasets/tomg-group-umd/GenQA                       |
| [Sharegpt4v: Improving large multi-modal models with better captions](https://arxiv.org/abs/2311.12793)                                                              | arxiv 2023   | https://sharegpt4v.github.io/                                              |
| [What makes for good visual instructions? synthesizing complex visual reasoning instructions for visual instruction tuning](https://arxiv.org/abs/2311.01487)        | arxiv 2023   | https://github.com/RUCAIBox/ComVint                                        |
| [Stablellava: Enhanced visual instruction tuning with synthesized image-dialogue data](https://arxiv.org/abs/2308.10253)                                             | arxiv 2023   | https://github.com/icoz69/StableLLAVA                                      |
| [Anygpt: Unified multimodal llm with discrete sequence modeling](https://arxiv.org/abs/2402.12226)                                                                   | arxiv 2024   | https://junzhan2000.github.io/AnyGPT.github.io/                            |
| [ Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model](https://arxiv.org/abs/2407.07053)                        | arxiv 2024   | https://github.com/zwq2018/Multi-modal-Self-instruct                       |
| [Chartllama: A multimodal llm for chart understanding and generation](https://arxiv.org/abs/2311.16483)                                                              | arxiv 2023   | https://tingxueronghua.github.io/ChartLlama/                               |
| [Genixer: Empowering Multimodal Large Language Models as a Powerful Data Generator](https://arxiv.org/abs/2312.06731)                                                | arxiv 2023   | https://github.com/zhaohengyuan1/Genixer                                   |
| [Open-Source LLMs for Text Annotation: A Practical Guide for Model Setting and Fine-Tuning](https://arxiv.org/abs/2307.02179)                                        | arxiv 2024   | https://osf.io/ctgqx/                                                      |
| [ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks](https://www.pnas.org/doi/10.1073/pnas.2305016120)                                                      | NAS 2023     | -                                                                          |
| [Can Large Language Models Aid in Annotating Speech Emotional Data? Uncovering New Frontiers](https://arxiv.org/abs/2307.06090)                                      | arxiv 2023   | -                                                                          |
| [Can ChatGPT Reproduce Human-Generated Labels? A Study of Social Computing Tasks](https://arxiv.org/abs/2304.10145)                                                  | arxiv 2023   | -                                                                          |
| [Chatgpt-4 outperforms experts and crowd workers in annotating political twitter messages with zero-shot learning](https://arxiv.org/abs/2304.06588)                 | arxiv 2023   | -                                                                          |
| [Unraveling chatgpt: A critical analysis of ai-generated goal-oriented dialogues and annotations](https://arxiv.org/abs/2305.14556)                                  | ICIAAI       | -                                                                          |
| [FullAnno: A Data Engine for Enhancing Image Comprehension of MLLMs](https://arxiv.org/abs/2409.13540)                                                               | arxiv 2024   | https://arcana-project-page.github.io/                                     |
| [DISCO: Distilling counterfactuals with large language models](https://arxiv.org/abs/2212.10534)                                                                     | arxiv 2023   | https://github.com/eric11eca/disco                                         |
| [Tinygsm: achieving> 80% on gsm8k with small language models](https://arxiv.org/abs/2312.09241)                                                                      | arxiv 2023   | -                                                                          |
| [Gpt3mix: Leveraging large-scale language models for text augmentation](https://arxiv.org/abs/2104.08826)                                                            | arxiv 2021   | https://github.com/naver-ai/hypermix                                       |
| [CORE: A retrieve-then-edit framework for counterfactual data generation](https://arxiv.org/abs/2210.04873)                                                          | arxiv 2022   | https://github.com/tanay2001/CORE                                          |
| [Diversify your vision datasets with automatic diffusion-based augmentation](https://arxiv.org/abs/2305.16289)                                                       | arxiv 2023   | https://github.com/lisadunlap/ALIA                                         |
| [Closing the loop: Testing chatgpt to generate model explanations to improve human labelling of sponsored content on social media](https://arxiv.org/abs/2306.05115) | arxiv 2023   | https://github.com/thalesbertaglia/chatgpt-explanations-sponsored-content/ |
| [Toolcoder: Teach code generation models to use api search tools](https://arxiv.org/abs/2305.04032)                                                                  | arxiv 2023   | -                                                                          |
| [Coannotating: Uncertainty-guided work allocation between human and large language models for data annotation](https://arxiv.org/abs/2310.15638)                     | arxiv 2023   | https://github.com/SALT-NLP/CoAnnotating                                   |
| [Does Collaborative Human-LM Dialogue Generation Help Information Extraction from Human Dialogues?](https://arxiv.org/abs/2307.07047)                                | arxiv 2023   | https://boru-roylu.github.io/DialGen                                       |
| [Measuring mathematical problem solving with the math dataset](https://arxiv.org/abs/2103.03874)                                                                     | arxiv 2021   | -                                                                          |
| [Llemma: An open language model for mathematics](https://arxiv.org/abs/2310.10631)                                                                                   | arxiv 2023   | https://github.com/EleutherAI/math-lm                                      |
| [Code Less, Align More: Efficient LLM Fine-tuning for Code Generation with Data Pruning](https://arxiv.org/abs/2407.05040)                                           | arxiv 2024   | -                                                                          |

## Pretraining

| Paper                                                                                                                                                                   | Published in | Code/Project                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:-------------------------------------------------------:|
| [VILA2: VILA Augmented VILA](https://arxiv.org/abs/2407.17453)                                                                                                          | arxiv 2024   | https://github.com/NVlabs/VILA                          |
| [Textbooks are all you need](https://arxiv.org/abs/2306.11644)                                                                                                          | arxiv 2023   | -                                                       |
| [Textbooks are all you need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)                                                                             | arxiv 2023   | -                                                       |
| [Is Child-Directed Speech Effective Training Data for Language Models](https://arxiv.org/abs/2408.03617)                                                                | arxiv 2024   | https://babylm.github.io/index.html                     |
| [SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding](https://arxiv.org/abs/2408.15545)                                                                | arxiv 2024   | https://github.com/dptech-corp/Uni-SMART                |
| [Anygpt: Unified multimodal llm with discrete sequence modeling](https://arxiv.org/abs/2402.12226)                                                                      | arxiv 2024   | https://junzhan2000.github.io/AnyGPT.github.io/         |
| [Is synthetic data from generative models ready for image recognition](https://arxiv.org/abs/2210.07574)                                                                | arxiv 2023   | https://github.com/CVMI-Lab/SyntheticData               |
| [Rephrasing the web: A recipe for compute and data-efficient language modeling](https://arxiv.org/abs/2401.16380)                                                       | arxiv 2024   | -                                                       |
| [Physics of language models: Part 3.1, knowledge storage and extraction](https://arxiv.org/abs/2309.14316)                                                              | arxiv 2024   | https://physics.allen-zhu.com/part-3-knowledge/part-3-1 |
| [Llemma: An open language model for mathematics](https://arxiv.org/abs/2310.10631)                                                                                      | arxiv 2023   | https://github.com/EleutherAI/math-lm                   |
| [Enhancing multilingual language model with massive multilingual knowledge triples](https://arxiv.org/abs/2111.10962)                                                   | arxiv 2021   | https://github.com/ntunlp/kmlm.git                      |
| [Large language models, physics-based modeling, experimental measurements: the trinity of data-scarce learning of polymer properties](https://arxiv.org/abs/2407.02770) | arxiv 2024   | -                                                       |

## Fine-Tuning

| Paper                                                                                                                                                                 | Published in | Code/Project                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:------------------------------------------------------------:|
| [Self-Instruct: Aligning language models with self-generated instructions](https://arxiv.org/abs/2212.10560)                                                          | arxiv 2023   | https://github.com/yizhongw/self-instruct                    |
| [WizardLM: Empowering large language models to follow complex instructions](https://arxiv.org/abs/2304.12244)                                                         | arxiv 2023   | https://github.com/nlpxucan/WizardLM                         |
| [Code Llama: Open foundation models for code](https://arxiv.org/abs/2308.12950)                                                                                       | arxiv 2023   | https://github.com/meta-llama/codellama                      |
| [Scaling Relationship on Learning Mathematical Reasoning with Large Language Models](https://arxiv.org/abs/2308.01825)                                                | arxiv 2023   | https://github.com/OFA-Sys/gsm8k-ScRel                       |
| [Self-Translate-Train: A Simple but Strong Baseline for Cross-lingual Transfer of Large Language Models](https://arxiv.org/abs/2407.00454)                            | arxiv 2024   | -                                                            |
| [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://arxiv.org/abs/2207.01780)                                       | NeurIPS 2022 | https://github.com/salesforce/CodeRL                         |
| [Self-play fine-tuning converts weak language models to strong language models](https://arxiv.org/abs/2401.01335)                                                     | arxiv 2024   | https://github.com/uclaml/SPIN                               |
| [Language models can teach themselves to program better](https://arxiv.org/abs/2207.14502)                                                                            | arxiv 2022   | https://github.com/microsoft/PythonProgrammingPuzzles        |
| [DeepSeek-Prover: Advancing theorem proving in LLMs through large-scale synthetic data](https://arxiv.org/abs/2405.14333v1)                                           | arxiv 2024   | -                                                            |
| [STaR: Bootstrapping reasoning with reasoning](https://arxiv.org/abs/2203.14465)                                                                                      | arxiv 2022   | -                                                            |
| [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998)                                                                             | arxiv 2023   | -                                                            |
| [Beyond human data: Scaling self-training for problem-solving with language models](https://arxiv.org/abs/2312.06585)                                                 | arxiv 2023   | -                                                            |
| [Code alpaca: An instruction-following llama model for code generation](https://github.com/sahil280114/codealpaca)                                                    | github 2023  | https://github.com/sahil280114/codealpaca                    |
| [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)                                                                 | github 2023  | https://github.com/tatsu-lab/stanford_alpaca                 |
| [Huatuo: Tuning llama model with chinese medical knowledge](https://arxiv.org/abs/2304.06975)                                                                         | arxiv 2023   | https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese          |
| [Magicoder: Source code is all you need](https://arxiv.org/abs/2312.02120)                                                                                            | arxiv 2023   | https://github.com/ise-uiuc/magicoder                        |
| [Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models](https://arxiv.org/abs/2311.00287)                     | STEP         | https://github.com/ritaranx/ClinGen                          |
| [Unnatural instructions: Tuning language models with (almost) no human labor](https://arxiv.org/abs/2212.09689)                                                       | arxiv 2022   | https://github.com/orhonovich/unnatural-instructions         |
| [Baize: An open-source chat model with parameter-efficient tuning on self-chat data](https://arxiv.org/abs/2304.01196)                                                | arxiv 2023   | https://github.com/project-baize/baize-chatbot               |
| [Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Model](https://arxiv.org/abs/2305.16635)     | arxiv 2023   | -                                                            |
| [Llm2llm: Boosting llms with novel iterative data enhancement](https://arxiv.org/abs/2403.15042)                                                                      | arxiv 2024   | https://github.com/SqueezeAILab/LLM2LLM                      |
| [WizardCode: Empowering code large language models with Evol-Instruct](https://arxiv.org/abs/2306.08568)                                                              | arxiv 2023   | https://github.com/nlpxucan/WizardLM                         |
| [Generative AI for Math: Abel]()                                                                                                                                      | arxiv 2024   | -                                                            |
| [Orca: Progressive learning from complex explanation traces of gpt-4](https://arxiv.org/abs/2306.02707)                                                               | arxiv 2023   | https://www.microsoft.com/en-us/research/project/orca/       |
| [Orca 2: Teaching small language models how to reason](https://arxiv.org/abs/2311.11045)                                                                              | arxiv 2023   | -                                                            |
| [Mammoth: Building math generalist models through hybrid instruction tuning](https://arxiv.org/abs/2309.05653)                                                        | arxiv 2023   | https://tiger-ai-lab.github.io/MAmmoTH/                      |
| [Lab: Large-scale alignment for chatbots](https://arxiv.org/abs/2403.01081)                                                                                           | arxiv 2024   | -                                                            |
| [Synthetic data (almost) from scratch: Generalized instruction tuning for language models](https://arxiv.org/abs/2402.13064)                                          | arxiv 2024   | https://thegenerality.com/agi/                               |
| [SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding](https://arxiv.org/abs/2408.15545)                                                              | arxiv 2024   | https://github.com/dptech-corp/Uni-SMART/tree/main/SciLitLLM |
| [Llava-med: Training a large language-and-vision assistant for biomedicine in one day](https://arxiv.org/abs/2306.00890)                                              | arxiv 2024   | https://github.com/microsoft/LLaVA-Med                       |
| [Visual instruction tuning](https://arxiv.org/abs/2304.08485)                                                                                                         | NIPS 2024    | -                                                            |
| [Chartllama: A multimodal llm for chart understanding and generation](https://arxiv.org/abs/2311.16483)                                                               | arxiv 2023   | https://tingxueronghua.github.io/ChartLlama/                 |
| [Sharegpt4v: Improving large multi-modal models with better captions](https://arxiv.org/abs/2311.12793)                                                               | arxiv 2023   | https://sharegpt4v.github.io/                                |
| [Next-gpt: Any-to-any multimodal llm](https://arxiv.org/abs/2309.05519)                                                                                               | arxiv 2023   | https://next-gpt.github.io/                                  |
| [Does synthetic data generation of llms help clinical text mining? ](https://arxiv.org/abs/2303.04360)                                                                | arxiv 2023   | -                                                            |
| [Ultramedical: Building specialized generalists in biomedicine](https://arxiv.org/abs/2406.03949)                                                                     | arxiv 2024   | https://github.com/TsinghuaC3I/UltraMedical                  |
| [Q: How to Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!](https://arxiv.org/abs/2306.03932)                    | arxiv 2023   | https://github.com/codezakh/SelTDA                           |
| [MetaMeth: Bootstap your own mathematical questions for large language models](https://arxiv.org/abs/2309.12284)                                                      | arxiv 2024   | https://meta-math.github.io/                                 |
| [Symbol tuning improves in-context learning in language models](https://arxiv.org/abs/2305.08298)                                                                     | arxiv 2023   | -                                                            |
| [DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation](https://arxiv.org/abs/2308.14346)                                           | arxiv 2023   | https://github.com/FudanDISC/DISC-MedLLM                     |
| [Mathgenie: Generating synthetic data with question back-translation for enhancing mathematical reasoning of llms](https://arxiv.org/abs/2402.16352)                  | arxiv 2024   | -                                                            |
| [BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT](https://arxiv.org/abs/2310.15896) | arxiv 2023   | https://github.com/scutcyr/BianQue                           |

## Instruction-Tuning

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Preference Alignment

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Applications

### Math

| Paper                                                                                                                                                                | Published in          | Code/Project                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:---------------------:|:---------------------------------------------:|
| [Galactica: A Large Language Model for Science](http://arxiv.org/abs/2211.09085)                                                                                     | arxiv 2022            | -                                             |
| [STaR: Bootstrapping Reasoning With Reasoning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) | NeurIPS 2022          | https://github.com/ezelikman/STaR             |
| [Multilingual Mathematical Autoformalization](https://arxiv.org/abs/2311.03755)                                                                                      | arxiv 2023            | -                                             |
| [WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](http://arxiv.org/abs/2308.09583)                              | arxiv 2023            | https://github.com/nlpxucan/WizardLM          |
| [MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning](https://arxiv.org/abs/2309.05653)                                                       | arxiv 2023            | -                                             |
| [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](http://arxiv.org/abs/2309.12284)                                                     | arxiv 2023            | https://meta-math.github.io/                  |
| [Synthetic Dialogue Dataset Generation using LLM Agents](https://arxiv.org/abs/2401.17461)                                                                           | EMNLP Workshop 2023   | -                                             |
| [Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://openreview.net/forum?id=TPtXLihkny)                                                   | NeurIPS Workshop 2024 | -                                             |
| [Synthetic Dialogue Dataset Generation using LLM Agents](http://arxiv.org/abs/2401.17461)                                                                            | arxiv 2024            | -                                             |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)                                               | arxiv 2024            | https://github.com/deepseek-ai/DeepSeek-Math  |
| [DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://arxiv.org/abs/2405.14333)                                            | arxiv 2024            | -                                             |
| [Augmenting Math Word Problems via Iterative Question Composing](http://arxiv.org/abs/2401.09003)                                                                    | arxiv 2024            | https://huggingface.co/datasets/Vivacem/MMIQC |

### Science

| Paper                                                                                                                                                                                 | Published in                     | Code/Project                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------:|:------------------------------------------------------------:|
| [Galactica: A Large Language Model for Science](http://arxiv.org/abs/2211.09085)                                                                                                      | arxiv 2022                       | -                                                            |
| [Reflection-Tuning: Recycling Data for Better Instruction-Tuning](https://openreview.net/forum?id=xaqoZZqkPU)                                                                         | NeurIPS Workshop 2023 / ACL 2024 | https://github.com/tianyi-lab/Reflection_Tuning              |
| [Reflexion: language agents with verbal reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) | NeurIPS 2023                     | https://github.com/noahshinn024/reflexion                    |
| [SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding](https://arxiv.org/abs/2408.15545)                                                                              | NeurIPS Workshop 2024            | https://github.com/dptech-corp/Uni-SMART/tree/main/SciLitLLM |
| [SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models](https://arxiv.org/abs/2401.07950)                                              | NeurIPS 2024                     | https://github.com/THUDM/SciGLM                              |
| [ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852)                                                                                                          | arxiv 2024                       | -                                                            |

### Code

| Paper                                                                                                                                                                                                               | Published in         | Code/Project                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------------:|:-------------------------------------:|
| [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html) | NIPS 2022            | https://github.com/salesforce/CodeRL  |
| [Generating Programming Puzzles to Train Language Models](https://openreview.net/forum?id=H8cx0iO-y-9)                                                                                                              | ICLR 2022 (Workshop) | -                                     |
| [Language Models Can Teach Themselves to Program Better](http://arxiv.org/abs/2207.14502)                                                                                                                           | ICLR 2023            | -                                     |
| [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)                                                                                                                                                      | Arxiv 2023           | -                                     |
| [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)                                                                                                                         | Arxiv 2023           | -                                     |
| [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)                                                                                                         | ICLR 2023            | -                                     |
| [Learning Performance-Improving Code Edits](http://arxiv.org/abs/2302.07867)                                                                                                                                        | ICLR 2024            | https://pie4perf.com/                 |
| [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](http://arxiv.org/abs/2306.08568)                                                                                                            | ICLR 2024            | https://github.com/nlpxucan/WizardLM  |
| [Magicoder: Source Code Is All You Need](http://arxiv.org/abs/2312.02120)                                                                                                                                           | ICML 2024            | https://github.com/ise-uiuc/magicoder |

### Medical

| Paper                                                                                                                                             | Published in  | Code/Project                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------- |:-------------:|:---------------------------------------------------:|
| [MedDialog: Large-scale Medical Dialogue Datasets](https://aclanthology.org/2020.emnlp-main.743/)                                                 | EMNLP 2020    | https://github.com/UCSDAI4H/Medical-Dialogue-System |
| [HuatuoGPT, towards Taming Language Model to Be a Doctor](https://arxiv.org/abs/2305.15075)                                                       | EMNLP 2023    | https://github.com/FreedomIntelligence/HuatuoGPT    |
| [HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs](https://arxiv.org/abs/2311.09774)                                                 | arxiv 2023    | https://github.com/FreedomIntelligence/HuatuoGPT-II |
| [ChatCounselor: A Large Language Models for Mental Health Support](https://arxiv.org/abs/2309.15461)                                              | arxiv 2023    | -                                                   |
| [DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation](https://arxiv.org/abs/2308.14346)                       | arxiv 2023    | https://github.com/FudanDISC/DISC-MedLLM            |
| [Biomedical discovery through the integrative biomedical knowledge hub (iBKH)](https://www.cell.com/iscience/fulltext/S2589-0042(23)00537-0)      | iScience 2023 | https://github.com/wcm-wanglab/iBKH                 |
| [Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models](https://arxiv.org/abs/2311.00287) | arxiv 2023    | https://github.com/ritaranx/ClinGen                 |
| [ShenNong-TCM](https://github.com/michael-wzhu/ShenNong-TCM-LLM)                                                                                  | Github repo   | https://github.com/michael-wzhu/ShenNong-TCM-LLM    |
| [ZhongJing(仲景)](https://github.com/pariskang/CMLM-ZhongJing)                                                                                      | Github repo   | https://github.com/pariskang/CMLM-ZhongJing         |

### Law

| Paper                                                                                                            | Published in | Code/Project                                     |
| ---------------------------------------------------------------------------------------------------------------- |:------------:|:------------------------------------------------:|
| [DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services](http://arxiv.org/abs/2309.11325) | arxiv 2023   | https://github.com/FudanDISC/DISC-LawLLM         |
| [Lawyer LLaMA Technical Report](http://arxiv.org/abs/2305.15062)                                                 | arxiv 2023   | -                                                |
| [LawGPT: A Chinese Legal Knowledge-Enhanced Large Language Model](http://arxiv.org/abs/2406.04614)               | arxiv 2024   | https://github.com/pengxiao-song/LaWGPT          |
| [WisdomInterrogatory](https://github.com/zhihaiLLM/wisdomInterrogatory)                                          | Github repo  | https://github.com/zhihaiLLM/wisdomInterrogatory |

### Education

| Paper                                                                                                                                  | Published in                                                        | Code/Project |
| -------------------------------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------:|:------------:|
| [A Comparative Study of AI-Generated (GPT-4) and Human-crafted MCQs in Programming Education](https://doi.org/10.1145/3636243.3636256) | Proceedings of the 26th Australasian Computing Education Conference | -            |

### Financial

| Paper                                                                                                          | Published in | Code/Project                      |
| -------------------------------------------------------------------------------------------------------------- |:------------:|:---------------------------------:|
| [FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models](http://arxiv.org/abs/2402.10986) | Arxiv 2024   | http://arxiv.org/abs/2402.10986   |
| [FinGLM Competition](https://github.com/MetaGLM/FinGLM)                                                        | Github repo  | https://github.com/MetaGLM/FinGLM |

## Agent

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

# Functionality

## Understanding

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Logic

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Memory

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Generation

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

# Challenges and Limitations

## Synthesizing and Augmenting Method

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Data Quality

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Impact of Data Synthesis and Augmentation

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

## Impact on Different Applications and Tasks

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|

# Future Directions

| Paper | Published in | Code/Project |
| ----- |:------------:|:------------:|
