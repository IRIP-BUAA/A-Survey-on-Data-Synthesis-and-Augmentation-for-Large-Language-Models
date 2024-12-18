# A Survey on Data Synthesis and Augmentation for Large Language Models

A collection of AWESOME papers on data synthesis and augmentation for Large Language Models.

🌏Please check out our survey paper: https://arxiv.org/abs/2410.12896.

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
| [Diversify your vision datasets with automatic diffusion-based augmentation](https://arxiv.org/pdf/2210.04873)                                                                                                                                                       | NeurlPS 2023 | https://github.com/lisadunlap/ALIA                                         |
| [Llamax: Scaling linguistic horizons of llm by enhancing translation capabilities beyond 100 languages](https://arxiv.org/pdf/2407.05975)                                                                                                                            | EMNLP 2024   | https://github.com/CONE-MT/LLaMAX/                                         |
| [Gpt3mix: Leveraging large-scale language models for text augmentation](https://arxiv.org/pdf/2104.08826)                                                                                                                                                            | EMNLP 2021   | https://github.com/naver-ai/hypermix                                       |
| [Closing the loop: Testing chatgpt to generate model explanations to improve human labelling of sponsored content on social media](https://arxiv.org/pdf/2306.05115)                                                                                                 | xAI 2023     | https://github.com/thalesbertaglia/chatgpt-explanations-sponsored-content/ |
| [Data augmentation using llms: Data perspectives, learning paradigms and challenges](https://arxiv.org/pdf/2403.02990)                                                                                                                                               | arxiv 2024   | -                                                                          |
| [Coannotating: Uncertainty-guided work allocation between human and large language models for data annotations](https://arxiv.org/pdf/2310.15638)                                                                                                                    | EMNLP 2023   | https://github.com/SALT-NLP/CoAnnotating                                   |

## Data Synthesis

| Paper                                                                                                                                        | Published in       | Code/Project                                 |
| -------------------------------------------------------------------------------------------------------------------------------------------- |:------------------:|:--------------------------------------------:|
| [AlpaGasus: Training A Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701)                                                      | arxiv 2023         | https://lichang-chen.github.io/AlpaGasus/    |
| [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2307.08701)                          | arxiv 2023         | -                                            |
| [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)                                                  | arxiv 2023         | -                                            |
| [Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model](https://arxiv.org/abs/2407.07053) | arxiv 2023         | https://multi-modal-self-instruct.github.io/ |
| [Genixer: Empowering Multimodal Large Language Models as a Powerful Data Generator](https://arxiv.org/abs/2312.06731)                        | arxiv 2023         | https://github.com/zhaohengyuan1/Genixer     |
| [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)                                             | arxiv 2022         | -                                            |
| [WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](https://arxiv.org/abs/2308.09583)     | arxiv 2023         | -                                            |
| [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/arXiv:2306.08568)                              | arxiv 2023         | https://github.com/nlpxucan/WizardLM.        |
| [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120)                                                  | arxiv 2023         | -                                            |
| [VILA$^2$: VILA Augmented VILA](https://arxiv.org/abs/2407.17453)                                                                            | arxiv 2024         | -                                            |
| [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380)                            | arxiv 2024         | -                                            |
| [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/)                      | ACL Anthology 2023 | https://github.com/yizhongw/self-instruct    |
| [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)                                                             | arxiv 2022         | -                                            |

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

| Paper                                                                                                                                                                | Published in   | Code/Project                                           |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------:|:------------------------------------------------------:|
| [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)                                                         |                | https://github.com/tatsu-lab/stanford_alpaca           |
| [AlpaGasus: Training A Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701)                                                                              | arXiv 2023     | https://lichang-chen.github.io/AlpaGasus/              |
| [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/)                                               |                | https://github.com/lm-sys/FastChat                     |
| [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)                                                        | arXiv 2023     | https://github.com/nlpxucan/WizardLM                   |
| [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707)                                                              | arXiv 2023     | https://www.microsoft.com/en-us/research/project/orca/ |
| [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)                                                                             | arXiv 2023     | https://www.microsoft.com/en-us/research/project/orca/ |
| [Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data](https://arxiv.org/abs/2304.01196)                                               | arXiv 2023     | https://github.com/project-baize/baize-chatbot         |
| [LongForm: Effective Instruction Tuning with Reverse Instructions](https://arxiv.org/abs/2304.08460)                                                                 | arXiv 2023     | https://github.com/akoksal/LongForm                    |
| [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)                                                                                                        | NeurIPS 2024   | https://llava-vl.github.io/                            |
| [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)                                                                                | IEEE 2024      | https://llava-vl.github.io/                            |
| [LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](https://arxiv.org/abs/2311.05437)                                                                 | arXiv 2023     | https://llava-vl.github.io/llava-plus/                 |
| [LLaVA-Interactive: An All-in-One Demo for Image Chat, Segmentation, Generation and Editing](https://arxiv.org/abs/2311.00571)                                       | arXiv 2023     | -                                                      |
| [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890)                                             | NeurIPS 2024   | https://aka.ms/llava-med                               |
| [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)                                                         | arXiv 2022     | https://github.com/yizhongw/self-instruct              |
| [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259)                                                                                  | arXiv 2023     |                                                        |
| [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)                                                    | arXiv 2024     | https://github.com/uclaml/SPIN                         |
| [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)                                                | arXiv 2023     | -                                                      |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)                                                                                 | arXiv 2022     | -                                                      |
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)                                                                    | arXiv 2023     | -                                                      |
| [Q: How to Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!](https://arxiv.org/abs/2306.03932)                   | CVPR 2023      | https://github.com/codezakh/SelTDA                     |
| [ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning](https://arxiv.org/abs/2304.06588)                 | arXiv 2023     | -                                                      |
| [Prompting Large Language Model for Machine Translation: A Case Study](arxiv.org/abs/2301.07069)                                                                     | arXiv 2023     | -                                                      |
| [T-SciQ: Teaching Multimodal Chain-of-Thought Reasoning via Mixed Large Language Model Signals for Science Question Answering](https://arxiv.org/abs/2305.03453)     | AAAI 2024      | https://github.com/T-SciQ/T-SciQ                       |
| [CORE: A Retrieve-then-Edit Framework for Counterfactual Data Generation](https://arxiv.org/abs/2210.04873)                                                          | arXiv 2022     | https://github.com/tanay2001/CORE                      |
| [Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation](https://arxiv.org/abs/2305.16289)                                                       | NeurIPS 2023   | https://github.com/lisadunlap/ALIA                     |
| [AugGPT: Leveraging ChatGPT for Text Data Augmentation](https://arxiv.org/abs/2302.13007)                                                                            | arXiv 2023     | -                                                      |
| [CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation](https://arxiv.org/abs/2310.15638)                     | arXiv 2023     | https://github.com/SALT-NLP/CoAnnotating               |
| [Closing the Loop: Testing ChatGPT to Generate Model Explanations to Improve Human Labelling of Sponsored Content on Social Media](https://arxiv.org/abs/2306.05115) | Springer, Cham | -                                                      |
| [ToolCoder: Teach Code Generation Models to use API search tools](https://arxiv.org/abs/2305.04032)                                                                  | arXiv 2023     | -                                                      |

## Preference Alignment

| Paper                                                                                                                      | Published in | Code/Project                                                  |
| -------------------------------------------------------------------------------------------------------------------------- |:------------:|:-------------------------------------------------------------:|
| [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://arxiv.org/abs/2310.01377)                        | arXiv 2023   | -                                                             |
| [HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM](https://arxiv.org/abs/2311.09528)                             | arXiv 2023   | https://huggingface.co/datasets/nvidia/HelpSteer              |
| [Learning From Mistakes Makes LLM Better Reasoner](https://arxiv.org/abs/2310.20689)                                       | arXiv 2023   | https://github.com/microsoft/LEMA                             |
| [Bot-Adversarial Dialogue for Safe Conversational Agents](https://aclanthology.org/2021.naacl-main.235/)                   | ACL 2021     | https://parl.ai/projects/safety_recipes/                      |
| [BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://arxiv.org/abs/2307.04657)   | NIPS 2024    | https://sites.google.com/view/pku-beavertails                 |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)                                                              | arXiv 2023   | -                                                             |
| [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)                        | arXiv 2021   | https://www.microsoft.com/en-us/bing/apis/bing-web-search-api |
| [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.04792)                                | arXiv 2024   | -                                                             |
| [Self-Judge: Selective Instruction Following with Alignment Self-Evaluation](https://arxiv.org/abs/2409.00935)             | arXiv 2024   | https://github.com/nusnlp/Self-J                              |
| [SALMON: Self-Alignment with Instructable Reward Models](https://iclr.cc/virtual/2024/poster/17454)                        | ICLR 2024    | https://github.com/IBM/SALMON                                 |
| [SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF](https://arxiv.org/abs/2310.05344)          | arXiv 2023   | https://huggingface.co/nvidia/SteerLM-llama2-13B              |
| [Starling-7B: Increasing LLM Helpfulness & Harmlessness with RLAIF](https://openreview.net/forum?id=GqDntYTTbk#discussion) | COLM 2024    | https://starling.cs.berkeley.edu/                             |
| [Advancing LLM Reasoning Generalists with Preference Trees](https://arxiv.org/abs/2404.02078)                              | arXiv 2024   | https://github.com/OpenBMB/Eurus                              |
| [CriticBench: Benchmarking LLMs for Critique-Correct Reasoning](https://arxiv.org/abs/2402.14809)                          | arXiv 2024   | https://criticbench.github.io/                                |

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

# Functionality

## Understanding

| Paper                                                                                                                 | Published in | Code/Project                                 |
| --------------------------------------------------------------------------------------------------------------------- |:------------:|:--------------------------------------------:|
| [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)          | -            | https://github.com/tatsu-lab/stanford_alpaca |
| [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)         | arxiv 2023   | https://github.com/nlpxucan/WizardLM         |
| [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380)     | arxiv 2024   | -                                            |
| [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)                                                         | NIPS 2024    | https://llava-vl.github.io                   |
| [ChartLlama: A Multimodal LLM for Chart Understanding and Generation](https://arxiv.org/abs/2311.16483)               | arxiv 2023   | https://tingxueronghua.github.io/ChartLlama/ |
| [Genixer: Empowering Multimodal Large Language Models as a Powerful Data Generator](https://arxiv.org/abs/2312.06731) | arxiv 2023   | https://github.com/zhaohengyuan1/Genixer     |

## Logic

| Paper                                                                                                                                              | Published in | Code/Project                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:---------------------------------------------:|
| [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)                              | arxiv 2023   | -                                             |
| [Case2Code: Learning Inductive Reasoning with Synthetic Data](https://arxiv.org/abs/2407.12504)                                                    | arxiv 2024   | https://github.com/choosewhatulike/case2code  |
| [MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning](https://arxiv.org/abs/2309.05653)                                     | arxiv 2023   | https://tiger-ai-lab.github.io/MAmmoTH/       |
| [Augmenting Math Word Problems via Iterative Question Composing](https://arxiv.org/abs/2401.09003)                                                 | ICLR 2024    | https://huggingface.co/datasets/Vivacem/MMIQC |
| [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)                                                                   | arxiv 2022   | -                                             |
| [Q: How to Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!](https://arxiv.org/abs/2306.03932) | CVPR 2023    | https://github.com/codezakh/SelTDA            |

## Memory

| Paper                                                                                                                                 | Published in   | Code/Project                                           |
| ------------------------------------------------------------------------------------------------------------------------------------- |:--------------:|:------------------------------------------------------:|
| [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)                         | arxiv 2024     | -                                                      |
| [LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities](https://arxiv.org/abs/2305.13168) | World Wide Web | https://github.com/zjunlp/AutoKG                       |
| [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)                                       | arxiv 2024     | https://github.com/tencent-ailab/persona-hub           |
| [AceCoder: Utilizing Existing Code to Enhance Code Generation](https://arxiv.org/abs/2303.17780)                                      | arxiv 2023     | -                                                      |
| [RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation](https://arxiv.org/abs/2303.12570)            | arXiv 2023     | https://github.com/microsoft/CodeT/tree/main/RepoCoder |

## Generation

| Paper                                                                                                               | Published in | Code/Project                                        |
| ------------------------------------------------------------------------------------------------------------------- |:------------:|:---------------------------------------------------:|
| [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367)           | arxiv 2024   | -                                                   |
| [UltraMedical: Building Specialized Generalists in Biomedicine](https://arxiv.org/abs/2406.03949)                   | arxiv 2024   | https://github.com/TsinghuaC3I/UltraMedical         |
| [HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://arxiv.org/abs/2304.06975)                       | arxiv 2023   | https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese |
| [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) | arxiv 2023   | -                                                   |
| [Controllable Dialogue Simulation with In-Context Learning](https://arxiv.org/abs/2210.04185)                       | arxiv 2022   | https://github.com/Leezekun/dialogic                |
| [Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation](https://arxiv.org/abs/2305.16289)      | NIPS 2023    | https://github.com/lisadunlap/ALIA                  |

# Challenges and Limitations

## Synthesizing and Augmenting Method

| Paper                                                                                                                                    | Published in | Code/Project                            |
| ---------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:---------------------------------------:|
| [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)                  | arxiv 2023   | -                                       |
| [LLM2LLM: Boosting LLMs with Novel Iterative Data Enhancement](https://arxiv.org/abs/2403.15042)                                         | arxiv 2024   | https://github.com/SqueezeAILab/LLM2LLM |
| [WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](https://arxiv.org/abs/2308.09583) | arxiv 2023   | https://github.com/nlpxucan/WizardLM    |
| [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)                                                         | NIPS 2022    | -                                       |
| [SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models](https://arxiv.org/abs/2401.07950) | arxiv 2024   | https://github.com/THUDM/SciGLM         |
| [ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852)                                                             | arxiv 2024   | -                                       |

## Data Quality

| Paper                                                                                                                                             | Published in | Code/Project               |
| ------------------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:--------------------------:|
| [LLMs4Synthesis: Leveraging Large Language Models for Scientific Synthesis](https://arxiv.org/abs/2409.18812)                                     | arxiv 2024   | -                          |
| [CoRAL: Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation](https://dl.acm.org/doi/10.1145/3637528.3671901) | ACM          | -                          |
| [Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate](https://arxiv.org/abs/2305.11595)           | arxiv 2023   | -                          |
| [LTGC: Long-tail Recognition via Leveraging LLMs-driven Generated Content](https://arxiv.org/abs/2403.05854)                                      | CVPR 2024    | https://ltgccode.github.io |

## Impact of Data Synthesis and Augmentation

| Paper                                                                                                                | Published in | Code/Project                                   |
| -------------------------------------------------------------------------------------------------------------------- |:------------:|:----------------------------------------------:|
| [DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows](https://arxiv.org/abs/2402.10379) | arxiv 2024   | https://github.com/datadreamer-dev/DataDreamer |
| [HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection](https://arxiv.org/abs/2408.02927)      | arxiv 2024   | https://github.com/The-FinAI/HARMONIC          |

## Impact on Different Applications and Tasks

| Paper                                                                                                                                  | Published in | Code/Project                       |
| -------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:----------------------------------:|
| [PANDA: Preference Adaptation for Enhancing Domain-Specific Abilities of LLMs](https://aclanthology.org/2024.findings-acl.651/)        | ACL 2024     | https://github.com/THUNLP-MT/PANDA |
| [Role Prompting Guided Domain Adaptation with General Capability Preserve for Large Language Models](https://arxiv.org/abs/2403.02756) | arxiv 2024   | -                                  |

# Future Directions

| Paper                                                                                                              | Published in  | Code/Project                                                  |
| ------------------------------------------------------------------------------------------------------------------ |:-------------:|:-------------------------------------------------------------:|
| [A Universal Metric for Robust Evaluation of Synthetic Tabular Data](https://ieeexplore.ieee.org/document/9984938) | IEEE 2022     | -                                                             |
| [CoLa-Diff: Conditional Latent Diffusion Model for Multi-Modal MRI Synthesis](https://arxiv.org/abs/2303.14081)    | Springer 2023 | -                                                             |
| [LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](https://arxiv.org/abs/2311.05437)               | arxiv 2023    | https://llava-vl.github.io/llava-plus/                        |
| [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568)          | arxiv 2023    | https://github.com/nlpxucan/WizardLM                          |
| [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/abs/2402.12226)                 | arxiv 2024    | https://junzhan2000.github.io/AnyGPT.github.io/               |
| [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)                | arxiv 2021    | https://www.microsoft.com/en-us/bing/apis/bing-web-search-api |
| [NExT-GPT: Any-to-Any Multimodal LLM](https://arxiv.org/abs/2309.05519)                                            | arxiv 2023    | https://next-gpt.github.io/                                   |
