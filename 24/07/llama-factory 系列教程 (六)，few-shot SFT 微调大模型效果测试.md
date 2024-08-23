## 背景

## 实验：在不同的文本分类数据集尺寸上微调大模型

在这次实验中，我们分别使用了100、500、1000和2000条数据对大模型进行了微调。我们的目标是评估不同大小的数据集对大模型表现的影响。

项目开源地址：[https://github.com/JieShenAI/csdn/blob/main/24/07/few_shot_sft/readme.md](https://github.com/JieShenAI/csdn/blob/main/24/07/few_shot_sft/readme.md)

### 实验方法

为了高效地完成微调任务，我们使用了Linux shell脚本 自动化运行。具体步骤如下：
1. **数据准备**：将不同大小的数据集准备好。
2. **批量微调**：利用Linux shell脚本批量化地微调大模型，自动保存微调后的模型权重。
3. **自动评估**：微调完成后，脚本会自动调用评估程序，对模型在测试集上的表现进行评估。

这种方法极大地提高了工作效率。若不使用自动化脚本，我们需要手动逐个训练模型，然后手动运行评估程序，这不仅耗时，而且容易出错。

### 优势

- **时间节省**：利用自动化脚本，我们可以在夜间让计算机自行完成微调和评估工作，第二天早上起床后即可查看结果。
- **减少人工干预**：整个过程无需过多人工干预，减少了人工的时间与精力。

通过这种方式，我们能够得出不同大小数据集对大模型表现的影响，为进一步的研究提供了宝贵的数据支持。

## 项目文件介绍

* `build_llm_data.ipynb`
  从训练集中随机筛选并转换为Alpaca样式的数据集格式
  在大模型的微调过程中，从训练集中随机抽取不同规模的数据样本，以便进行模型的测试和优化。本文从训练集中随机筛选100、500、1000和2000条数据，并将这些数据转换为Alpaca样式的微调数据集格式，最后将筛选后的数据保存在data文件夹下。

  本文在文本分类数据集上进行模型训练。

  下述是转化为大模型微调的数据集样例：
  
  ```json
  [
    {
      "instruction": "You are a document classifier. When given the text, you classify the text into one of the following categories:\n\n\"Human Necessities\"\n\"Performing Operations; Transporting\"\n\"Chemistry; Metallurgy\"\n\"Textiles; Paper\"\n\"Fixed Constructions\"\n\"Mechanical Engineering; Lightning; Heating; Weapons; Blasting\"\n\"Physics\"\n\"Electricity\"\n\"General tagging of new or cross-sectional technology\"\n\"Unknown\"\n\nYour output should only contain one of the categories and no explanation or any other text.",
      "input": "Classify the document:\nan image sensor device may include a dual - gated charge storage region within a substrate . the dual - gated charge storage region includes first and second diodes within a common charge generating region . this charge generating region is configured to receive light incident on a surface of the image sensor device . the first and second diodes include respective first conductivity type regions responsive to first and second gate signals , respectively . these first and second gate signals are active during non - overlapping time intervals .",
      "output": "Electricity"
    },
    ...
  ]
  ```


* `train.sh`
  在开始训练之前，需要在 `LLaMA-Factory/data/dataset_info.json` 文件中注册 `data` 目录下的数据集。接下来，从 LLaMA-Factory 的可视化界面获取 LoRA 微调的命令行。`train.sh` 脚本实现了批量化训练，并在训练完成后保存 LoRA 的权重。

  ```shell
  # 对所有切分后的数据集进行训练
  cd LLaMA-Factory
  data_files=(llm_train_100 llm_train_500 llm_train_1000 llm_train_2000)
  echo ${data_files[@]}
  
  for data_file in ${data_files[@]}; do
      echo ${data_file}
      llamafactory-cli train \
          --stage sft \
          --do_train True \
          --model_name_or_path ZhipuAI/glm-4-9b-chat \
          --preprocessing_num_workers 16 \
          --finetuning_type lora \
          --template glm4 \
          --flash_attn auto \
          --dataset_dir data \
          --dataset ${data_file} \
          --cutoff_len 1024 \
          --learning_rate 5e-05 \
          --num_train_epochs 3.0 \
          --max_samples 100000 \
          --per_device_train_batch_size 2 \
          --gradient_accumulation_steps 4 \
          --lr_scheduler_type cosine \
          --max_grad_norm 1.0 \
          --logging_steps 5 \
          --save_steps 100 \
          --warmup_steps 0 \
          --optim adamw_torch \
          --packing False \
          --report_to none \
          --output_dir saves/GLM-4-9B-Chat/lora/240731-${data_file} \
          --fp16 True \
          --plot_loss True \
          --ddp_timeout 180000000 \
          --include_num_input_tokens_seen True \
          --lora_rank 8 \
          --lora_alpha 16 \
          --lora_dropout 0 \
          --lora_target all
  done
  
  # nohup bash train.sh > train.log 2>&1 &
  ```


* `eval.sh`
  在训练完成后，使用 VLLM 部署训练完成的 LoRA 模型，并将其部署成 API 接口，便于通过 `infer_eval.py` 进行评估。`eval.sh` 脚本实现了对训练模型的批量部署与评估，自动化地逐个部署和推理。在评估完成一个大模型后，脚本会杀死正在部署的进程，开始部署下一个大模型，并进行新的评估。

  ```python
  # conda activate llm
  cd LLaMA-Factory
  
  # kw_arr=(llm_train_100 llm_train_500 llm_train_1000 llm_train_2000)
  kw_arr=(llm_train_100 llm_train_500 llm_train_1000)
  
  
  for kw in "${kw_arr[@]}"; do
      echo $kw
      CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
          --model_name_or_path /home/jie/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat \
          --adapter_name_or_path ./saves/GLM-4-9B-Chat/lora/240731-${kw} \
          --template glm4 \
          --finetuning_type lora \
          --infer_backend vllm \
          --vllm_enforce_eager &
  
      python ../infer_eval.py ${kw} > ../logs/${kw}.log 2>&1
      # 杀掉服务进程
      pkill -f llamafactory
      echo "Stopped llamafactory"
  done
  
  # nohup bash eval.sh > eval.log 2>&1 &
  ```

* `infer_eval.py`
  利用在线部署的大模型，结合 LangChain 工具，在测试集上逐个进行评估。

  ```python
  import os
  import json
  import random
  import logging
  import argparse
  import pickle
  import evaluate
  from tqdm import tqdm
  from datasets import load_dataset
  from dataclasses import dataclass, field
  from langchain_openai import ChatOpenAI
  from langchain_core.messages import HumanMessage, SystemMessage
  from langchain_core.output_parsers import StrOutputParser
  
  
  os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
  os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
  
  
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      handlers=[logging.FileHandler('../eval.log')],
      level=logging.INFO
  )
  
  
  @dataclass
  class EvalData:
      name : str
      in_cnt : int = 0
      not_in_cnt : int = 0
      preds : list = field(default_factory=list)
      labels : list = field(default_factory=list)
      not_in_texts : list = field(default_factory=list)
      eval : dict = field(default_factory=dict)
  
  def save_obj(obj, name):  
      """  
      将对象保存到文件  
      :param obj: 要保存的对象  
      :param name: 文件的名称（包括路径）  
      """  
      with open(name, 'wb') as f:  
          pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
  
  
  def load_obj(name):  
      """  
      从文件加载对象  
      :param name: 文件的名称（包括路径）  
      :return: 反序列化后的对象  
      """  
      with open(name, 'rb') as f:  
          return pickle.load(f)
  
  
  LABELS_DICT = {
      0: "Human Necessities",
      1: "Performing Operations; Transporting",
      2: "Chemistry; Metallurgy",
      3: "Textiles; Paper",
      4: "Fixed Constructions",
      5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
      6: "Physics",
      7: "Electricity",
      8: "General tagging of new or cross-sectional technology",
  }
  
  
  LABELS_NAME = [
      LABELS_DICT[i]
      for i in range(9)
  ]
  
  LABELS_2_IDS = {
      v : k
      for k, v in LABELS_DICT.items()
  }
  
  
  def compute_metrics(pred, label):
      res = {}
      accuracy = evaluate.load("accuracy")
      res.update(accuracy.compute(
              predictions=pred, 
              references=label
          ))
  
      precision = evaluate.load("precision")
      res.update(precision.compute(
              predictions=pred, 
              references=label,
              average="macro"
          ))
  
      recall = evaluate.load("recall")
      res.update(recall.compute(
              predictions=pred, 
              references=label,
              average="macro"
          ))
  
      f1 = evaluate.load("f1")
      res.update(f1.compute(
              predictions=pred, 
              references=label,
              average="macro"
          ))
      return res
  
  
  def eval(kw):
      eval_data = EvalData(name=kw)
      model = ChatOpenAI(
          api_key="0",
          base_url="http://localhost:8000/v1",
          temperature=0
      )
  
      valid_dataset = load_dataset(
          "json",
          data_files="../data/llm_valid.json"
      )["train"]
      # labels = valid_dataset["output"][:50]
      labels = valid_dataset["output"]
      
      eval_data.labels = labels
      
      parser = StrOutputParser()
      preds = []
      cnt = 0
      for item in tqdm(valid_dataset):
          cnt += 1
          messages = [
              SystemMessage(content=item['instruction']),
              HumanMessage(content=item['input']),
          ]
          chain = model | parser
          pred = chain.invoke(messages).strip()
          preds.append(pred)
          # if cnt == 50:
          #     break
      
      eval_data.preds = preds
  
      not_in_texts = []
      in_cnt = 0
      not_in_cnt = 0
  
      for pred in preds:
          if pred in LABELS_NAME:
              in_cnt += 1
          else:
              not_in_cnt += 1
              not_in_texts.append(pred)
      
      eval_data.in_cnt = in_cnt
      eval_data.not_in_cnt = not_in_cnt
      eval_data.not_in_texts = not_in_texts
      
      pred_num = [
          LABELS_2_IDS[pred] if pred in LABELS_NAME else random.choice(range(9))
          for pred in preds
      ]
      label_num = [
          LABELS_2_IDS[label]
          for label in labels
      ]
      
      eval_data.eval = compute_metrics(pred=pred_num, label=label_num)
      
      logging.info(f"in_cnt: {in_cnt}, not_in_cnt: {not_in_cnt}")
      logging.info(f"eval: {eval_data.eval}")
      
      # 推理结果保存
      save_obj(
              eval_data,
              f"../objs/{kw}.pkl"
          )
  
  
  if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="输入大模型名，开始推理")
      parser.add_argument("kw", help="目前部署的大模型名字")
      args = parser.parse_args()
      logging.info(args.kw)
      eval(args.kw)
  ```

* `see_result.ipynb`
导入保存到objs文件夹中的预测结果，并进行结果的渲染
最后结果如下图所示：

![result.png](https://i-blog.csdnimg.cn/direct/59ee747c45f84834a6afdeca41824e99.png)



各位读者在看完，训练脚本 `train.sh`， 部署和推理脚本 `eval.sh`，应该已经明白本项目大致流程。

一言以蔽之，就是使用 for 逐步训练、部署、评估。

若大家想复现本文实验，本项目已经Github开源，项目开源地址：[https://github.com/JieShenAI/csdn/blob/main/24/07/few_shot_sft/readme.md](https://github.com/JieShenAI/csdn/blob/main/24/07/few_shot_sft/readme.md)

> 应该与Bert文本分类进行对比，就可以明显看出大模型的few-shot能力，有读者感兴趣可以实现一下。