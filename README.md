# CSE4207_Summer26
This is the repository for recording the necessary code files for the redisign of the WashU CSE4207 Cloud Computing Lab 5.

## Progress
### Local Training Scripts
This part includes a simple machine learning pipeline such that it can use Polynomial Regression from **scikit-learn** library. It does both regular regression and classification tasks, tested by the Wine Quality Dataset and Iris Dataset from scikit-learn package. All of the functions are able to run in local environment without any cloud application included. So, it is more like a toy example.

### Docker Training
This part builds a simple Docker environment for a simple machine learning pipeline. The pipeline is built in a shell script file that will automatically run when the container is activated. This pipeline accepts data from **Amazon S3** buckets and will store the trained models and prediction results in the bucket as well. The main change is conternerization of the scripts.

### Kubernetes ML Pipeline
In this section, the whole pipeline is hosted on the **Kubernetes** network. Training and inference are considered as one-shot jobs because the nature of those scripts are just submission, running, finishing, and stroing if needed. Jobs are great fit for those kinds of work. Also, the persistent volume is set up by a persistent volume claim which mounts a storage space for the **trained models**.

In the first design, there was **StatefulSet** for model registry and sotrage. However, this application is not necessary and even useless. This pod only acts as an entry point to read what models are stored in the volume and what are the names of the models. A **Deployment** should satisfy this function. Also, this pod does not need to be stateful since there is no such **important memory**. So in later design, this section has been disgarded.

### Small Language Model (So far best)
Instead of doing a traditional machine learning lab, this version is more modern and interesting for catching what might be a good industrial standard illustration. It will create a network that supports fine-tuning of a pre-trained small language model (universal to all laptops) and inferring a question using the tuned model (even though the results are bad and ugly due to the model restrictions). The training is not considered in this lab since not all students have enoguh GPU resource to train a million-parameter-size model and it will so slow for the Kubernetes network (minikube) to do this work.

This system includes ***Deployment*** for **long-time inference** session (which needs more investigation to forward messages to the terminal), ***Job*** for **one-shot work**, ***Stateful Set*** for systemt and training logs, and ***PV/PVC*** for both **model storage** and **logs history**. A similar access control ***Role Based Access Control*** will be introduced to submit job from a pod to the Kubernetes network. This grants certain pods certain access to different applications like *IAM* does.

The intended workflow would be starting from locally worked scripts, first containerize the scripts and create an app for listening request. Then try to setup PV/PVC and PostgreSQL database on Kubernetes. Connect the app and api to the corresponding volume. At the end, test the integrity of the whole network so it can work smoothly.

The original system only supports job submission from the terminal directly. Later an interesting ***API #docs*** from **FastAPI** in developing AI agent comes to the front. It is a GUI that supports user to submit a web request to FastAPI app/api directly and receive the web response. This will help visualize the workflow and also edit request more quickly and easily. That inspired the creation api version of this lab.

The challenge will be at the design part and ochestration of different types of applications in Kubernetes.

See more details in the Word document.

### Naive Bayes Classifier
This is only a toy example of the experiment system which classifies some text and record the accuracy using **Naive Bayes Classifier**. However, this suggestion is too simple and can be included in other parts. There is no further investigation in this direction.

### AI Agent
This part is the most popular and also up-to-date application using cloud computing. It will create an intelligent AI agent to help do the model fine tuning and inference job. The workflow is very similar to Small Language Model lab but there is ***Deployment*** required for this. The overall architecture is pretty clear with simple API listening for incoming prompt, AI agnet parsing the request and give back job detail to submit, Kubernetes executing the job and returning the logs, and backend responding to the user directly through web packages.

The AI agent model is chosen as ***Llama_3 2-Billion instruct version*** from Meta. The previous diluted GPT2 is too simple to complete the request. However, there are two major problem preventing this lab from success. 

The first problem is the model size. The Llama_3 is too large for Kubernetes network to even load this model from the HuggingFace. Ideally, this AI agent should be hosted in the Kubernetes. The compromise way is to host this API in local network and ask Kubernetes pod to submit the prompt to the local network. This will work and has been tested locally. 

The second problem is more fatal regarding to the model selection. The middle-sized model like Llama3 is also not intelligent enough to handle all the incoming prompt while fine-tuning a large language model locally is not practical for all students.

Therefore, this has been moved to the challenge problem in the Small Language Model proposal which students are freely to explore the AI agent if their computation resource is enough. But it doesn't require all students to finish this part.

## APIs

The initial Small Language Model command-line scripts are located in
`checkpoints/1_initial_scripts/machine_learning/`. They operate entirely on local files:
`fine-tune.py` fine-tunes `distilgpt2` on the CPU and saves it to a local directory, while
`predict.py` loads that saved model and generates answers.

### `fine-tune.py`

```bash
python3 checkpoints/1_initial_scripts/machine_learning/fine-tune.py \
  --dataset_location path/to/qa_training.txt \
  --model_storage_location path/to/saved_model \
  --epochs 3
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--dataset_location` | Yes | None | Path to a local UTF-8 text file containing the training examples. The script raises `Dataset file not found.` if the path does not exist. |
| `--model_storage_location` | Yes | None | Local directory where the fine-tuned model and tokenizer will be saved. The directory and any missing parent directories are created automatically. |
| `--epochs` | No | `1` | Number of complete passes through the training dataset. The value must be an integer. |

The training file must contain one or more pairs in this format:

```text
Question: What is cloud computing?
Answer: Cloud computing provides computing resources over a network.
```

The script parses the labeled pairs, limits each example to 128 tokens, masks the
question so that loss is calculated only from the answer, and trains with a batch size of
1 and a learning rate of `5e-5`. It prints the average loss after each epoch.

### `predict.py`

```bash
python3 checkpoints/1_initial_scripts/machine_learning/predict.py \
  --model_location path/to/saved_model \
  --question "What is cloud computing?"
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--model_location` | Yes | None | Local directory containing the saved Hugging Face model and tokenizer. The script raises `Model not found.` if the path is missing or is not a directory. |
| `--question` | No | None | A single question to send to the model. When supplied, the script prints one answer and exits. |

If `--question` is omitted, the script starts an interactive session instead:

```bash
python3 checkpoints/1_initial_scripts/machine_learning/predict.py \
  --model_location path/to/saved_model
```

The interactive mode repeatedly prompts for questions until the user enters `exit`.
Question generation is limited to 256 total tokens. The generated text is normalized by replacing repeated
whitespace with single spaces before it is displayed.
