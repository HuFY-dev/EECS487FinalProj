# In-Context Learning (Main Method)

## Notes

Used [`LLaMA-2-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) by [Meta](https://ai.meta.com/llama/) with the HuggingFace [`transformers`](https://huggingface.co/docs/transformers/index) Library. For optimization, HuggingFace [`accerelare`](https://huggingface.co/docs/accelerate/index) library was used to quantize the LLM. [`wandb`](https://wandb.ai/) is used for logging and monitoring performance. Used the [`SQuAD_v2`](https://huggingface.co/datasets/squad_v2) dataset with the HuggingFace [`datasets`](https://huggingface.co/docs/datasets/index) library

## Requirements

Run:

```bash
conda env create -f environment.yml
```

## Usage

Run `icl.py` with the following options:

Run with control group:

```bash
--control
```

Run with local models (If unspecified, the script will download the LLaMA-2 model from HuggingFace, and you need to login with a token generated from https://huggingface.co/settings/tokens):

```bash
--model_path <your path>
```

Parameters:
```bash
--batch_size <batch_size, default=8>
--num_shots <num_shots, default=2>
```

Debug mode:

```bash
--debug
```

# Initial Quiz Generation Notebook

## Overview
This Jupyter Notebook is designed for creating educational quizzes, focusing on question-answer formats. It leverages Python and various NLP techniques for generating different types of quiz questions.

## Features
- **Simple Question Generation**: Functions to convert text into basic questions (e.g., Who, What, When).
- **Fill-in-the-Blank Quiz Creation**: Using NLP libraries to create fill-in-the-blank style questions.
- **ECR (Entity, Concept, and Relation) Analysis**: An attempt to analyze text for entities, concepts, and their relations, useful for creating more contextually rich questions.
- **Seq2Seq Model Implementation**: Exploratory work on using sequence-to-sequence models for generating quiz questions, aiming at more complex and nuanced question formation.

## Requirements
- `spacy`: A popular library for advanced Natural Language Processing in Python.
- Other NLP and machine learning libraries as per the specific sections (like NLTK, TensorFlow, etc.)

## Usage
1. **Simple Question Generation**: This section contains a function `simple_question_generator` which takes a block of text and generates basic questions. The function identifies key words to form relevant questions.

   Example: 
   ```python
   summary = "The sun is a star at the center of our solar system."
   generated_questions = simple_question_generator(summary)
   ```

2. **Fill-in-the-Blank Quiz**: Utilizes NLP techniques for creating fill-in-the-blank questions.

3. **ECR Analysis**: This part includes attempts to extract entities, concepts, and relations from the text, which could potentially be used for creating more detailed and specific quiz questions.

4. **Seq2Seq Model**: Experimental implementation of sequence-to-sequence models for question generation, aiming to produce more complex and varied types of questions.

## Contribution
Contributions are welcome to enhance the question generation algorithms, expand the range of question types, and optimize the existing codebase.

# Initial Summarization Notebook

## Overview
This Jupyter Notebook explores the fine-tuning of the T5-Small model from Hugging Face using the CNN_DailyMail Dataset for text summarization tasks.

## Features
- **Traning Argument Selection**: Choose from a variety of hyperparameters to optimize the fine-tuning process, including learning rate, epochs, and batch size.
- **Summarization**:  Leverage the T5-Small model from Hugging Face, a pre-trained powerhouse for text summarization tasks. .
- **Rouge Score Clculation**: Analyze the generated summaries with established metrics like ROUGE, which measures similarity between the model's output and reference summaries. This provides a quantitative assessment of summarization accuracy and fluency.


## Requirements
- `torch`: Deep learning framework (>=1.13.1)
- `datasets`: Hugging Face Datasets library for loading and preparing datasets (>=4.28.0)
- `rouge`: ROUGE score calculation for evaluating summaries (>=1.2.2)
- `transformers`: Hugging Face Transformers library for accessing and fine-tuning pre-trained models (>=4.28.0)
- `accelerate`: Distributed GPU/TPU training library (optional, but recommended for large datasets/models)

## How to Use the Fine-tuned T5-Small Summarization Model
This section guides you through utilizing the fine-tuned T5-Small model for generating summaries of your text.

1. Installing Dependencies:

Ensure you have installed the required libraries mentioned in the "Requirements" section of this README.
If using a virtual environment, activate it before running commands.


Certainly! Here's a modified version of step 2:

2. Loading Custom Data (Optional):

The script is pre-configured with a default dataset: CNN-DailyMail. However, you can replace it with your own data for fine-tuning on specific topics or domains.
To achieve this, follow these steps:
Prepare your own dataset in a format compatible with the Hugging Face Datasets library.
Modify the script to specify the path to your dataset file or provide it as an argument during execution (e.g., --data_path=/path/to/your/dataset.json).
Run the script as usual with the updated configuration.

3. Generating the Output:

The script will print the generated summary to the cell. You can also redirect the output to a file or store it in a variable.

The script also allows you to compare the generated summary's quality with the original text using ROUGE scores. 

4. Credits: 
https://r.search.yahoo.com/_ylt=AwrFQnyDeHtl.z8Udd5XNyoA;_ylu=Y29sbwNiZjEEcG9zAzEEdnRpZAMEc2VjA3Ny/RV=2/RE=1702619396/RO=10/RU=https%3a%2f%2fhuggingface.co%2ft5-small/RK=2/RS=QEeGW_.9.co1t._aaWk90B22GmE-
