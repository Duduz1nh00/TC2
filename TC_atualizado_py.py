{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNDneeQeKOeIYH8iIVdL8Cd",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Duduz1nh00/TC2/blob/main/TC_atualizado_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "skCIznF89PDL"
      },
      "outputs": [],
      "source": [
        "!pip install datasets\n",
        "import datasets\n",
        "import wandb\n",
        "import random"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import json\n",
        "\n",
        "# Carregar o dataset trn.json\n",
        "with open(\"part.json\", \"r\") as file:\n",
        "    data = json.load(file)\n",
        "\n",
        "# Criar um DataFrame com as colunas relevantes\n",
        "df = pd.DataFrame(data)\n",
        "df = df[[\"title\", \"content\"]].dropna()\n",
        "\n",
        "# Exemplo de formatação de prompt\n",
        "df[\"prompt\"] = \"Qual é a descrição do produto chamado '\" + df[\"title\"] + \"'?\"\n",
        "df[\"response\"] = df[\"content\"]\n",
        "\n",
        "# Divisão do dataset\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "train, test = train_test_split(df, test_size=0.2, random_state=42)\n",
        "val, test = train_test_split(test, test_size=0.5, random_state=42)\n",
        "\n",
        "# Salvar os conjuntos preparados\n",
        "train.to_json(\"train.json\", orient=\"records\", lines=True)\n",
        "val.to_json(\"val.json\", orient=\"records\", lines=True)\n",
        "test.to_json(\"test.json\", orient=\"records\", lines=True)"
      ],
      "metadata": {
        "id": "IiCJlb1c9T85"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer\n",
        "from datasets import load_dataset"
      ],
      "metadata": {
        "id": "XNO5s7I79o53"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#2. Testando o Foundation Model (Antes do Fine-Tuning)\n",
        "#Carregar e testar um modelo pré-treinado:\n",
        "\n",
        "from transformers import pipeline\n",
        "\n",
        "# Escolher o modelo\n",
        "model = pipeline(\"text2text-generation\", model=\"google/flan-t5-base\")\n",
        "\n",
        "# Exemplo de teste\n",
        "prompt = \"Qual é a descrição do produto chamado 'Smartphone X'?\"\n",
        "print(model(prompt))\n",
        "#3. Fine-Tuning do Modelo\n",
        "#Configuração e treinamento com Hugging Face:"
      ],
      "metadata": {
        "id": "9ecswMfZ9ZhO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Carregar o modelo e tokenizer\n",
        "model_name = \"google/flan-t5-base\"\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
        "model = AutoModelForSeq2SeqLM.from_pretrained(model_name)\n",
        "\n",
        "# Carregar o dataset\n",
        "# Substitua \"your_dataset_name\" pelo nome do dataset ou o caminho para o seu dataset local\n",
        "dataset = load_dataset(\"json\", data_files={\"train\": \"train.json\", \"validation\": \"val.json\"})\n",
        "\n",
        "# Pré-processamento dos dados\n",
        "def preprocess_data(examples):\n",
        "    # Obter inputs (prompts) e targets (respostas)\n",
        "    inputs = examples[\"prompt\"]\n",
        "    targets = examples[\"response\"]\n",
        "\n",
        "    # Tokenizar os inputs\n",
        "    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=\"max_length\")\n",
        "\n",
        "    # Tokenizar os rótulos (targets)\n",
        "    with tokenizer.as_target_tokenizer():\n",
        "        labels = tokenizer(targets, max_length=512, truncation=True, padding=\"max_length\")\n",
        "\n",
        "    # Adicionar rótulos aos inputs\n",
        "    model_inputs[\"labels\"] = labels[\"input_ids\"]\n",
        "    return model_inputs\n",
        "\n",
        "# Aplicar o pré-processamento ao dataset\n",
        "tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=dataset[\"train\"].column_names)\n",
        "\n",
        "# Configuração do treinamento\n",
        "training_args = Seq2SeqTrainingArguments(\n",
        "    output_dir=\"./results\",  # Pasta para salvar os resultados\n",
        "    evaluation_strategy=\"epoch\",  # Avaliar a cada época\n",
        "    learning_rate=5e-5,  # Taxa de aprendizado\n",
        "    per_device_train_batch_size=8,  # Tamanho do lote para treinamento\n",
        "    per_device_eval_batch_size=8,  # Tamanho do lote para avaliação\n",
        "    num_train_epochs=3,  # Número de épocas\n",
        "    weight_decay=0.01,  # Decaimento de peso\n",
        "    save_total_limit=2,  # Limite de checkpoints salvos\n",
        "    predict_with_generate=True,  # Geração durante a avaliação\n",
        "    fp16=True,  # Usar precisão mista para acelerar na GPU\n",
        ")\n",
        "\n",
        "# Criar o Trainer\n",
        "trainer = Seq2SeqTrainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=tokenized_datasets[\"train\"],  # Dataset de treinamento\n",
        "    eval_dataset=tokenized_datasets[\"validation\"],  # Dataset de validação\n",
        "    tokenizer=tokenizer,\n",
        ")\n",
        "\n",
        "# Executar o treinamento\n",
        "trainer.train()\n",
        "\n",
        "# Salvar o modelo treinado\n",
        "trainer.save_model(\"fine_tuned_model\")\n",
        "\n",
        "# Salvar o tokenizer (opcional)\n",
        "tokenizer.save_pretrained(\"fine_tuned_model\")"
      ],
      "metadata": {
        "id": "M9G8VFtu9u_j"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "IX8i62tV-Jvj"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}