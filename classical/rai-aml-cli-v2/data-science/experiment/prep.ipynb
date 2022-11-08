{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false,
          "source_hidden": false
        },
        "nteract": {
          "transient": {
            "deleting": false
          }
        }
      },
      "outputs": [],
      "source": [
        "import argparse\n",
        "\n",
        "from pathlib import Path\n",
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "import mlflow"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {},
      "outputs": [],
      "source": [
        "TARGET_COL = \"cost\"\n",
        "\n",
        "NUMERIC_COLS = [\n",
        "    \"distance\",\n",
        "    \"dropoff_latitude\",\n",
        "    \"dropoff_longitude\",\n",
        "    \"passengers\",\n",
        "    \"pickup_latitude\",\n",
        "    \"pickup_longitude\",\n",
        "    \"pickup_weekday\",\n",
        "    \"pickup_month\",\n",
        "    \"pickup_monthday\",\n",
        "    \"pickup_hour\",\n",
        "    \"pickup_minute\",\n",
        "    \"pickup_second\",\n",
        "    \"dropoff_weekday\",\n",
        "    \"dropoff_month\",\n",
        "    \"dropoff_monthday\",\n",
        "    \"dropoff_hour\",\n",
        "    \"dropoff_minute\",\n",
        "    \"dropoff_second\",\n",
        "]\n",
        "\n",
        "CAT_NOM_COLS = [\n",
        "    \"store_forward\",\n",
        "    \"vendor\",\n",
        "]\n",
        "\n",
        "CAT_ORD_COLS = [\n",
        "]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false,
          "source_hidden": false
        },
        "nteract": {
          "transient": {
            "deleting": false
          }
        }
      },
      "outputs": [],
      "source": [
        "# Define Arguments for this step\n",
        "\n",
        "class MyArgs:\n",
        "    def __init__(self, /, **kwargs):\n",
        "        self.__dict__.update(kwargs)\n",
        "\n",
        "args = MyArgs(\n",
        "            raw_data = \"../../data/\", \n",
        "            train_data = \"/tmp/prep/train\",\n",
        "            val_data = \"/tmp/prep/val\",\n",
        "            test_data = \"/tmp/prep/test\",\n",
        "            )\n",
        "\n",
        "os.makedirs(args.train_data, exist_ok = True)\n",
        "os.makedirs(args.val_data, exist_ok = True)\n",
        "os.makedirs(args.test_data, exist_ok = True)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "def main(args):\n",
        "    '''Read, split, and save datasets'''\n",
        "\n",
        "    # ------------ Reading Data ------------ #\n",
        "    # -------------------------------------- #\n",
        "\n",
        "    print(\"mounted_path files: \")\n",
        "    arr = os.listdir(args.raw_data)\n",
        "    print(arr)\n",
        "\n",
        "    data = pd.read_csv((Path(args.raw_data) / 'taxi-data.csv'))\n",
        "    data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]\n",
        "\n",
        "    # ------------- Split Data ------------- #\n",
        "    # -------------------------------------- #\n",
        "\n",
        "    # Split data into train, val and test datasets\n",
        "\n",
        "    random_data = np.random.rand(len(data))\n",
        "\n",
        "    msk_train = random_data < 0.7\n",
        "    msk_val = (random_data >= 0.7) & (random_data < 0.85)\n",
        "    msk_test = random_data >= 0.85\n",
        "\n",
        "    train = data[msk_train]\n",
        "    val = data[msk_val]\n",
        "    test = data[msk_test]\n",
        "\n",
        "    mlflow.log_metric('train size', train.shape[0])\n",
        "    mlflow.log_metric('val size', val.shape[0])\n",
        "    mlflow.log_metric('test size', test.shape[0])\n",
        "\n",
        "    train.to_parquet((Path(args.train_data) / \"train.parquet\"))\n",
        "    val.to_parquet((Path(args.val_data) / \"val.parquet\"))\n",
        "    test.to_parquet((Path(args.test_data) / \"test.parquet\"))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false,
          "source_hidden": false
        },
        "nteract": {
          "transient": {
            "deleting": false
          }
        }
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Raw data path: ../../data/\n",
            "Train dataset output path: /tmp/prep/train\n",
            "Val dataset output path: /tmp/prep/val\n",
            "Test dataset path: /tmp/prep/test\n",
            "mounted_path files: \n",
            "['taxi-batch.csv', 'taxi-data.csv', 'taxi-request.json']\n"
          ]
        }
      ],
      "source": [
        "mlflow.start_run()\n",
        "\n",
        "lines = [\n",
        "    f\"Raw data path: {args.raw_data}\",\n",
        "    f\"Train dataset output path: {args.train_data}\",\n",
        "    f\"Val dataset output path: {args.val_data}\",\n",
        "    f\"Test dataset path: {args.test_data}\",\n",
        "\n",
        "]\n",
        "\n",
        "for line in lines:\n",
        "    print(line)\n",
        "\n",
        "main(args)\n",
        "\n",
        "mlflow.end_run()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "jupyter": {
          "outputs_hidden": false,
          "source_hidden": false
        },
        "nteract": {
          "transient": {
            "deleting": false
          }
        },
        "vscode": {
          "languageId": "shellscript"
        }
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            " Volume in drive C is Local Disk\n",
            " Volume Serial Number is 583C-74B4\n",
            "\n",
            " Directory of c:\\tmp\\prep\\train\n",
            "\n",
            "10/07/2022  12:08 AM    <DIR>          .\n",
            "10/07/2022  12:08 AM    <DIR>          ..\n",
            "10/07/2022  12:08 AM           277,190 train.parquet\n",
            "               1 File(s)        277,190 bytes\n",
            "               2 Dir(s)  788,218,421,248 bytes free\n"
          ]
        }
      ],
      "source": [
        "ls \"/tmp/prep/train\" "
      ]
    }
  ],
  "metadata": {
    "kernel_info": {
      "name": "local-env"
    },
    "kernelspec": {
      "display_name": "Python 3.9.6 64-bit",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.6"
    },
    "nteract": {
      "version": "nteract-front-end@1.0.0"
    },
    "vscode": {
      "interpreter": {
        "hash": "c87d6401964827bd736fe8e727109b953dd698457ca58fb5acabab22fd6dac41"
      }
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
