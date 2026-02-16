<h1 style="color:red">New annotation file (test.csv) added to annotations_v2 folder</h1>


# Pose Estimation for MSLR CSLR Track

Welcome to the Pose Estimation repository! This repository contains the starter kit for the **MSLR CSLR Track** and provides a simple baseline for two important tasks in Continuous Sign Language Recognition (CSLR).

The tasks include:
1. **Signer Independent** [View Competition](https://www.codabench.org/competitions/13266/)
2. **Unseen Sentences** [View Competition](https://www.codabench.org/competitions/13267/)


<!-- ## Updates
* Test Set released (Feb 15, 2026) -->

<!-- #### Instructions for the Test Set
1. Download `test_script.py` and `data_loader_test.py` and place in the root directory of your project
2. Download SI_test.txt or US_test.txt from `./annotations_v2/SI` or `./annotations_v2/US` directory respectively.
   - For **Signer Independent** task, place it in `./annotations_v2/SI`.
   - For **Unseen Sentences** task, place it in `./annotations_v2/US`.

3. Download the test data from CodaLab and place in `./annotations_v2/SI` or `./annotations_v2/US` depending on the task.

4. Change the model in ```test_script.py``` on line 229 to your model.

5. Run the test script:
   ```bash
   python test_script.py --work_dir ./work_dir/test --w_path <path to your model weights> --mode SI
   ``` -->


## Baseline Overview

We use a simple **Transformer model** to produce the baseline for these tasks. The architecture of the baseline model is shown below:

![Baseline](fig/transformer.png)

| Task              | Baseline Dev (WER) |
|-------------------|----------|
| **Signer Independent** | 20.12% |
| **Unseen Sentences**    | 80.31% |

## Setup Instructions

Follow these steps to set up the environment and get started:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gufranSabri/Pose86K-CSLR-Isharah.git
   cd Pose86K-CSLR-Isharah
   ```

2. **Download the dataset** from [TASK 1](https://www.kaggle.com/datasets/gufransabri3/mslr-task1); [TASK 2](https://www.kaggle.com/datasets/gufransabri3/mslr-task2). Place the dataset in the `./data` folder.

3. **Set up the Python environment**:
   - Install `virtualenv`:
     ```bash
     pip install virtualenv
     ```

   - Create a virtual environment and activate it:
     ```bash
     python<version> -m venv pose
     source pose/bin/activate  # On Windows: pose\Scriptsctivate
     ```

   - Install the required dependencies:
     ```bash
     pip install torch==1.13 torchvision==0.14 tqdm numpy==1.23.5 pandas opencv-python
     git clone --recursive https://github.com/parlance/ctcdecode.git
     cd ctcdecode && pip install .
     ```


## Running the Model
Once your environment is ready and the data is in place, you can run the main script using the following format:
```
python main.py \
  --work_dir ./work_dir/test \
  --data_dir ./data \
  --mode SI \
  --model base \
  --device 0 \
  --lr 0.0001 \
  --num_epochs 300
```

### Argument Descriptions
 * ```--work_dir:``` Path to store logs and model checkpoints (default: ./work_dir/test)
 * ```--data_dir:``` Path to the dataset directory (default:``` /data/sharedData/Smartphone/)
 * ```--mode:``` Task mode, either SI (Signer Independent) or US (Unseen Sentences)
 * ```--model:``` Model variant to use (base, or any other available variant)
 * ```--device:``` GPU device index (default: 0)
 * ```--lr:``` Learning rate (default: 0.0001)
 * ```--num_epochs:``` Number of training epochs (default: 300)

You can modify these arguments as needed for your experiments.

### Example Command
```
python main.py --work_dir ./work_dir/base_US --model base --mode US
```

## Usage

Once the environment is set up, you can train or test the model on the available tasks. Follow the instructions in the individual task directories for specific commands.

## License

This project is licensed under the MIT License.
