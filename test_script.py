

from data_loader_test import PoseDataset
from utils.datasetv2 import PoseDatasetV2
import torch
import os
from torch.utils.data import  DataLoader
from utils.decode import Decode
import os
import torch
import shutil
from models.transformer import CSLRTransformer

import pandas as pd
import torch
from utils.metrics import normalize_gloss_sequence
from tqdm import tqdm
import argparse

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    """
    Decodes CTC predictions into gloss sequences.
    - Converts logits to probabilities (softmax).
    - Gets the highest probability gloss at each timestep.
    - Converts numerical predictions back into gloss words.
    - Removes duplicate glosses.
    """
    preds = torch.softmax(preds, 2)  # Convert logits to probabilities
    preds = torch.argmax(preds, 2)  # Get most likely gloss per frame
    preds = preds.detach().cpu().numpy()  # Convert to NumPy

    sign_preds = []
    for j in range(preds.shape[0]):  # Iterate over batch
        temp = []
        for k in preds[j, :]:
            k = k - 1  # Shift index to match vocabulary
            if k == -1:
                temp.append("§")  # Placeholder for blank (CTC)
            else:
                p = encoder.inverse_transform([k])[0]  # Convert number to gloss
                temp.append(p)

        gloss_seq = " ".join(temp).replace("§", "")  # Remove blank characters
        sign_preds.append(remove_duplicates(gloss_seq))  # Remove duplicate glosses

    return sign_preds

def numerize(sents, vocab_map, full_transformer):
    """
    Converts gloss sequences into numerical format.
    """
    outs = []
    for sent in sents:
        if type(sent) != float:
            if full_transformer:
                outs.append([32] + [vocab_map[g] for g in sent.split()] + [0])  # Add BOS and EOS
            else:
                outs.append([vocab_map[g] for g in sent.split()])
    return outs

def invert_to_chars(sents, inv_ctc_map):
    sents = sents.detach().numpy()
    outs = []
    for sent in sents:
        for x in sent:
            if x == 0:
                break
            outs.append(inv_ctc_map[x]) 
    return outs

def get_ctc_vocab(char_list):
    # blank
    ctc_char_list = "_" + char_list
    ctc_map, inv_ctc_map = {}, {}
    for i, char in enumerate(ctc_char_list):
        ctc_map[char] = i
        inv_ctc_map[i] = char
    return ctc_map, inv_ctc_map, ctc_char_list

def get_autoreg_vocab(char_list):
    # blank
    ctc_map, inv_ctc_map = {}, {}
    for i, char in enumerate(char_list):
        ctc_map[char] = i
        inv_ctc_map[i] = char
    return ctc_map, inv_ctc_map, char_list


def convert_text_for_ctc(dataset_name, train_csv, dev_csv, test_csv):
    """
    Reads CSLR annotation CSVs, extracts vocabulary, and encodes annotations for CTC training.

    Args:
        train_csv (str): Path to training annotation file.
        dev_csv (str): Path to development annotation file.
        test_csv (str): Path to testing annotation file.

    Returns:
        tuple: (Processed DataFrames, vocab_map, inv_vocab_map, vocab_list)
    """

    # Load all CSVs
    train_data = pd.read_csv(train_csv, delimiter="|")
    dev_data = pd.read_csv(dev_csv, delimiter="|")
    test_data = pd.read_csv(test_csv, delimiter="|")

    # Concatenate all data
    all_data = pd.concat([train_data, dev_data, test_data])

    # Remove rows where filename or annotation is missing
    if "isharah" in dataset_name.lower() or "csl" in dataset_name.lower():
        
        all_data = all_data[all_data['id'].notna()]
        
        #all_data = all_data[all_data['annotation'].notna()]
        all_data = all_data[all_data['gloss'].notna()]

        # Extract all glosses and remove duplicates
        all_glosses = set()
        for annotation in all_data["gloss"]:
            annotation = normalize_gloss_sequence(annotation)
            glosses = annotation.split()  # Split into words
            all_glosses.update(glosses)  # Add unique glosses

        # Create vocabulary mappings
        vocab_list = ["_"] + sorted(all_glosses)  # Ensure "_" is at index 0
        vocab_map = {g: i for i, g in enumerate(vocab_list)}  # "_": 0, "HELLO": 1, "WORLD": 2
        inv_vocab_map = {i: g for i, g in enumerate(vocab_list)}

        print(f"Extracted Vocabulary Size: {len(vocab_map)}")

        # Function to encode annotations into numerical format
        def encode_annotations(df):
            df = df.copy()
         #   print(df)
            # Apply normalization to the annotation string
            df["gloss"] = df["gloss"].apply(normalize_gloss_sequence)
            df["enc"] = df["gloss"].apply(lambda x: [vocab_map[g] for g in x.split()])  # Convert glosses to numbers
            return df[["id", "enc"]]  # Keep only necessary columns
        
    else: #id|folder|signer|annotation
        all_data = all_data[all_data['id'].notna()]
        
        #all_data = all_data[all_data['annotation'].notna()]
        all_data = all_data[all_data['annotation'].notna()]

        # Extract all glosses and remove duplicates
        all_glosses = set()
        for annotation in all_data["annotation"]:
            annotation = normalize_gloss_sequence(annotation)
            glosses = annotation.split()  # Split into words
            all_glosses.update(glosses)  # Add unique glosses

        # Create vocabulary mappings
        vocab_list = ["_"] + sorted(all_glosses)  # Ensure "_" is at index 0
        vocab_map = {g: i for i, g in enumerate(vocab_list)}  # "_": 0, "HELLO": 1, "WORLD": 2
        inv_vocab_map = {i: g for i, g in enumerate(vocab_list)}

        print(f"Extracted Vocabulary Size: {len(vocab_map)}")

        # Function to encode annotations into numerical format
        def encode_annotations(df):
            df = df.copy()
            print(df)
            # Apply normalization to the annotation string
            df["annotation"] = df["annotation"].apply(normalize_gloss_sequence)
            df["enc"] = df["annotation"].apply(lambda x: [vocab_map[g] for g in x.split()])  # Convert glosses to numbers
            return df[["id", "enc"]]  # Keep only necessary columns
        

    # Process train/dev/test separately
    train_processed = encode_annotations(train_data)
    print("processsed train")
    dev_processed = encode_annotations(dev_data)
    print("processsed dev")
    test_processed = encode_annotations(test_data)
    print("processsed test")

    return train_processed, dev_processed, test_processed, vocab_map, inv_vocab_map, vocab_list


def main(args):
    if os.path.exists(args.work_dir):
        answer = input('Current dir exists, do you want to remove and refresh it?\n')
        if answer in ['yes', 'y', 'ok', '1']:
            shutil.rmtree(args.work_dir)
            os.makedirs(args.work_dir)
    else:
        os.makedirs(args.work_dir)

    test_pkl_file_path = f"./annotations_v2/SI/pose_data_isharah1000_{args.mode}_test.pkl 

    gpu_id =0 # Change this to the GPU you want to use (0, 1, 2, etc.)
    num_workers = 10
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print ('device', device)

    train_csv = f"./annotations_v2/{args.mode}/train.txt"
    dev_csv = f"./annotations_v2/{args.mode}/dev.txt"
    test_csv = f"./annotations_v2/{args.mode}/{args.mode}_test.txt"

    _, _, test_processed, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc("isharah", train_csv, dev_csv, test_csv)
    print("Vocabulary: ", len(vocab_map))

    decoder_dec= Decode(vocab_map, len(vocab_list), 'beam')

    # CHANGE DATASET HERE ===============================================
    dataset_test = PoseDataset("isharah",test_pkl_file_path, test_csv , "test", test_processed, augmentations =False, additional_joints=args.additional_joints)
    testdataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers)
    # CHANGE DATASET HERE ===============================================


    # CHANGE MODEL HERE ===============================================
    model = CSLRTransformer(input_dim=86, num_classes=len(vocab_map)).to(device)
    msg = model.load_state_dict(torch.load(args.w_path))
    print("Model weights:", msg)
    # /CHANGE MODEL HERE ===============================================

    preds = []
    print("\n ***** Evaluation Test ******")
    predictions_file = f"{args.work_dir}/test.csv"
    with open(predictions_file, "w") as pred_file:
        pred_file.write(f"id,gloss\n")

        with torch.no_grad():
            for i, (video, poses, labels) in tqdm(enumerate(testdataloader), ncols=100, desc="Testing", total=len(testdataloader)):
                poses = poses.to(device)

                logits= model(poses)

                vid_lgt = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(device)
                decoded_list = decoder_dec.decode(logits, batch_first=True, probs=False, vid_lgt=vid_lgt)
                flat_preds = [gloss for pred in decoded_list for gloss, _ in pred]
                current_preds = ' '.join(flat_preds)
                preds.append(current_preds)

                pred_file.write(f"{i+1},{current_preds}\n")

    print(f"Saved final gloss predictions to: {predictions_file}")



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work_dir', dest='work_dir', default="./work_dir/test")
    parser.add_argument('--w_path', dest='w_path', required=True)
    parser.add_argument('--additional_joints', dest='additional_joints', default="1")
    parser.add_argument('--mode', dest='mode', default="SI")

    args=parser.parse_args()
    args.additional_joints = True if args.additional_joints == "1" else False
    
    main(args)
