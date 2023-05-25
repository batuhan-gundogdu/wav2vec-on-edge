#!/home/batuhangundogdu/other_codes/hug/bin/python
from config import *
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
import time
import soundfile as sf
import numpy as np
from datasets import load_dataset


def chop_into_pieces(data, piece_length=24596):

    dataX = np.empty((max(len(data)//piece_length,1),piece_length))
    if len(data) < piece_length:
        dataX[0,:len(data)] = data
    else:
        for i in range(dataX.shape[0]):
            dataX[i] = data[i*piece_length : i*piece_length + piece_length]
        dataX = dataX.reshape(-1, piece_length)
    return dataX

def main():
    
    
    keywords = [x for x in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, x))]
    processor = Wav2Vec2Processor.from_pretrained(teacher_model)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(teacher_model, output_hidden_states=True)
    
    
    for label in keywords:
        print(f'Extracting features for keyword : {label}')
        speaker_list = sorted(os.listdir(os.path.join(raw_folder, label)))
        if not os.path.isdir(os.path.join(output_folder, label)):
            os.makedirs(os.path.join(output_folder,label))           
        for speaker in tqdm(speaker_list):
            record_list = sorted(os.listdir(os.path.join(raw_folder, label, speaker)))
            time_s = time.time()
            for record_name in record_list:
                record_pth = os.path.join(raw_folder, label, speaker, record_name)
                waveform, sampling_rate = sf.read(record_pth)
                rec_name = record_name[:-4]
                input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
                outputs = model(**input_values).hidden_states[teacher_layer].mean(dim=1)
                outputs = outputs.cpu().detach().numpy()
                    
                out_file_name = os.path.join(output_folder, label, rec_name + '.npy')
                with open(out_file_name, 'wb') as f:
                    np.save(out_file_name, outputs)
                    
    print('Now processing some background speech')
    if not os.path.exists(os.path.join(output_folder, 'background')):
        os.mkdir(os.path.join(output_folder, 'background'))
        
    merged_test_audio = []
    background = load_dataset("librispeech_asr", 'clean')    
    for i in range(len(background['validation'])):
        x = background['validation'][i]
        merged_test_audio.append(x['audio']['array'])
   
    for i in range(len(background['test'])):
        x = background['test'][i]
        merged_test_audio.append(x['audio']['array'])
        
        
    merged_test_audio = np.concatenate(merged_test_audio, axis=0)
    print(merged_test_audio.shape)
    chopped_background = chop_into_pieces(merged_test_audio)
    
    for i in tqdm(range(chopped_background.shape[0])):
        waveform = chopped_background[i]    
        input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        outputs = model(**input_values).hidden_states[teacher_layer].mean(dim=1)
        outputs = outputs.cpu().detach().numpy()
        out_file_name = os.path.join(output_folder, 'background', str(i) + '.npy')
        with open(out_file_name, 'wb') as f:
            np.save(out_file_name, outputs)


    
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    print('Done!')   
