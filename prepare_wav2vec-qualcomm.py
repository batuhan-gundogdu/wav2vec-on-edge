#!/home/batuhangundogdu/other_codes/hug/bin/python
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import fairseq
import os
import time
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torch


def main():
    
    raw_folder = '/home/batuhangundogdu/qualcomm_keyword_speech_dataset'
    keywords = [x for x in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, x))]
    
    model_type = 'transformer'#options are : 'transformer', 'wav2vec'
    output_folder = 'qualcomm-wav2vec2' # Change this everytime
    
    if model_type == 'transformer':
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-960h", output_hidden_states=True)
        layer = 16
    elif model_type == 'wav2vec':
        model_path = '/home/batuhangundogdu/wav2vec_large.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        model = model[0]
        
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
                if model_type == 'transformer':
                    input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
                    outputs = model(**input_values).hidden_states[layer].mean(dim=1)
                    outputs = outputs.cpu().detach().numpy()
                elif model_type == 'wav2vec':
                    model.eval()
                    input_values = torch.from_numpy(waveform).unsqueeze(dim=0).float()
                    input_values = input_values
                    z = model.feature_extractor(input_values)
                    #c = model.feature_aggregator(z)
                    outputs = z.mean(dim=-1).cpu().detach().numpy()
                    
                out_file_name = os.path.join(output_folder, label, rec_name + '.npy')
                with open(out_file_name, 'wb') as f:
                    np.save(out_file_name, outputs)


    
if __name__ == "__main__":
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    print('Done!')   
