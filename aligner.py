#!/home/batuhangundogdu/other_codes/hug/bin/python
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
import torch
from datasets import load_dataset, DatasetDict
from itertools import groupby
import logging
from tqdm import tqdm




def main():
    
    sampling_rate = 16_000
    word_dict = {}
    logging.basicConfig(filename='segments_2.csv', level=logging.INFO, format='%(message)s')
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.info(f'word, utt_id, start, dur')
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    libre_dataset = load_dataset("librispeech_asr", 'clean')
    total_duration = 0
    print(libre_dataset)
    
    for utterance_counter in range(len(libre_dataset['train.360'])):
        print(utterance_counter)
        x = libre_dataset['train.360'][utterance_counter]
        waveform = x['audio']['array']
        gold_label = x['text'].lower()
        input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        logits = model(**input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids = predicted_ids[0].tolist()
        transcription = processor.decode(predicted_ids).lower()
        if transcription == gold_label:
            transcription = [w for w in transcription.split(' ') if len(w) > 0]
            duration_sec = input_values.input_values.shape[1] / sampling_rate
            total_duration += duration_sec
            ids_w_time = [((i / len(predicted_ids)) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
            ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]
            split_ids_w_time = [list(group) for k, group
                            in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                            if not k]
            word_start_times = []
            word_end_times = []
            for cur_ids_w_time, cur_word in zip(split_ids_w_time, transcription):
                _times = [_time for _time, _id in cur_ids_w_time]
                word_start_times.append(min(_times))
                word_end_times.append(max(_times))
            for word, end, start in zip(transcription, word_end_times, word_start_times):
                if int(end*sampling_rate) - int(start*sampling_rate):
                    dur = int(end*sampling_rate) - int(start*sampling_rate)
                    logging.info(f'{word}, {utterance_counter}, {int(start*sampling_rate)}, {int(dur)}')
    

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    print('Done!')   