#!/home/batuhangundogdu/other_codes/hug/bin/python
import os
from pathlib import Path
from kws20download import download_and_extract
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import time
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
from KWSDML import DML, MaskedXentMarginLoss, get_batch, calculate_scores
    

def main():
      
    keywords = ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
            'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'UNKNOWN')   
    raw_folder = 'raw'
    output_folder = 'bert_features'
    model_address = 'juliensimon/wav2vec2-conformer-rel-pos-large-finetuned-speech-commands'
    student_input_shape = (128, 128)
    input_length = student_input_shape[0]*student_input_shape[1]
    feat_shape = 36
    
    feature_extractor = Wav2Vec2FeatureExtractor()
    model = AutoModelForAudioClassification.from_pretrained(model_address, output_hidden_states =True)
    model = model.cuda()
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    if not os.path.isfile('.dataset_downloaded'):
        print('Downloading and extracting KWS dataset')
        download_and_extract()
        Path('.dataset_downloaded').touch()
        #TODO : At this point download and segment other datasets like librispeech
        
    kws = []
    for i in range(len(model.config.id2label)):
        kws.append(model.config.id2label[i])
        
    if not os.path.isfile('.bert_features_extracted'): 
        print('Extracting Teacher Features')
        for label in kws:
            if os.path.isdir(os.path.join(raw_folder, label)):
                print(f'Extracting features for keyword : {label}')
                record_list = sorted(os.listdir(os.path.join(raw_folder, label)))
                time_s = time.time()
                if not os.path.isdir(os.path.join(output_folder,label)):
                    os.makedirs(os.path.join(output_folder,label))
                for record_name in tqdm(record_list):            
                    record_pth = os.path.join(raw_folder, label, record_name)
                    waveform, sampling_rate = sf.read(record_pth)
                    rec_name = record_name[:-4]
                    inputs = feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
                    input_values = inputs.input_values.cuda()
                    with torch.no_grad():
                        outputs = model(input_values)
                    # TODO: Currently, we are extracting logits but we can investigate other hidden-layer activations
                    outputs1 = outputs.logits.cpu().detach().numpy()
                    predicted_class_ids = torch.argmax(outputs.logits, dim=-1).item()
                    predicted_label = model.config.id2label[predicted_class_ids]
                    out_file_name = os.path.join(output_folder, label, rec_name + '.npy')
                    with open(out_file_name, 'wb') as f:
                        np.save(out_file_name, outputs1)
                print(f'Finished {label} in {time.time()-time_s} seconds')
        Path('.bert_features_extracted').touch()
        
    if not os.path.isfile('.student_and_DML_inputs_created'):
        print('Preparing dataset for DML')

        number_of_train_samples = [0 for label in keywords]
        number_of_test_samples = [0 for label in keywords]
        feat_folder = output_folder
        #count train-test first
        keyword_id = 0
        for label in os.listdir(feat_folder):      
            record_list = sorted(os.listdir(os.path.join(feat_folder, label)))
            if label in keywords:
                for r, record_name in enumerate(record_list):
                    if hash(record_name) % 10 < 9:
                        number_of_train_samples[keyword_id] += 1
                    else:
                        number_of_test_samples[keyword_id] += 1

                keyword_id += 1
            else:
                for r, record_name in enumerate(record_list):
                    if hash(record_name) % 10 < 9:
                        number_of_train_samples[-1] += 1
                    else:
                        number_of_test_samples[-1] += 1


        unknown_train_count, unknown_test_count = 0, 0

        unknown_train_class = np.full((number_of_train_samples[20], 1), 20, dtype=np.uint8)
        unknown_train_DML_data = np.empty((number_of_train_samples[20], feat_shape), dtype=np.float64)
        unknown_train_student_data = np.empty((number_of_train_samples[20], student_input_shape[0], student_input_shape[1]), dtype=np.float64)

        unknown_test_class = np.full((number_of_test_samples[20], 1), 20, dtype=np.uint8)
        unknown_test_DML_data = np.empty((number_of_test_samples[20], feat_shape), dtype=np.float64)
        unknown_test_student_data = np.empty((number_of_test_samples[20], student_input_shape[0], student_input_shape[1]), dtype=np.float64)

        keyword_id = 0
        first_sample = True
        for label in tqdm(os.listdir(feat_folder)):
            if label in keywords:
                record_list = sorted(os.listdir(os.path.join(feat_folder, label)))
                train_count,test_count = 0, 0
                train_class_ = np.full((number_of_train_samples[keyword_id], 1), keyword_id, dtype=np.uint8)
                train_DML_data_ = np.empty((number_of_train_samples[keyword_id], feat_shape), dtype=np.float64)
                train_student_data_ = np.empty((number_of_train_samples[keyword_id],
                                               student_input_shape[0], student_input_shape[1]), dtype=np.float64)
                test_class_ = np.full((number_of_test_samples[keyword_id], 1), keyword_id, dtype=np.uint8)
                test_DML_data_ = np.empty((number_of_test_samples[keyword_id], feat_shape), dtype=np.float64)
                test_student_data_ = np.empty((number_of_test_samples[keyword_id],
                                               student_input_shape[0], student_input_shape[1]), dtype=np.float64)

                for r, record_name in enumerate(record_list):
                    record_pth = os.path.join(feat_folder, label, record_name)
                    rec_name = record_name[:-4]
                    audio_record_name = rec_name + '.wav'
                    audio_path = os.path.join(raw_folder, label, audio_record_name)
                    waveform, sampling_rate = sf.read(audio_path) 
                    if len(waveform) > input_length:
                        waveform = waveform[:input_length]
                    elif len(waveform) < input_length:
                        waveform = np.pad(waveform, (0, max(0, input_length - len(waveform))), "constant")
                    record = np.squeeze(np.load(record_pth))      
                    if hash(record_name) % 10 < 9:       
                        train_DML_data_[train_count, :] = record
                        train_student_data_[train_count, :, :] = np.reshape(waveform, student_input_shape)
                        train_count += 1
                    else:
                        test_DML_data_[test_count, :] = record
                        test_student_data_[test_count, :, :] = np.reshape(waveform, student_input_shape)
                        test_count += 1

                keyword_id += 1
                if first_sample:

                    train_class = train_class_.copy()
                    train_DML_data = train_DML_data_.copy()
                    train_student_data = train_student_data_.copy()
                    test_class = test_class_.copy()
                    test_DML_data = test_DML_data_.copy()
                    test_student_data = test_student_data_.copy()
                    first_sample = False
                else:

                    train_class = np.concatenate((train_class, train_class_), axis=0) 
                    train_DML_data = np.concatenate((train_DML_data, train_DML_data_), axis=0)
                    train_student_data = np.concatenate((train_student_data, train_student_data_), axis=0)
                    test_class = np.concatenate((test_class, test_class_), axis=0)     
                    test_DML_data = np.concatenate((test_DML_data, test_DML_data_), axis=0)
                    test_student_data = np.concatenate((test_student_data, test_student_data_), axis=0)

            else:
                # TODO: Prepare dataset for the E2E training
                record_list = sorted(os.listdir(os.path.join(feat_folder, label)))
                for r, record_name in enumerate(record_list):
                    record_pth = os.path.join(feat_folder, label, record_name)   
                    rec_name = record_name[:-4]
                    audio_record_name = rec_name + '.wav'
                    audio_path = os.path.join(raw_folder, label, audio_record_name)
                    waveform, sampling_rate = sf.read(audio_path) 
                    if len(waveform) > input_length:
                        waveform = waveform[:input_length]
                    elif len(waveform) < input_length:
                        waveform = np.pad(waveform, (0, max(0, input_length - len(waveform))), "constant")
                    record = np.squeeze(np.load(record_pth))      
                    if hash(record_name) % 10 < 9:
                        unknown_train_DML_data[unknown_train_count, :] = record
                        unknown_train_student_data[unknown_train_count, :, :] = np.reshape(waveform, student_input_shape)
                        unknown_train_count += 1

                    else:
                        unknown_test_DML_data[unknown_test_count, :] = record
                        unknown_test_student_data[unknown_test_count, :, :] = np.reshape(waveform, student_input_shape)
                        unknown_test_count += 1

        # TODO: save once done
        train_class = np.concatenate((train_class, unknown_train_class), axis=0)          
        train_DML_data = np.concatenate((train_DML_data, unknown_train_DML_data), axis=0)
        train_student_data = np.concatenate((train_student_data, unknown_train_student_data), axis=0)
        test_class = np.concatenate((test_class, unknown_test_class), axis=0)  
        test_DML_data = np.concatenate((test_DML_data, unknown_test_DML_data), axis=0)
        test_student_data = np.concatenate((test_student_data, unknown_test_student_data), axis=0)

        variable_address = 'student_and_DML_inputs.npz'
        np.savez(variable_address,
                 train_class=train_class,
                 train_DML_data=train_DML_data,
                 train_student_data=train_student_data, 
                 test_class=test_class,
                 test_DML_data=test_DML_data,
                 test_student_data=test_student_data)
        Path('.student_and_DML_inputs_created').touch()
        
    else:
        print('Loading the data')
        data = np.load('student_and_DML_inputs.npz', allow_pickle=True)
        train_class = data.f.train_class
        train_DML_data = data.f.train_DML_data
        train_student_data = data.f.train_student_data 
        test_class = data.f.test_class
        test_DML_data = data.f.test_DML_data
        test_student_data = data.f.test_student_data

            
    classes = set(np.squeeze(train_class)) 
    sigma = DML()
    sigma.double().cuda().train()
    optimizer = torch.optim.Adam(sigma.parameters(), lr=0.0003)
    optimizer.zero_grad()
    loss_fn = MaskedXentMarginLoss(margin=0.95)
    print('Training DML')
    for i in tqdm(range(5001)):
        anchor, alien, labels = get_batch(train_DML_data, train_class, batch_size=8)
        optimizer.zero_grad()
        output = sigma.forward(anchor, alien)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        if not i%1000:
            test_DML_embeddings = sigma.forward_one(torch.from_numpy(test_DML_data).cuda()).detach().cpu().numpy()
            train_DML_embeddings = sigma.forward_one(torch.from_numpy(train_DML_data).cuda()).detach().cpu().numpy()
            # TODO: Are the medoids really the best way to go?
            medoids = np.empty((train_DML_embeddings.shape[1], len(classes)), dtype=np.float32)
            for _class in classes:
                class_inx = (np.squeeze(train_class) == _class)
                class_samples = train_DML_embeddings[class_inx,:]
                medoids[:,_class] = np.mean(class_samples, axis=0)    
            _nn = calculate_scores(test_DML_embeddings, medoids, test_class)
    print("{:.2f}".format(_nn*100))
    variable_address = 'student_and_DML_inputs.npz'
    print('Saving embeddings')
    np.savez(variable_address,
             train_class=train_class,
             train_DML_data=train_DML_data,
             train_student_data=train_student_data, 
             test_class=test_class,
             test_DML_data=test_DML_data,
             test_student_data=test_student_data,
             train_DML_embeddings=train_DML_embeddings,
             test_DML_embeddings=test_DML_embeddings)


            
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    print('Done!')   

        

