import os
from pathlib import Path
import numpy as np

import pisces

output_dir = Path(__file__).parent / 'preprocessed_data'
# change this for your machine
# this is the output of RGB_Spectrograms/preprocessing.py or NHRC/preprocessing.py
static_preprocesed = '/home/eric/Engineering/Work/pisces/examples/RGB_Spectrograms/pre_processed_data/stationary/stationary_preprocessed_data_50.npy'

psg_key = 'psg'
psg_str_key = 'psg_str'
specgram_key = 'spectrogram'
activity_key = 'activity'

prepro_item_keys = [specgram_key, 'spec_times', 'spec_freqs', 
                    activity_key, psg_key] # drop these 2 after this prepro
new_prepro_keys = [*(prepro_item_keys[:-2]), psg_str_key]

def int_psg_to_WLDM(psg_ints: np.ndarray) -> str:
    def mapper(s) -> str:
        return {
            0: 'W',
            1: 'L',
            2: 'L',
            3: 'D',
            4: 'D',
            5: 'M',
        }.get(s, '')
    v_mapper = np.vectorize(mapper)
    # convert psg_ints to WLDM
    psg_strs = list(v_mapper(psg_ints))
    return ''.join(psg_strs)

if __name__ == '__main__':

    data = np.load(static_preprocesed, allow_pickle=True).item()

    print(data.keys())
    new_data = {}

    for key in data.keys():
        key_data = data[key]
        key_data.pop(activity_key)
        psg_data = key_data.pop(psg_key)
        psg_str = int_psg_to_WLDM(psg_data[:, 1])
        key_data[psg_str_key] = psg_str
        new_data[key] = key_data
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir / 'stationary_preprocessed_data_WLDM_str.npy', new_data)

