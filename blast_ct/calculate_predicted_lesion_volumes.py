import argparse
import pandas as pd
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

CLASS_NAMES = ['background', 'iph', 'eah', 'oedema', 'ivh']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-csv-path',
                        required=True,
                        type=str,
                        help='Path to the prediction csv file.')
    parse_args, unknown = parser.parse_known_args()
    prediction_csv_path = parse_args.prediction_csv_path
    dataframe = pd.read_csv(prediction_csv_path, index_col='id')
    for id, row in tqdm(dataframe.iterrows()):
        prediction = sitk.ReadImage(row['prediction'])
        voxel_volume_ml = np.prod(prediction.GetSpacing()) / 1000.
        array = sitk.GetArrayFromImage(prediction)
        for i, class_name in enumerate(CLASS_NAMES):
            if i == 0:
                continue
            dataframe.loc['id', f'{class_name:s}_predicted_volume_ml'] = np.sum(prediction == i) * voxel_volume_ml
    dataframe.to_csv(prediction_csv_path)