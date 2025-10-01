"""
Rebuild split data files
Run this script once after cloning the repository
"""

import os
import glob

def rebuild_similarity_matrix():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    parts = sorted(glob.glob(os.path.join(data_dir, 'package_similarity_matrix_10k.pkl.part*')))

    output_file = os.path.join(data_dir, 'package_similarity_matrix_10k.pkl')

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping rebuild.")
        return

    with open(output_file, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())

    print(f"Rebuilt {output_file} from {len(parts)} parts")

if __name__ == '__main__':
    rebuild_similarity_matrix()
