"""
Rebuild split data files
Run this script once after cloning the repository
"""

import os
import glob

def rebuild_similarity_matrix():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    parts = sorted(glob.glob(os.path.join(data_dir, 'sim.part*')))

    output_gz = os.path.join(data_dir, 'package_similarity_matrix_10k.pkl.gz')
    output_file = os.path.join(data_dir, 'package_similarity_matrix_10k.pkl')

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping rebuild.")
        return

    with open(output_gz, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())

    print(f"Rebuilt {output_gz} from {len(parts)} parts")
    print("Now decompressing...")

    import gzip
    import shutil
    with gzip.open(output_gz, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(output_gz)
    print(f"Decompressed to {output_file}")

if __name__ == '__main__':
    rebuild_similarity_matrix()
