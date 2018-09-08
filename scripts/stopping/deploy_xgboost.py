#!/usr/bin/env python3
import argparse
import treelite
import os
import subprocess

import zipfile
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)

    args = parser.parse_args()
    model_file = args.model_file
    model_dir = os.path.dirname(model_file)
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    model_out_name = model_name + '-treelite'
    # export xgboost model to treelite format
    model = treelite.Model.load(args.model_file, 'xgboost')
    model_pkg = os.path.join(model_dir, model_out_name + '.zip')
    model.export_srcpkg(platform='unix', toolchain='gcc', pkgpath=model_pkg,
                        libname=model_out_name + '.so', verbose=True)

    # # save treelite runtime
    # treelite.save_runtime_package(destdir=model_dir)
    # # unzip packaged treelite runtime
    # with zipfile.ZipFile(os.path.join(model_dir, 'treelite_runtime.zip'), "r") as zip_ref:
    #     zip_ref.extractall(model_dir)
    # print('treelite runtime saved to:', os.path.join(model_dir, 'runtime'))

    # unzip packaged model
    with zipfile.ZipFile(model_pkg, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    # compile exported treelite model
    subprocess.call('make', cwd=os.path.join(model_dir, model_out_name))

    import treelite.runtime

    X = np.array([0.0, 5685.62158203125, 15.6940155, 0, 0])
    model_so_path = os.path.join(model_dir, model_out_name, model_out_name + '.so')
    predictor = treelite.runtime.Predictor(model_so_path, verbose=True)
    prob = predictor.predict_instance(X)
    print(prob)
