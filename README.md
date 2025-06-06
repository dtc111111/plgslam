<p align="center">

  <h1 align="center">PLGSLAM: Progressive Neural Scene Representation with Local to Global Bundle Adjustment</h1>
  <h3 align="center">CVPR 2024</h3>
  <h3 align="center"><a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_PLGSLAM_Progressive_Neural_Scene_Represenation_with_Local_to_Global_Bundle_CVPR_2024_paper.pdf">Paper</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="Fig/framework.png" alt="Logo" width="100%">
  </a>
</p>

## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `plgslam`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash
sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate plgslam
```
If desired, the Open3D package can be installed in the [headless rendering](http://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html) mode. This is useful for running PLGSLAM on a server without a display. We recommend to install from [this commit](https://github.com/isl-org/Open3D/tree/v0.15.1) as we observed bugs in other releases of Open3D.

## Run

### Replica
Download the data as below and the data is saved into the `./Datasets/Replica` folder.
```bash
bash scripts/download_replica.sh
```
and you can run PLGSLAM:
```bash
python -W ignore run.py configs/Replica/room0.yaml
```
The mesh for evaluation is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply`, where the unseen and occluded regions are culled using all frames.


### ScanNet
Please follow the data downloading procedure on [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

<details>
  <summary>[Directory structure of ScanNet (click to expand)]</summary>
  
  DATAROOT is `./Datasets` by default. If a sequence (`sceneXXXX_XX`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── scannet
      └── scans
          └── scene0000_00
              └── frames
                  ├── color
                  │   ├── 0.jpg
                  │   ├── 1.jpg
                  │   ├── ...
                  │   └── ...
                  ├── depth
                  │   ├── 0.png
                  │   ├── 1.png
                  │   ├── ...
                  │   └── ...
                  ├── intrinsic
                  └── pose
                      ├── 0.txt
                      ├── 1.txt
                      ├── ...
                      └── ...

```
</details>

Once the data is downloaded and set up properly, you can run PLGSLAM:
```bash
python -W ignore run.py configs/ScanNet/scene0000.yaml
```
The final mesh is saved as `$OUTPUT_FOLDER/mesh/final_mesh_culled.ply`.

### TUM RGB-D
Download the data as below and the data is saved into the `./Datasets/TUM` folder.
```bash
bash scripts/download_tum.sh
```
and you can run PLGSLAM:
```bash
python -W ignore run.py configs/TUM_RGBD/freiburg1_desk.yaml
```
The final mesh is saved as `$OUTPUT_FOLDER/mesh/final_mesh_culled.ply`.

## Evaluation

### Average Trajectory Error
To evaluate the average trajectory error. Run the command below with the corresponding config file:
```bash
# An example for room0 of Replica
python src/tools/eval_ate.py configs/Replica/room0.yaml
```

### Reconstruction Error
To evaluate the reconstruction error, first download the ground truth Replica meshes and the files that determine the unseen regions.
```bash
bash scripts/download_replica_mesh.sh
```
Then run the `cull_mesh.py` with the following commands to exclude the unseen and occluded regions from evaluation.
```bash
# An example for room0 of Replica
# this code should create a culled mesh named 'room0_culled.ply'
GT_MESH=cull_replica_mesh/room0.ply
python src/tools/cull_mesh.py configs/Replica/room0.yaml --input_mesh $GT_MESH
```

Then run the command below. The 2D metric requires rendering of 1000 depth images, which will take some time. Use `-2d` to enable 2D metric. Use `-3d` to enable 3D metric.
```bash
# An example for room0 of Replica
OUTPUT_FOLDER=output/Replica/room0
GT_MESH=cull_replica_mesh/room0_culled.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply --gt_mesh $GT_MESH -2d -3d
```

## Visualizing PLGSLAM Results
For visualizing the results, we recommend to set `mesh_freq: 40` in [configs/PLGSLAM.yaml](configs/PLGSLAM.yaml) and run PLGSLAM from scratch.

After PLGSLAM is trained, run the following command for visualization.

```bash
python visualizer.py configs/Replica/room0.yaml --output output/Replica/room0 --top_view --save_rendering
```
The result of the visualization will be saved at `output/Replica/room0/vis.mp4`. The green trajectory indicates the ground truth trajectory, and the red one is the trajectory of PLGSLAM.

Note: `mesh_freq: 40` means extracting a mesh every 40 input frames. Since extracting a mesh with a high resolution takes some time, for faster running of PLGSLAM for visualization set `meshing resolution` in [configs/Replica/replica.yaml](configs/Replica/replica.yaml) to a higher value before running PLGSLAM (*e.g.*, 5 cm).

### Visualizer Command line arguments
- `--output $OUTPUT_FOLDER` output folder (overwrite the output folder in the config file)
- `--top_view` set the camera to top view. Otherwise, the camera is set to the first frame of the sequence
- `--save_rendering` save rendering video to `vis.mp4` in the output folder
- `--no_gt_traj` do not show ground truth trajectory

## Related Repositories
We would like to extend our gratitude to the authors of [NICE-SLAM](https://github.com/cvg/nice-slam) for their 
exceptional work. Their code served as a valuable foundation for our own project, and we are appreciative of the 
effort they put into their work.

## Contact
You can contact the author through email: mohammad.johari At idiap.ch.

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{deng2024plgslam,
  title={Plgslam: Progressive neural scene represenation with local to global bundle adjustment},
  author={Deng, Tianchen and Shen, Guole and Qin, Tong and Wang, Jianyu and Zhao, Wentao and Wang, Jingchuan and Wang, Danwei and Chen, Weidong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19657--19666},
  year={2024}
}
```
