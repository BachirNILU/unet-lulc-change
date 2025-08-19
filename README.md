# Unet ML model for Land Cover (LULC) change detection

A small ```CLI``` to produce a binary change mask from two co-registered GeoTIFFs (earlier  ```t1 ```, later  ```t2 ```).
It loads a Lightning ```.ckpt``` checkpoint, handles band reordering, optional histogram matching, sliding window inference with cosine blending, and writes both a GeoTIFF and a PNG mask.

**Important:** best.ckpt is larger than 90 MP it has been uploaded here: [Trained model](https://nilu365-my.sharepoint.com/:u:/g/personal/bbel_nilu_no/ETgCjFkfxdFCgZ_3rPTnDH0BzSCZdA-1G625G_8kPtp2KA?e=pkPKSy)

## Quick start
```
python -m pip install --upgrade pip
pip install torch rioxarray xarray rasterio scikit-image imageio
 ```
 ```
python infer_change.py 2018-IR_LUXEMBOURG_1m_T20.tif 2021-IR_LUXEMBOURG_1m_T20.tif  --ckpt best.ckpt --stats stats.json --out outputs_result  --diff-mode auto --band-order 1,2,0 --threshold 0.30 --min-blob 0 --stride 128 --no-prob
 ```
 Use the same ```stats.json``` from training so normalization matches what the checkpoint expects.

 ## Arguments
```t1``` ```t2```
Paths to earlier and later GeoTIFFs. Order matters. Pass 2018 first, then 2021 or 2024. The script aligns t1 to the t2 grid.

- ```--ckpt PATH```
Trained Lightning checkpoint, for example ```best.ckpt```.

- ```--stats PATH```
Per-channel normalization stats from training. For a 12-channel model the order is ```[t1 RGB, t2 RGB, (t2−t1) RGB, |t2−t1| RGB]```.

- ```--out DIR```
Output directory.

- ```--diff-mode {auto|wrap|float}```
How to build the difference channels.

  - ```auto``` inspects ```stats.json```. If it detects historical uint16 wrapping in diff channels, it reproduces it for compatibility. Otherwise it uses float diffs.

  - ```wrap``` forces wrapped-uint16 behavior.

  - ```float``` uses true signed float diffs.

- ```--band-order a,b,c```
Indices (0-based) that tell the script how to form ```[R,G,B]``` from the source.
Examples: ```CIR [NIR,R,G] -> 1,2,0. RGB [R,G,B] -> 0,1,2```.

- ```--threshold FLOAT```
Probability threshold for the change class. Start in the 0.25 to 0.45 range.

- ```--min-blob INT```
Remove components smaller than this number of pixels. Set 0 to disable. For 1 m data, 4 to 16 is a gentle cleanup.

- ```--stride INT```
Window stride in pixels. 128 reduces seam artifacts. 256 is faster.

- ```--no-prob```
Do not save the probability GeoTIFF.

### Others
- ```--patch-size``` 512 Inference window size.

- ```--no-hist-match``` Disable histogram matching of ```t1``` to ```t2``` (enabled by default).

- ```--tta {none|flip|d4}``` Optional test-time augmentation.

- ```--sweep``` 0.22,0.26,0.30 Save masks for multiple thresholds in one run.
 
## Output

### Output Example:
Change between 2018 and 2021 (1m resolution):





<img width="629" height="631" alt="image" src="https://github.com/user-attachments/assets/47730b9a-26f2-4726-a5af-3d5d9a44e3a6" />
