# vqa-binary
VQA Binarized

## Quick start
1. Download train/val annotations, questions, and images from VQA website (http://www.visualqa.org/) and put them in one folder (e.g. `~/vqa-data`).
2. Make changes to paths in `prepro.sh` depending on where #1 is stored, and run it (`chmod +x prepro.sh; ./prepro.sh`). If you store it in `~/vqa-data`, you can just run it as it is.
`prepro.sh` runs three scripts in order.
3. To train and test with default settings, run:
```python main.py --train True```
For descriptions about more options, run:
```python main.py --help```
