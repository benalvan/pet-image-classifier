# Dog Breed Image Classifier

A Python command-line tool that uses pre-trained CNNs to check whether an image is a dog and, if so, identify its breed. Applies the tool to a realistic use case: verifying entries for a dog show registration system that needs to catch non-dog submissions.

Compares three CNN architectures (ResNet, AlexNet, VGG) on two tasks: detecting dogs vs. non-dogs, and classifying breed, then benchmarks accuracy against runtime to recommend the best fit for the use case.

## Usage

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

| Argument | Default | Description |
|---|---|---|
| `--dir` | `pet_images/` | Folder of images to classify |
| `--arch` | `vgg` | CNN architecture: `resnet`, `alexnet`, or `vgg` |
| `--dogfile` | `dognames.txt` | List of valid dog breed names |

Run all three architectures at once:
```bash
sh run_models_batch.sh
```

## How It Works

`check_images.py` runs a six-stage pipeline:

| Stage | File | Task |
|---|---|---|
| 1 | `get_input_args.py` | Parse `--dir`, `--arch`, `--dogfile` |
| 2 | `get_pet_labels.py` | Extract ground-truth label from filename |
| 3 | `classify_images.py` | Classify image with CNN, compare to label |
| 4 | `adjust_results4_isadog.py` | Flag dog vs. not-dog via `dognames.txt` |
| 5 | `calculates_results_stats.py` | Compute accuracy stats |
| 6 | `print_results.py` | Print summary |

Each image's data is tracked in a dictionary of lists:

```python
results_dic = {
    'Beagle_01141.jpg': ['beagle', 'english foxhound', 0, 1, 1]
}
```

## Results

Tested on 40 images (30 dogs, 10 non-dogs):

| Metric | AlexNet | ResNet | VGG |
|---|---|---|---|
| Correct dog detection | 100% | 100% | **100%** |
| Correct non-dog detection | 100% | 90% | **100%** |
| Correct breed | 80% | 90% | **93.3%** |
| Runtime | ~3s | ~5s | ~21s |

**VGG performed best overall**, but took 4 to 7 times longer than the other two. For a high-volume system, ResNet's 90% breed accuracy at a fraction of the runtime could be the more practical choice.

All three models made the same two breed errors: Great Pyrenees mistaken for Kuvasz, and Beagle mistaken for Walker Hound. These breeds look visually similar, so the mistake likely comes from the images themselves rather than any one model's weakness.

On a custom test (same dog photo, original vs. horizontally flipped), VGG gave different breed predictions for each version while AlexNet and ResNet stayed consistent, a robustness gap the accuracy numbers alone don't show.

## Skills Used

- Command-line interfaces with `argparse`
- String parsing and formatting for label matching
- Nested data structures (dictionary of lists)
- File I/O for structured text (`dognames.txt`)
- CNN inference with pre-trained models (PyTorch/torchvision)
- Runtime benchmarking and accuracy/speed tradeoff analysis

## Project Structure
