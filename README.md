# Setup

This document explains how to set up a Python environment to run the code in this assignment.

## 1. Python version

Use atleast **Python 3.9** (tested with 3.10.9 and 3.10.12). Check your version with:

```bash
python3 --version
```

In some cases you may have the command `python` instead of `python3`.

If you don't have Python, install it from [python.org](https://www.python.org/downloads/) or via your system's package manager before continuing.
If you have problem installing Python. Try googling for the solution.

## 2. Create a virtual environment

From this directory, create an isolated environment named `cv_assignment2`:

```bash
python3 -m venv cv_assignment2
```

Activate it:

- **Linux / macOS**
  ```bash
  source cv_assignment2/bin/activate
  ```
- **Windows (PowerShell)**
  ```powershell
  cv_assignment2\Scripts\Activate.ps1
  ```

Your shell prompt should now be prefixed with `(cv_assignment2)`. Keep the environment activated for every step below and whenever you run the task scripts.

## 3. Install dependencies

Upgrade `pip` first, then install the pinned packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

| Package | Version | Used for |
| --- | --- | --- |
| `numpy` | 1.26.4 | All the array maths you will write |
| `opencv-python` | 4.7.0.72 | Reading images and resizing the template |
| `matplotlib` | 3.8.4 | Plotting results |

## 4. Verify the setup

Run:

```bash
python -c "import cv2, numpy, matplotlib; print('All good')"
```

If this prints `All good` with no errors, you're ready to run the task scripts.

## 5. Run the tasks

Run them **in order** and from this directory, because each one imports the
one before it:

```bash
python task1.py
python task2.py
python task3.py
python task4.py
```

Each script writes its figures into `plots/`. They will not produce correct
results until you have filled in the functions marked

```python
return ...   # comment this line and write your code for the function
```

## 6. What is in this folder

| File | What it is |
| --- | --- |
| `task1.py` | Filtering, and measuring whether a filter helped |
| `task2.py` | Finding Waldo by sliding the template around |
| `task3.py` | Corners, blobs and descriptors |
| `task4.py` | Matching those descriptors and voting on an answer |
| `imgs/` | The scenes, the Waldo template, and `annotations.json` |
| `plots/` | Created by the scripts |
| `subjective_answers_task1..4.md` | Write your written answers here |
| `assignment2.pdf` | The question sheet, with reading links |

`imgs/annotations.json` holds Waldo's true position in every scene. Use it to
**score** your detector, which is what the `__main__` blocks already do. Using
it *inside* a detector is not a detector.

## 7. Deactivate when done

```bash
deactivate
```
